# SPDX-License-Identifier: MIT
"""CPU-only contracts of the opt-in int8-lite-lossy-original runtime route.

No GPU, DLL or converter is required. GPU parity is a separate harness:
tools/bench/fsr412_original_runtime_ab.mm (see docs/FSR412_LOSSY.md).
"""
import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/fsr412'))
sys.path.insert(0, str(ROOT / 'tools'))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


gate = module('fsr412_original_neural', ROOT / 'tools/fsr412/original_neural.py')
import upstream_oracle as oracle  # dataclasses need a registered module
import compile_original as compiler
SESSION = (ROOT / 'src/fsr412_original_session.inc').read_text()
RUNTIME = (ROOT / 'src/fsr411_runtime.h').read_text()


class NeuralGateTests(unittest.TestCase):
    def test_every_integer_kernel_returns_on_original_skip_before_barriers(self):
        for name in ('mpp', 'simd', 'reference'):
            source = (ROOT / f'src/fsr411_{name}.metal').read_text()
            gated = gate.apply_original_gate(source)
            kernels = [m[1] for m in re.finditer(r'\bkernel\s+void\s+(\w+)\s*\(', source)
                       if m[1] not in ('fsr411_mpp_chain_probe', 'fsr411_simd_probe')]
            self.assertEqual(gated.count('if (fsr412OriginalSkipped(scratch)) return;'), len(kernels), name)
            self.assertEqual(gate.apply_original_gate(gated), gated, 'not idempotent: ' + name)
            for probe in ('fsr411_mpp_chain_probe', 'fsr411_simd_probe'):
                if probe in source:
                    body = gated[gated.index('kernel void ' + probe):]
                    self.assertNotIn('fsr412OriginalSkipped', body[:body.index('{') + 80], probe)

    def test_marker_offset_and_words_match_original_geometry(self):
        for tier in (1080, 2160):
            g = oracle.scratch_geometry(tier)
            width, height = tier * 8 // 9, tier // 2
            self.assertEqual((width + 2) * (height + 2) * 16 + 128, g['skip_marker_byte_offset'])
        self.assertIsNone(oracle.scratch_geometry(4320)['skip_marker_byte_offset'])
        self.assertIn('1511506142u', gate.HELPER)
        self.assertIn('168889943u', gate.HELPER)
        self.assertIn('#if FSR411_MAXH <= 1080', gate.HELPER)

    def test_unknown_or_collided_sources_fail_closed(self):
        with self.assertRaises(ValueError):
            gate.apply_original_gate('kernel void x(device uchar *scratch [[buffer(0)]]) {}')
        with self.assertRaises(ValueError):
            gate.apply_original_gate('#include "fsr411_model.h"\nint fsr412OriginalSkipped;')


class SkipPredicateTests(unittest.TestCase):
    """Runtime CPU prediction must equal the PRE predicate (GPU marker is authoritative)."""

    @staticmethod
    def original(jx, jy, reset, rw, w):
        import struct
        x = struct.unpack('<I', struct.pack('<f', jx))[0]
        y = struct.unpack('<I', struct.pack('<f', jy))[0]
        nx = (x >> 31) and (x & 0x7fffffff) != 0
        ny = (y >> 31) and (y & 0x7fffffff) > 1042983597
        return bool(nx and ((rw << 1) >= w or not ny) and reset == 0)

    def test_predicate_constants_are_the_released_ones(self):
        self.assertIn('1042983597u', SESSION)
        self.assertIn('uint32_t(p.rw<<1u)>=p.alignedUW', SESSION)
        pre = ROOT / 'build/fsr412_upstream/9523e223136343d56c1c3b2c89fbdea29adf511343bfcb7fb35ba7afbead4bba.metal'
        if pre.is_file():
            text = pre.read_text()
            self.assertIn('1042983597u', text)
            self.assertIn('(_62.w << 1u) >= _62.x', text)

    def test_reference_cases(self):
        self.assertFalse(self.original(-0.0, 0.1, 0, 384, 640))
        self.assertTrue(self.original(-0.25, -1 / 6, 0, 384, 640))
        self.assertFalse(self.original(-0.25, -0.2, 0, 300, 640))
        self.assertTrue(self.original(-0.25, -0.1, 0, 300, 640))
        self.assertFalse(self.original(-0.25, 0.1, 1, 384, 640))


class BundleContractTests(unittest.TestCase):
    def test_bundle_key_and_revision_match_runtime(self):
        revision = int(re.search(r'kOriginalRevision = (\d+);', SESSION)[1])
        self.assertEqual(revision, compiler.REVISION)
        self.assertEqual(compiler.bundle_key(2160, 1, 1, 0, 0, 1, 0),
                         f'fsr412_original_2160_di1_lm1_jm0_cs0_ae1_up0_r{revision}')
        self.assertIn('"_r"+std::to_string(kOriginalRevision)', SESSION)

    def test_ready_rejects_stale_or_self_consistent_but_wrong_bundles(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / 'a.metal').write_text('x'); (out / 'a.metallib').write_bytes(b'y')
            (out / 'init.bin').write_bytes(bytes(131072))
            kernel = {'source': 'a.metal', 'metal_sha256': oracle.sha256(b'x'),
                      'library': 'a.metallib', 'library_sha256': oracle.sha256(b'y')}
            names = ('pre', 'pass1', 'pass1_border', 'pass2', 'border', 'pass3', 'post', 'ae', 'ae_finalize', 'rcas')
            manifest = {'schema': 'fsr412_original_runtime_v1', 'original_revision': compiler.REVISION,
                        'source_hash': 'h', 'kernels': {n: dict(kernel) for n in names},
                        'initializers': 'init.bin', 'initializers_sha256': oracle.sha256(bytes(131072))}
            self.assertTrue(compiler.ready(out, manifest, 'h'))
            self.assertFalse(compiler.ready(out, manifest, 'other-source'))
            self.assertFalse(compiler.ready(out, {**manifest, 'original_revision': compiler.REVISION - 1}, 'h'))
            broken = json.loads(json.dumps(manifest)); broken['kernels']['post']['library_sha256'] = '0' * 64
            self.assertFalse(compiler.ready(out, broken, 'h'))
            missing = json.loads(json.dumps(manifest)); del missing['kernels']['pass1_border']
            self.assertFalse(compiler.ready(out, missing, 'h'))

    def test_assets_never_leave_external_build_paths(self):
        with self.assertRaises(ValueError):
            oracle.external_output(ROOT / 'src/original')


class RuntimeContractTests(unittest.TestCase):
    def test_route_is_opt_in_and_fail_closed(self):
        self.assertIn('if(o.originalLossy) return ensure_original_session(tier,p,o,kind,error);',
                      (ROOT / 'src/fsr412_session.inc').read_text())
        self.assertIn('bool originalLossy = false;', RUNTIME)
        for text in ('original Lite Lossy bundle does not match this stream or revision',
                     'original Lite t18 initializer identity mismatch',
                     'original Lite library identity mismatch',
                     'c77c2d1a4d119ed76d94b8c23335490cc38280db3346c72396a1e1acb0523cfc'):
            self.assertIn(text, SESSION)

    def test_native_dimensions_follow_the_original_geometry_contract(self):
        body = SESSION[SESSION.index('static void fill_original_native_dimensions'):]
        body = body[:body.index('\n}\n')]
        self.assertIn('if(pass==1) {set(p.input,g.classHalf);set(p.output,g.CNBwritten);}', body)
        self.assertIn('else if(pass==4) {set(p.input,g.classHalf);set(p.output,g.DSwritten);}', body)
        self.assertIn('else if(pass==2) {set(p.input,g.classQuarter);set(p.output,g.actualQuarter);}', body)
        self.assertIn('set(p.input,g.classQuarter);set(p.output,g.actualHalf);set(p.skip,g.actualHalf);', body)
        self.assertIn('134217728u', SESSION)
        self.assertIn('P2sentinelInAllocationNeedsClear', SESSION)

    def test_isolated_post_copies_bracket_post(self):
        post = SESSION[SESSION.index('if(name=="post") {'):]
        post = post[:post.index('return;\n    }')]
        first, dispatch, last = (post.index(t) for t in ('"originalCacheIn"', 'dispatch(enc,m4,slot,s,fast?"post_fast":"post"', '"originalCacheOut"'))
        self.assertLess(first, dispatch)
        self.assertLess(dispatch, last)

    def test_fast_post_runs_only_on_run_frames_with_retargeted_native_cnb(self):
        post = SESSION[SESSION.index('if(name=="post") {'):]
        post = post[:post.index('return;\n    }')]
        self.assertIn('const bool fast=s.originalPostFast && !skip;', post)
        self.assertIn('if(fast) dispatch_native(enc,m4,slot,s,13,v,sizes);', post)
        self.assertIn('fast?"post_fast":"post"', post)
        self.assertIn('else if(pass==13) {set(p.input,g.actualHalf);set(p.output,g.actualHalf);}', SESSION)
        self.assertIn('q.input.offset=uint32_t(2*geometry.P+geometry.halfRowBytes+16u)', SESSION)
        self.assertIn('original POST CNB needs conditional first-tap biases', SESSION)
        import original_post_cnb
        self.assertEqual(original_post_cnb.DEFAULT_BASE, 99821056 + 33273664)
        self.assertEqual(original_post_cnb.DEFAULT_STRIDE, 1920 * 16)

    def test_history_is_double_buffered_like_the_oracle(self):
        self.assertIn('if(name=="rw_history_color") return b.originalLossy?b.view->historyNext:b.view->history;', RUNTIME)
        self.assertIn('if(name=="rw_recurrent_0") return b.originalLossy?b.view->recurrentNext:b.view->recurrent;', RUNTIME)
        self.assertIn('std::swap(view.history,view.historyNext);', RUNTIME)


if __name__ == '__main__':
    unittest.main()
