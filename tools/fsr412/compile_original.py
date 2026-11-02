#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compile an opt-in original Lite Lossy bundle from verified external assets.

No downloads, installs, DLL execution or approximate temporal rewrites. The
full original t18 and translated shaders stay in ignored build/ or system temp.
The runtime substitutes only independently validated integer neural operators.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(HERE))
from fsr412.model_data import load_lossy_initializer
from upstream_oracle import LOSSY_SHA256, external_output, require, sha256
from oracle_bundle import build_bundle, compile_source
import original_post_cnb
from oracle_selection import classify, select

# 2: corrected auto-exposure classification (AMD AUTOEXPOSURE_ENABLED=1 has
# no r_input_exposure); revision 1 bundles selected the opposite PRE/POST.
# 3: optional run-frame POST with native-precomputed CNB (post_fast).
REVISION = 3


def add_post_fast(out, manifest):
    """Run-frame POST that reads native-precomputed CNB cells.

    Only for the one isolated POST the rewrite is pinned and GPU-proven for
    (docs/FSR412_ORIGINAL_POST_CNB.md); other permutations keep the original.
    Skipped frames always use the unchanged isolated POST.
    """
    post = manifest['kernels']['post']
    if post['dxil_sha256'] != original_post_cnb.IDENTITY or manifest.get('post_mode') != 'isolated':
        return
    source = (out / post['source']).read_text()
    if sha256(source.encode()) != original_post_cnb.SOURCE_SHA256:
        return
    text, receipt = original_post_cnb.rewrite(source)
    stem = original_post_cnb.IDENTITY + '.precomputed'
    (out / (stem + '.metal')).write_text(text)
    (out / (stem + '.json')).write_text(json.dumps(receipt, indent=2) + '\n')
    row = {k: v for k, v in post.items() if k not in ('compile_command', 'library_sha256', 'optimization_profile')}
    row.update(library=stem + '.metallib', source=stem + '.metal', metal_sha256=receipt['metal_sha256'],
               precomputed_base=receipt['precomputed_base'], precomputed_stride=receipt['precomputed_stride'],
               execute_policy=receipt['production_execute_policy'])
    row.update(compile_source(out, stem))
    manifest['kernels']['post_fast'] = row
    manifest['scratch_allocation_bytes'] = receipt['physical_scratch_bytes']


def bundle_key(tier, depth_inverted, low_res_mv, jittered_mv, colorspace, auto_exposure, ultra_performance):
    return (f'fsr412_original_{tier}_di{depth_inverted}_lm{low_res_mv}_jm{jittered_mv}'
            f'_cs{colorspace}_ae{auto_exposure}_up{ultra_performance}_r{REVISION}')


def converter_path(value):
    candidate = value or os.environ.get('FSR412_DXIL_SPIRV') or shutil.which('dxil-spirv')
    path = Path(candidate).resolve() if candidate else None
    require(path is not None and path.is_file() and os.access(path, os.X_OK),
            'set FSR412_DXIL_SPIRV to an existing dxil-spirv converter; no converter is installed automatically')
    return path


def ready(directory, manifest, fingerprint):
    if (manifest.get('schema') != 'fsr412_original_runtime_v1' or
            manifest.get('original_revision') != REVISION or manifest.get('source_hash') != fingerprint):
        return False
    for entry in manifest.get('kernels', {}).values():
        for name, expected in ((entry.get('source'), entry.get('metal_sha256')),
                               (entry.get('library'), entry.get('library_sha256'))):
            if not name or not expected:
                return False
            path = directory / name
            if not path.is_file() or sha256(path.read_bytes()) != expected:
                return False
        if entry.get('literal_tables') and not (directory / entry['literal_tables']).is_file():
            return False
    initializer = directory / manifest.get('initializers', '')
    return (initializer.is_file() and len(initializer.read_bytes()) == 131072 and
            sha256(initializer.read_bytes()) == manifest.get('initializers_sha256') and
            all(stage in manifest.get('kernels', {}) for stage in ('pre', 'pass1', 'pass1_border', 'pass2', 'border', 'pass3', 'post', 'ae', 'ae_finalize', 'rcas')))


def build(root, builddir, assets, out, options, converter_argument, cross):
    root, builddir, assets = Path(root).resolve(), Path(builddir).resolve(), Path(assets).resolve()
    out = external_output(Path(out))
    require((assets / 'inventory.json').is_file(),
            'original shader inventory is missing; run tools/fsr412/upstream_oracle.py with the pinned DLL and reference provider first')
    _, trained, _ = load_lossy_initializer(root)
    index = classify(assets)
    selected = select(index, options)
    require(selected['source_dll_sha256'] == LOSSY_SHA256, 'original source identity changed')
    cross = Path(cross).resolve()
    require(cross.is_file(), 'existing spirv-cross is missing')
    # Shared helpers have no Lite neural coefficients. They keep the established
    # exposure/RCAS integration ABI; original PRE/POST themselves are unchanged.
    from fsr412.compile_fsr412 import support_bundle
    support_dir, support = support_bundle(root, builddir, options['colorspace'], options['auto_exposure'])
    digest = hashlib.sha256(json.dumps({'options': options, 'revision': REVISION}, sort_keys=True).encode())
    for path in (Path(__file__), HERE / 'oracle_bundle.py', HERE / 'oracle_selection.py',
                 HERE / 'oracle_variants.py', HERE / 'oracle_pass1_border.py', HERE / 'oracle_io.py',
                 HERE / 'upstream_oracle.py', HERE / 'original_post_cnb.py', assets / 'inventory.json', assets / selected['initializer_file']):
        digest.update(path.read_bytes())
    digest.update(trained)
    for row in selected['kernels'].values():
        for suffix in ('.dxil', '.ll'):
            digest.update((assets / (row['dxil_sha256'] + suffix)).read_bytes())
    for name in ('ae', 'ae_finalize', 'rcas'):
        row = support['kernels'][name]
        digest.update(json.dumps(row, sort_keys=True).encode())
        digest.update((support_dir / row['library']).read_bytes())
    from fsr412.compile_fsr412 import run
    digest.update(run(['xcrun', 'metal', '--version']))
    fingerprint = digest.hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.original-compile.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest_path = out / 'bindings.json'
        if manifest_path.is_file():
            try:
                previous = json.loads(manifest_path.read_text())
                if ready(out, previous, fingerprint):
                    return previous, True
            except (OSError, ValueError, TypeError, KeyError):
                pass
        # Resolve the converter only for a real rebuild. Cached bundles are
        # validated by source/output identities, not by converter presence.
        converter = converter_path(converter_argument)
        manifest = build_bundle(assets, out, selected, converter, cross, True, 'isolated')
        for name in ('ae', 'ae_finalize', 'rcas'):
            row = dict(support['kernels'][name])
            library = 'support_' + name + '.metallib'
            shutil.copyfile(support_dir / row['library'], out / library)
            row.update(library=library, library_sha256=sha256((out / library).read_bytes()))
            # Support sources are external imported data too. Preserve identities
            # so interrupted/self-consistent stale cache cannot pass readiness.
            original_source = support_dir / (name + '.metal')
            require(original_source.is_file(), 'missing support source for ' + name)
            source = 'support_' + name + '.metal'
            shutil.copyfile(original_source, out / source)
            row.update(source=source, metal_sha256=sha256((out / source).read_bytes()))
            if row.get('literal_tables'):
                tables = 'support_' + name + '.bin'
                shutil.copyfile(support_dir / row['literal_tables'], out / tables)
                row['literal_tables'] = tables
            manifest['kernels'][name] = row
        add_post_fast(out, manifest)
        native_hash = sha256(trained) if options['preset'] == 'normal' and options['tier'] <= 2160 else sha256((out / manifest['initializers']).read_bytes()[:27392])
        manifest.update(schema='fsr412_original_runtime_v1', version='4.1.2', model='int8-lite-lossy',
                        original_revision=REVISION, source_hash=fingerprint, native_initializer_sha256=native_hash,
                        converter_sha256=sha256(converter.read_bytes()), spirv_cross_sha256=sha256(cross.read_bytes()),
                        prepared_color=False, external_post_cnb=False, lossy_cache_bytes_per_pixel=0,
                        shader_graph='original PRE/P1/P2/border/P3/POST; integer-only native substitution',
                        windows_runtime_parity_verified=False)
        temporary = out / ('.bindings.' + str(os.getpid()) + '.json')
        temporary.write_text(json.dumps(manifest, indent=2) + '\n')
        temporary.replace(manifest_path)
        require(ready(out, manifest, fingerprint), 'compiled original bundle failed its own completeness/identity checks')
        return manifest, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--builddir', type=Path, default=ROOT / 'build')
    parser.add_argument('--assets', type=Path)
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--tier', type=int, choices=(1080, 2160, 4320), default=2160)
    parser.add_argument('--ultra-performance', type=int, choices=(0, 1), default=0)
    for flag, default in (('depth-inverted', 1), ('low-res-mv', 1), ('jittered-mv', 0), ('auto-exposure', 0)):
        parser.add_argument('--' + flag, type=int, choices=(0, 1), default=default)
    parser.add_argument('--colorspace', type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument('--converter', type=Path)
    parser.add_argument('--spirv-cross', type=Path, default=ROOT / 'tools/bin/spirv-cross')
    args = parser.parse_args()
    options = dict(tier=args.tier, preset='ultraperformance' if args.ultra_performance else 'normal',
                   depth_inverted=args.depth_inverted, low_res_mv=args.low_res_mv, jittered_mv=args.jittered_mv,
                   colorspace=args.colorspace, auto_exposure=args.auto_exposure)
    key = bundle_key(args.tier, args.depth_inverted, args.low_res_mv, args.jittered_mv,
                     args.colorspace, args.auto_exposure, args.ultra_performance)
    out = args.outdir or args.builddir / 'fsr412_upstream/compiled' / key
    manifest, cached = build(args.root, args.builddir, args.assets or args.root / 'build/fsr412_upstream',
                             out, options, args.converter, args.spirv_cross)
    print(json.dumps({'dir': str(out), 'key': key, 'version': manifest['version'], 'cached': cached, 'original_revision': REVISION}))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit('Original Lite Lossy compile failed: ' + str(error))
