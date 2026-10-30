#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""FSR 4.1.2 Lossy: the weight decoder and the depth-edge speck rule.

The decoder test runs on a synthetic DXIL listing in the layout of a trained
pass, so it needs neither the release DLL nor DXC. The shader test checks
that the speck rule is wired only into the lossy2 POST and that plain lossy
POST keeps its previous source.
"""

import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from fsr411 import lossy412_weights as decoder  # noqa: E402
from fsr411.lossy_shaders import apply_lossy_post  # noqa: E402


def pack(values):
    word = 0
    for byte, value in enumerate(values):
        word |= (value & 255) << (8 * byte)
    return word - (1 << 32) if word & 0x80000000 else word


def synthetic_cnb(seed=1):
    """A DXIL-like listing of pass 2 (CNB, 1080 tier) with random weights."""
    rng = random.Random(seed)
    A = [[[[rng.randint(-128, 127) for _ in range(3)] for _ in range(3)] for _ in range(16)] for _ in range(16)]
    Bw = [[rng.randint(-128, 127) for _ in range(16)] for _ in range(32)]
    Cw = [[rng.randint(-128, 127) for _ in range(32)] for _ in range(16)]
    bA, bB, bC = ([rng.randint(-900, 900) for _ in range(n)] for n in (16, 32, 16))
    base, row = 8342464, 15392
    lines, counter = [], [0]

    def value(rhs):
        counter[0] += 1
        lines.append("  %%%d = %s" % (counter[0], rhs))
        return "%%%d" % counter[0]

    words = {}
    for ky in range(3):
        for kx in range(3):
            for w in range(4):
                handle = value("call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, "
                               "%dx.types.Handle %h, i32 " + str(base + ky * row + kx * 16 + 4 * w) +
                               ", i32 undef, i8 1, i32 4)")
                words[(ky, kx, w)] = value("extractvalue %dx.types.ResRet.i32 " + handle + ", 0")

    def chain(bias, terms):
        # Like DXC: the bias joins the first product, then the requantize
        # adds the rounding constant 63 and shifts right by 7.
        acc = None
        for word, literal in terms:
            dot = value("call i32 @dx.op.dot4AddPacked.i32(i32 163, i32 0, i32 %s, i32 %d)" % (word, literal))
            acc = value("add i32 %s, %d" % (dot, bias)) if acc is None else value("add i32 %s, %s" % (acc, dot))
        acc = value("add i32 %s, 63" % acc)
        acc = value("ashr i32 %s, 7" % acc)
        acc = value("call i32 @dx.op.binary.i32(i32 37, i32 %s, i32 -128)" % acc)
        return value("call i32 @dx.op.binary.i32(i32 38, i32 %s, i32 127)" % acc)

    def pack4(sums):
        return value("call i32 @dx.op.pack4x8.i32(i32 220, i8 2, " + ", ".join("i32 " + s for s in sums) + ")")

    a_sums = [chain(bA[c], [(words[(ky, kx, w)], pack([A[c][4 * w + j][ky][kx] for j in range(4)]))
                            for ky in range(3) for kx in range(3) for w in range(4)]) for c in range(16)]
    a_words = [pack4(a_sums[4 * q:4 * q + 4]) for q in range(4)]
    b_sums = [chain(bB[c], [(a_words[g], pack(Bw[c][4 * g:4 * g + 4])) for g in range(4)]) for c in range(32)]
    b_words = [pack4(b_sums[4 * g:4 * g + 4]) for g in range(8)]
    c_sums = [chain(bC[c], [(b_words[g], pack(Cw[c][4 * g:4 * g + 4])) for g in range(8)]) for c in range(16)]
    c_words = [pack4(c_sums[4 * w:4 * w + 4]) for w in range(4)]
    lines.append("  call void @dx.op.rawBufferStore.i32(i32 140, %dx.types.Handle %h, i32 33304416, i32 undef, "
                 + ", ".join("i32 " + w for w in c_words) + ", i8 15, i32 16)")
    return "\n".join(lines), dict(A=A, bA=bA, Bw=Bw, bB=bB, Cw=Cw, bC=bC)


class Decoder(unittest.TestCase):
    def test_synthetic_pass_round_trips(self):
        listing, expected = synthetic_cnb()
        logical, tier, weights = decoder.decode_pass(listing)
        self.assertEqual((logical, tier), (2, 1080))
        self.assertEqual(weights, expected)

    def test_npz_is_byte_reproducible(self):
        _, weights = synthetic_cnb(seed=7)
        arrays = {"p2_" + name: weights[name] for name in decoder.ARRAYS}
        import tempfile
        with tempfile.TemporaryDirectory() as work:
            first = decoder.write_npz(arrays, Path(work) / "a.npz")
            second = decoder.write_npz(arrays, Path(work) / "b.npz")
            self.assertEqual(first, second)
            self.assertEqual(decoder.read_npz(Path(work) / "a.npz"), arrays)

    def test_not_a_trained_pass(self):
        with self.assertRaises(ValueError):
            decoder.decode_pass("  %1 = add i32 1, 2")


class SpeckRule(unittest.TestCase):
    SOURCE = ROOT / "third_party/fsr411/shaders/reimpl411/post_rev_411.hlsl"

    def test_only_lossy2_post_carries_the_rule(self):
        source = self.SOURCE.read_text()
        plain = apply_lossy_post(source, 1080)
        speck = apply_lossy_post(source, 1080, speck=True)
        self.assertIn("static const bool FSR411_LOSSY_SPECK = false;", plain)
        self.assertIn("static const bool FSR411_LOSSY_SPECK = true;", speck)
        self.assertEqual(plain.replace("speck=0", "speck=1").replace("SPECK = false", "SPECK = true"), speck)
        self.assertIn("res = Fsr411LossySpeck(", speck)
        self.assertEqual(apply_lossy_post(speck, 1080, speck=True), speck)
        with self.assertRaises(ValueError):
            apply_lossy_post(plain, 1080, speck=True)


if __name__ == "__main__":
    unittest.main()
