"""Verify the 4.0.2 overlays against the previous full headers, without a GPU."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from compile_variant import INT8_FUSED_HEADER, build_shadow_tree


class ShaderOverlayTests(unittest.TestCase):
    def test_headers_for_both_loop_modes(self):
        # PRE FP16 keeps the original hints even when fused passes use rolled loops.
        hashes = {
            "pre_common_f16.hlsli": ("75db0567477e513bc6db8cf05951bf20a5d7d0ad4b3b08ea84a5e862ea780886",) * 2,
            "post_common.hlsli": (
                "164eb7ba8ceda132a549c4fa968f7e15af8ba76f9f52063a306b0e99d7f2d362",
                "0d09731c723d6ac2c7ae060ee1d3bb510a4ef29c1a22dd5b85764afcb6c22a4e",
            ),
            INT8_FUSED_HEADER: (
                "1c0e4239f5303724bd6c2747d91a2b1493e6057aaf5531d93dbf02e417a6e5c6",
                "0db6890e791560a16d62c52ab758dd1fb236f7f76ebf9aac0147d77621dedf7f",
            ),
        }
        with tempfile.TemporaryDirectory() as cache:
            for keep_unroll in (False, True):
                tree = Path(build_shadow_tree(str(ROOT), cache, keep_unroll))
                for name, expected in hashes.items():
                    with self.subTest(name=name, keep_unroll=keep_unroll):
                        digest = hashlib.sha256((tree / name).read_bytes()).hexdigest()
                        self.assertEqual(digest, expected[keep_unroll])
