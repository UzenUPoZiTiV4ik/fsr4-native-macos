"""Pins, real AMD table mapping and complete subpixel heads, without a GPU."""
import copy
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from fsr412 import model_data
from fsr412.compile_fsr412 import pick,source_for,external_post_cnb
from fsr412.lossy_shaders import apply_lossy_post

@unittest.skipUnless((ROOT/'build/fsr412_lossy/import.json').is_file(),'fetch the optional Lite Lossy data first')
class ModelDataTests(unittest.TestCase):
    def setUp(self):
        self.path,self.data,self.receipt=model_data.load_lossy_initializer(ROOT)
        self.base=(ROOT/'build/fsr412_model/initializers-lite.bin').read_bytes()
    def test_pinned_initializer_and_head_order(self):
        matrix=model_data.validate_lossy_data(self.data,self.receipt)
        self.assertEqual(tuple(matrix[0]),struct.unpack_from('<8i',self.data,27264))
    def test_self_consistent_replacement_is_not_the_pin(self):
        data=bytearray(self.data);data[1152]^=1
        receipt=copy.deepcopy(self.receipt);receipt['initializers'][model_data.FILE]=model_data.sha(data)
        with self.assertRaises(ValueError):model_data.validate_lossy_data(data,receipt)
    def test_subpixel_override_tampering_rejected(self):
        receipt=copy.deepcopy(self.receipt);receipt['post_ct_biases_by_tap'][1][0]+=1
        with self.assertRaises(ValueError):model_data.validate_lossy_data(self.data,receipt)
    def test_malformed_receipts_fail_cleanly(self):
        for bad in (None,[],{},dict(self.receipt,initializers=[])):
            with self.assertRaises(ValueError):model_data.validate_lossy_data(self.data,bad)
    def test_up_and_4320_use_the_documented_base(self):
        for tier,preset in ((4320,'normal'),(1080,'ultraperformance')):
            path,data,trained=model_data.initializer_for(ROOT,tier,preset,'int8-lite-lossy')
            self.assertFalse(trained);self.assertNotEqual(path,self.path)
    def test_four_subpixel_heads_are_generated_not_dropped(self):
        rows=json.loads((ROOT/'tools/fsr412/shader_catalog.json').read_text())['shaders']
        row=pick(rows,'post',1080,{'cs':0,'ae':0,'up':0})
        source=external_post_cnb(source_for(row).read_text(),1080)
        self.assertEqual(model_data.patch_post_tail(source,self.base,self.base),source)
        primary=model_data.patch_post_tail(source,self.base,self.data)
        cached=apply_lossy_post(primary,1080)
        result=model_data.patch_post_biases_by_tap(cached,self.data,self.receipt)
        self.assertEqual(result.count('constant int4 Fsr412TapBias'),2)
        self.assertIn('= Fsr412TapBiasFilter[',result)
        self.assertIn('= Fsr412TapBiasRecurrent[',result)
        self.assertNotEqual(primary,source)
        with self.assertRaises(ValueError):model_data.patch_post_biases_by_tap(result,self.data,self.receipt)
if __name__=='__main__':unittest.main()
