"""Literal-table allocation must not collide with new temporal buffers."""
import re
import struct
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from fsr411_msl import buffer_integer_constants

class BufferSlotTests(unittest.TestCase):
    def source(self,occupied):
        args=', '.join('device uint *b%d [[buffer(%d)]]'%(i,i) for i in occupied)
        return ('constant spvUnsafeArray<uint, 4> weights = spvUnsafeArray<uint, 4>({ 1u, 2u, 3u, 4u });\n'
                'kernel void test('+args+') { uint value=weights[0]; b0[0]=value; }\n')
    def test_default_slot_retains_existing_abi(self):
        source,data=buffer_integer_constants(self.source([0,1,2]))
        self.assertIn('Fsr411Tables [[buffer(7)]]',source)
        self.assertEqual(data,struct.pack('<4I',1,2,3,4))
    def test_temporal_current_at_seven_moves_tables_to_eight(self):
        source,data=buffer_integer_constants(self.source(list(range(8))))
        self.assertIn('Fsr411Tables [[buffer(8)]]',source)
        self.assertEqual(len(re.findall(r'\[\[buffer\(7\)\]\]',source)),1)
    def test_no_free_supported_slot_fails_closed(self):
        with self.assertRaises(ValueError):buffer_integer_constants(self.source(list(range(16))))
if __name__=='__main__':unittest.main()
