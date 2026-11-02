"""CPU evidence for dropping temporal target clears only on reset frames."""
from pathlib import Path
import re
import unittest

ROOT=Path(__file__).resolve().parents[1]


def block(source,marker):
    start=source.index(marker)
    opening=source.index("{",start)
    depth=1
    for end in range(opening+1,len(source)):
        depth+=(source[end]=="{")-(source[end]=="}")
        if depth==0:
            return source[opening+1:end]
    raise AssertionError("unclosed source block")


class ResetClearTests(unittest.TestCase):
    def test_all_imported_pre_temporal_reads_are_reset_guarded(self):
        sources=list((ROOT/"third_party/fsr411/shaders").glob("*/pre_rev_411*.hlsl"))
        self.assertEqual(len(sources),3)
        for path in sources:
            with self.subTest(path=path):
                source=path.read_text()
                main=block(source,"void fsr4_model_v07_fp8_no_scale_prepass(")
                guarded=block(main,"if (disocclusion == 0 && reset == 0)")
                self.assertEqual(guarded.count("SampleTextureCatmullRom(history_uv"),1)
                self.assertEqual(main.count("SampleTextureCatmullRom(history_uv"),1)
                self.assertEqual(guarded.count("r_recurrent_0.SampleLevel("),1)
                self.assertEqual(source.count("r_recurrent_0.SampleLevel("),1)
                catmull=block(source,"bool SampleTextureCatmullRom(")
                self.assertEqual(catmull.count("r_history_color.SampleLevel("),9)
                self.assertEqual(source.count("r_history_color.SampleLevel("),9)
                self.assertIn("float3 history = reset ? upscale_input.rgb : 0.f;",main)
                self.assertRegex(main,r"recurrent\s*=\s*0;")
                self.assertEqual(main.count("rw_reprojected_color[dtid.xy] = DATA_TYPE3(history);"),1)
                # The PRE helpers reachable before this guard read current
                # inputs/exposure only, never one of the three cleared targets.
                helpers=(path.parent/"mlsr_optimized_includes.hlsli")
                if not helpers.exists(): helpers=ROOT/"third_party/fsr411/shaders/reimpl411/mlsr_optimized_includes.hlsli"
                for name in ("r_history_color", "r_reprojected_color", "r_recurrent_0"):
                    self.assertNotIn(name,helpers.read_text())

    def test_pre_and_both_post_geometries_cover_the_aligned_viewport(self):
        for logical_w,logical_h in ((1,1),(7,17),(24,40),(65,41),(72,48),(97,65)):
            width=(logical_w+7)&~7;height=(logical_h+7)&~7
            target={(x,y) for x in range(width) for y in range(height)}
            pre=[];post16=[];post32=[]
            for gy in range((height+15)//16):
                for gx in range((width+15)//16):
                    for lane in range(256):
                        quad=lane>>2;phase=lane&3
                        px=(gx<<4)|((quad%8)<<1)|(phase&1)
                        py=(gy<<4)+((quad//8)<<1)|(phase>>1)
                        if px<width and py<height:pre.append((px,py))
                        px=gx*16+(lane&15);py=gy*16+(lane>>4)
                        if px<width and py<height:post16.append((px,py))
            for gy in range((height+31)//32):
                for gx in range((width+31)//32):
                    for lane in range(256):
                        ix=(gx<<4)|(lane&1)|((lane>>1)&14)
                        iy=(gy<<4)+(((lane>>5)<<1)|((lane>>1)&1))
                        if ix>=width//2 or iy>=height//2:continue
                        for phase in range(4):post32.append((ix*2+(phase&1),iy*2+(phase>>1)))
            for values in (pre,post16,post32):
                self.assertEqual(set(values),target)
                self.assertEqual(len(values),len(target))

    def test_every_imported_post_overwrites_both_targets(self):
        sources=list((ROOT/"third_party/fsr411").glob("*/*/post_rev_411*.hlsl"))
        self.assertEqual(len(sources),6)
        for path in sources:
            main=block(path.read_text(),"void fsr4_model_v07_fp8_no_scale_postpass(")
            self.assertEqual(main.count("return;"),1)
            self.assertEqual(main.count("rw_recurrent_0[tid] = (half4)rec;"),1)
            self.assertEqual(main.count("rw_history_color[tid] = res;"),1)
            self.assertEqual(main.count("r_reprojected_color[tid]"),1)
            self.assertLess(main.index("return;"),main.index("for (uint pk = 0; pk < 4; pk++)"))

    def test_reset_keeps_exposure_init_and_nonreset_reallocation_clears(self):
        runtime=(ROOT/"src/fsr411_runtime.h").read_text()
        outer=block(runtime,"if (view.zeroTargets || p.reset)")
        conditional=block(outer,"if (!p.reset && !session->originalLossy)")
        for name in ("clearHistory","clearReprojected","clearRecurrent"):
            self.assertEqual(conditional.count('"'+name+'"'),1)
            self.assertEqual(outer.count('"'+name+'"'),1)
        self.assertNotIn("initExposure",conditional)
        self.assertEqual(outer.count('"initExposure"'),1)
        self.assertIn('if (view.zeroScratch) clear_buffer(enc, m4, *slot, "clearScratch", view.scratch, 8192);',runtime)
        self.assertIn('if (skip && !session->originalLossy) clear_buffer(enc,m4,*slot,"clearLossyPlane",view.lossyPlane,10240);',runtime)
        # Opt-in original graph zeros only new targets (reset or not), like the oracle.
        original=block(runtime,"if (session->originalLossy && view.zeroTargets)")
        for name in ("originalClearHistory","originalClearReprojected","originalClearRecurrent","clearHistoryNext","clearRecurrentNext"):
            self.assertEqual(original.count('"'+name+'"'),1)


if __name__=="__main__":unittest.main()
