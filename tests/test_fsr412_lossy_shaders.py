# SPDX-License-Identifier: MIT
"""CPU/source/optional small Metal compilation; never dispatches a GPU."""
from __future__ import annotations

import json
import math
from pathlib import Path
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "tools"))
try:
    from fsr412 import lossy_shaders as lossy
    from fsr412 import compile_fsr412 as compiler
    from fsr412 import optimization
finally:
    sys.path.pop(0)

CATALOG = json.loads((ROOT / "tools/fsr412/shader_catalog.json").read_text())["shaders"]
ROWS = [r for r in CATALOG if r["stage"] in ("pre","post")]
AVAILABLE = all((ROOT / "build/fsr412_model/lite-shaders" / (r["sha256"]+".metal")).exists() for r in ROWS)


def original(row):
    return compiler.source_for(row).read_text()


def external(row):
    return compiler.external_post_cnb(original(row),row["tier"])


def transform_pre(row, reuse_only=False):
    return lossy.apply_lossy_pre(original(row),row["tier"],bool(row.get("low_res_mv")),
                                bool(row.get("depth_inverted")),reuse_only)


def block(source, marker):
    begin=source.index("{",source.index(marker))
    end=lossy._close(source,begin,"{","}")
    return source[begin+1:end]


def motion_record_bytes(width,height):
    return (width//2)*(height//2)*lossy.MOTION_BYTES_PER_CELL


class HostAndCPUTests(unittest.TestCase):
    def test_cache_cost_is_explicit_int32_not_silent_short_truncation(self):
        self.assertEqual(lossy.CACHE_BYTES_PER_PIXEL,32)
        self.assertEqual(lossy.CONSTANT_BYTES,16)
        self.assertEqual(lossy.cache_buffer_bytes(1920,1080),66355200)
        self.assertEqual(lossy.cache_buffer_bytes(3840,2160),265420800)
        self.assertEqual(motion_record_bytes(1920,1080),8294400)
        for dims in ((0,2),(1,2),(3,2),(2,3),(7682,4320),(2.0,2),(True,2)):
            with self.assertRaises(ValueError): lossy.cache_buffer_bytes(*dims)

    def test_partial_fresh_mask_matches_native_contract_and_cell_ownership(self):
        for size in ((8,8),(1024,512),(1920,1080),(3840,2160)):
            points=[(0,0),(1,3),(2,2),(510,510),(511,511),(512,512),
                    (size[0]-3,size[1]-3),(size[0]-2,2)]
            for p in points:
                if not all(0<=p[a]<size[a] for a in (0,1)): continue
                edge=p[0]<2 or p[1]<2 or p[0]>=size[0]-2 or p[1]>=size[1]-2
                for phase in (0,1):
                    wanted=edge or ((p[0]//512+p[1]//512)&1)==phase
                    self.assertEqual(lossy.fresh_pixel(p,size,6,512,phase),wanted)
            # All four output pixels of a 2x2 CNB cell have the same predicate.
            for y in range(0,size[1],2):
                for x in (0,2,510,512,size[0]-4,size[0]-2):
                    if x>=size[0]:continue
                    phases={lossy.fresh_pixel((x+dx,y+dy),size,6,512,1) for dx,dy in ((0,0),(1,0),(0,1),(1,1))}
                    self.assertEqual(len(phases),1)
        for flags in (0,1,2,3,4,7,8,13):
            self.assertTrue(lossy.fresh_pixel((700,100),(1024,512),flags))
        self.assertFalse(lossy.fresh_pixel((0,0),(1024,512),5))  # Whole skips CNB even at edges.
        self.assertTrue(lossy.fresh_pixel((700,100),(1024,512),5,reset=True))
        self.assertTrue(lossy.fresh_pixel((700,100),(1025,512),5))
        for tile_size in (0,32,127,129,513,2049,4096):
            self.assertTrue(lossy.fresh_pixel((700,100),(1024,512),6,tile_size,0))
        for tile_size in lossy.PARTIAL_TILE_SIZES:
            for phase in (0,1):
                for pixel in ((2,2),(127,127),(128,129),(255,255),(256,257),(513,100),(700,100)):
                    expected=((pixel[0]//tile_size+pixel[1]//tile_size)&1)==phase
                    self.assertEqual(lossy.fresh_pixel(pixel,(1024,512),6,tile_size,phase),expected)
        self.assertTrue(lossy.fresh_pixel((700,100),(1024,512),6,512,2))

    def test_motion_is_round_even_bounded_and_never_nan_to_int(self):
        for pixels,wanted in (((.5,1.5),(0,2)),((2.5,-1.5),(2,-2)),
                              ((-.03125,.03125),(0,0)),((254,-254),(254,-254))):
            self.assertEqual(lossy.sanitized_motion(tuple(x/1000 for x in pixels),(1000,1000)),(wanted,True))
        for vel in ((math.nan,0),(math.inf,0),(-math.inf,0),(1e30,0),(.255,0)):
            self.assertEqual(lossy.sanitized_motion(vel,(1000,1000)),((0,0),False))
        self.assertEqual(lossy.sanitized_motion((0,0),(1000,1000),invalid=True),((0,0),False))
        self.assertEqual(lossy.sanitized_motion((0,0),(1000,1000),reset=True),((0,0),False))

    def test_partial_pingpong_in_place_writes_are_never_previous_reads(self):
        size=(1024,8)
        previous={(x,y):(x,y,"raw8") for y in range(size[1]) for x in range(size[0])}
        current={}; refreshed=0; reused=0
        for y in range(size[1]):
            for x in range(size[0]):
                if lossy.fresh_pixel((x,y),size,6,512,0):
                    current[x,y]=(x,y,"fresh"); refreshed+=1
                else:
                    p=(min(max(x-3,0),size[0]-1),y)
                    current[x,y]=previous[p]; reused+=1
        self.assertGreater(refreshed,0);self.assertGreater(reused,0)
        self.assertEqual(len(current),len(previous))
        self.assertTrue(all(v[2]=="raw8" for v in previous.values()))
        # Odd motion picks previous final PIXEL parameters, not current CT phase.
        self.assertEqual(current[700,3],previous[697,3])

    def test_temporal_core_keeps_uniform_detail_but_clamps_motion_boundary(self):
        # CPU specification of the emitted bounded policy, no FP16 equality test.
        def core(current,history,share,samples,rest=False,boundary=False,uncovered=False):
            lo=[min(s[ch] for s in samples) for ch in range(3)]
            hi=[max(s[ch] for s in samples) for ch in range(3)]
            span=[b-a for a,b in zip(lo,hi)]
            diff=[abs(c-h) for c,h in zip(current,history)]
            changed=any(h<a or h>b or d>r*.25+.05 for h,a,b,d,r in zip(history,lo,hi,diff,span))
            changed=changed or any(abs(a+b-2*h)>r*.85 and d>r*.05+.01
                                  for a,b,h,r,d in zip(lo,hi,history,span,diff))
            fresh=1-share; weight=fresh*fresh
            if rest and not changed:weight*=min(max((fresh-.95)/.045,0),1)
            old=[min(max(h,a),b) if boundary else h for h,a,b in zip(history,lo,hi)]
            dark=not rest and all(h<=max(c*.1,0) for h,c in zip(history,current))
            return tuple(current) if uncovered or dark else tuple(c*weight+h*(1-weight) for c,h in zip(current,old))
        samples=[(.3,)*3,(1.0,)*3]+[(.5,)*3]*7
        self.assertEqual(core((.55,)*3,(.6,)*3,.8,samples,rest=True),(.6,)*3)
        samples=[(.3,)*3,(.5,)*3]+[(.4,)*3]*7
        uniform=core((.4,)*3,(1.,)*3,.8,samples)
        boundary=core((.4,)*3,(1.,)*3,.8,samples,boundary=True)
        self.assertTrue(all(a>.9 for a in uniform))
        self.assertTrue(all(.3<=a<=.5 for a in boundary))
        self.assertEqual(core((.4,)*3,(1.,)*3,.8,samples,uncovered=True),(.4,)*3)
        # A bright sample without support on BOTH axes cannot trigger a spark.
        support=lambda center,left,right,above,below:min(center,max(left,right),max(above,below))
        self.assertEqual(support(10,10,10,.5,.5),.5)
        for span in (0,.01,1,100):
            self.assertTrue(.08<=min(max(.08+span*.5,.08),.25)<=.25)

    def test_pixel_source_clamp_and_short8_would_overflow(self):
        for p,m in (((0,0),(-254,-254)),((1919,1079),(254,254)),((700,10),(-3,1))):
            safe=tuple(min(max(p[a]+m[a],0),(1920,1080)[a]-1) for a in (0,1))
            self.assertTrue(0<=safe[0]<1920 and 0<=safe[1]<1080)
        self.assertNotEqual(53084,struct.unpack("<h",struct.pack("<H",53084&65535))[0])
        self.assertNotEqual(73070,struct.unpack("<h",struct.pack("<H",73070&65535))[0])


@unittest.skipUnless(AVAILABLE,"imported Lite shader artifacts not present")
class ImportedSourceTests(unittest.TestCase):
    def test_all_pre_variants_tiers_and_reuse_specializations(self):
        count=0
        for row in ROWS:
            if row["stage"]!="pre":continue
            for reuse in (False,True):
                with self.subTest(tier=row["tier"],sha=row["sha256"],reuse=reuse):
                    source=transform_pre(row,reuse)
                    self.assertEqual(lossy.apply_lossy_pre(source,row["tier"],bool(row.get("low_res_mv")),
                                     bool(row.get("depth_inverted")),reuse),source)
                    main=source[source.index("kernel void"):]
                    record=main.index("Fsr412LossyRecord(")
                    reprojection=main.index("rw_reprojected_color.write(")
                    tensor=main.index("ScratchBuffer._m0[")
                    self.assertLess(reprojection,record); self.assertLess(record,tensor)
                    self.assertEqual(main.count("Fsr412LossyRecord("),1)
                    if reuse:
                        self.assertLess(main.index("== 1u) return;"),tensor)
                    else:
                        self.assertNotIn("== 1u) return;",main)
                    self.assertIn("as_type<uint2>(displacement)",source)
                    self.assertNotIn("Fsr412LossyPrevious [[",source)
                    indices=list(map(int,re.findall(r"\[\[buffer\((\d+)\)\]\]",main.split("\n")[0])))
                    self.assertEqual(len(indices),len(set(indices)))
                    self.assertLessEqual(max(indices),7)
            count+=1
        self.assertEqual(count,90)

    def test_actual_ct_bounds_all47_post_variants_and_exact_int32_arithmetic(self):
        maxima={"normal":set(),"ultraperformance":set()};rng=random.Random(412)
        for row in ROWS:
            if row["stage"]!="post":continue
            src=original(row); layout=lossy.cache_layout(src)
            self.assertEqual(layout["cache_bytes_per_pixel"],32)
            self.assertFalse(layout["int16_supported"])
            self.assertLess(layout["max_accumulator_bound"],2**24)
            maxima[row["preset"]].add(layout["max_accumulator_bound"])
            tail=lossy._tail(src)
            table=lossy._one(r"^constant spvUnsafeArray<uint, 128> _\d+ = spvUnsafeArray<uint, 128>\(\{(.*?)\}\);$",src,"weights")
            words=list(map(int,re.findall(r"(\d+)u",table[1])))
            weights=[(word>>shift)&255 for word in words for shift in (0,8,16,24)]
            weights=[w if w<128 else w-256 for w in weights]
            bias=[tuple(map(int,b)) for b in re.findall(r"^            _\d+ = int4\((-?\d+), (-?\d+), (-?\d+), (-?\d+)\);$",tail["raw"],re.M)]
            for phase in range(4):
                for half in range(2):
                    for ch in range(4):
                        column=weights[phase*128+half*64+ch*16:phase*128+half*64+ch*16+16]
                        for values in ([rng.randrange(-128,128) for _ in range(16)],
                                       [127 if w>=0 else -128 for w in column],
                                       [-128 if w>=0 else 127 for w in column]):
                            acc=sum(w*x for w,x in zip(column,values))+bias[1-half][ch]
                            self.assertLessEqual(abs(acc),layout["column_bounds"][phase*8+half*4+ch])
                            self.assertEqual(struct.unpack("<i",struct.pack("<i",acc))[0],acc)
        self.assertEqual(maxima,{"normal":{53084},"ultraperformance":{73070}})

    def test_post_allflags_preserve_original_filter(self):
        count=0
        for row in ROWS:
            if row["stage"]!="post":continue
            raw=external(row)
            sources=[raw]
            if row["colorspace"]==0:
                prepared,_=optimization.input_color_cache(raw,9)
                sources.append(prepared)
            for src in sources:
                with self.subTest(tier=row["tier"],cs=row["colorspace"],preset=row["preset"],tiling="fsr412PostTile" in src):
                    tail=lossy._tail(src)
                    result=lossy.apply_lossy_post(src,row["tier"])
                    self.assertEqual(lossy.apply_lossy_post(result,row["tier"]),result)
                    main=result[result.index("kernel void"):]
                    # Filter expressions between raw scale and history array write
                    # remain intact, including Gaussian order and tiled read calls.
                    before=tail["body"][tail["filt"].end():]
                    prefix=before[:before.index("r_reprojected_color.read(")]
                    self.assertIn(prefix,result)
                    self.assertEqual(main.count("Fsr412LossyCurrent["),1)
                    self.assertEqual(main.count("Fsr412LossyPrevious["),1)
                    self.assertIn("if (Fsr412LossyMode(Fsr412LossyConstants, MLSR_Optimized_Constants) != 1u)",main)
                    self.assertIn("if (!fsr412PixelReuse)",main)
                    self.assertIn("if (fsr412PixelReuse)",main)
                    self.assertEqual(main.count("ScratchBuffer._m0["),4)
                    self.assertEqual(len(re.findall(r"^        uint _\d+ = 0u;$",main,re.M)),4)
                    guarded=block(main,"if (Fsr412LossyFresh(")
                    self.assertEqual(guarded.count("ScratchBuffer._m0["),4)
                    self.assertNotIn("ScratchBuffer",block(main,"else\n            {\n                uint4 fsr412Own"))
                    self.assertEqual(main.count("rw_history_color.write("),4)
                    self.assertEqual(main.count("rw_mlsr_output_color.write("),4)
                    self.assertEqual(main.count("rw_recurrent_0.write("),4)
                    self.assertNotIn("short",result)
                    if "fsr412PostTile" in src:
                        self.assertEqual(main.count("threadgroup_barrier"),1)
                        self.assertLess(main.index("threadgroup_barrier"),main.index("fsr412PixelReuse"))
            count+=1
        self.assertEqual(count,47)

    def test_fail_closed_wrong_tier_flags_tail_weights_and_mutated_receipts(self):
        pre=next(r for r in ROWS if r["stage"]=="pre" and r.get("low_res_mv") and r.get("depth_inverted"))
        with self.assertRaises(ValueError):
            lossy.apply_lossy_pre(original(pre),pre["tier"],True,False)
        with self.assertRaises(ValueError):
            lossy.apply_lossy_pre(original(pre),pre["tier"],False,True)
        p=transform_pre(pre,True)
        with self.assertRaises(ValueError):
            lossy.apply_lossy_pre(p+"\n",pre["tier"],True,True,True)
        row=next(r for r in ROWS if r["stage"]=="post" and r["tier"]==2160)
        raw=external(row)
        for invalid,tier in ((raw,7),(raw,1080),(original(row),2160),
            (raw.replace("0.0009765625","0.0009",1),2160),
            (raw.replace("spvUnsafeArray<uint, 128>","spvUnsafeArray<uint, 127>"),2160),
            (raw.replace("int4(-352", "int4(-2147483648") if "int4(-352" in raw else raw.replace("int4(-114","int4(-2147483648"),2160)):
            with self.assertRaises(ValueError):lossy.apply_lossy_post(invalid,tier)
        result=lossy.apply_lossy_post(raw,2160)
        with self.assertRaises(ValueError):lossy.apply_lossy_post(result+"\n",2160)
        with self.assertRaises(ValueError):lossy.apply_lossy_post(result,1080)
        with self.assertRaises(ValueError):lossy.apply_lossy_post(raw.replace("kernel void","kernel void Fsr412LossyCollision",1),2160)

    @unittest.skipUnless(shutil.which("xcrun"),"Apple Metal compiler not installed")
    def test_small_actual_metal_compile_pre_post_and_tiled_prepared_post(self):
        row=next(r for r in ROWS if r["stage"]=="pre" and r["tier"]==2160 and r.get("low_res_mv")
                 and r.get("depth_inverted") and r.get("colorspace")==0 and r.get("auto_exposure")==0)
        post=next(r for r in ROWS if r["stage"]=="post" and r["tier"]==2160 and r["preset"]=="normal"
                  and r["colorspace"]==0 and r.get("auto_exposure")==0)
        pre=compiler.signed_square(original(row))
        pre=lossy.apply_lossy_pre(pre,2160,True,True,True)
        prepared,_=optimization.input_color_cache(external(post),9)
        sources=[("pre_reuse",pre),("post",lossy.apply_lossy_post(external(post),2160)),
                 ("post_prepared",lossy.apply_lossy_post(prepared,2160))]
        for colorspace in (1,2,3):
            flagged=next(r for r in ROWS if r["stage"]=="post" and r["tier"]==2160 and
                         r["colorspace"]==colorspace and r["preset"]==("ultraperformance" if colorspace==2 else "normal"))
            sources.append(("post_cs%d" % colorspace,lossy.apply_lossy_post(external(flagged),2160)))
        full_mv=next(r for r in ROWS if r["stage"]=="pre" and r["tier"]==2160 and
                     not r.get("low_res_mv") and r.get("colorspace")==0 and r.get("auto_exposure")==0)
        sources.append(("pre_full_mv",lossy.apply_lossy_pre(compiler.signed_square(original(full_mv)),2160,False,False,True)))
        with tempfile.TemporaryDirectory(prefix="fsr412-lossy-test-") as directory:
            folder=Path(directory)
            for name,text in sources:
                src=folder/(name+".metal"); out=folder/(name+".air"); src.write_text(text)
                built=subprocess.run(["xcrun","metal","-std=macos-metal2.3","-O0","-c",str(src),"-o",str(out)],
                                     capture_output=True,text=True)
                self.assertEqual(built.returncode,0,built.stdout+built.stderr)
                self.assertTrue(out.exists())


if __name__=="__main__":unittest.main()
