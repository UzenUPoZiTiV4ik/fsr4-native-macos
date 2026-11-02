# SPDX-License-Identifier: MIT
import importlib.util
from pathlib import Path
import unittest

SPEC=importlib.util.spec_from_file_location("thermal_benchmark",Path(__file__).resolve().parents[1]/"tools/bench/benchmark_modes_thermal.py")
bench=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def fixture(size="1080p",value=2):
    rows=[]
    for name in sorted(bench.NAMES):
        reused=name in {"lossy","lossy2","lossy2-temporal1","lossy2-temporal2","int8-lite-lossy","int8-lite-lossy-original"}
        partial=name=="int8-lite-lossy-tiled"
        full=[] if partial else [value]*(8 if reused else 16)
        reuse=[value]*8 if reused else []
        part=[value]*16 if partial else []
        rows.append({"name":name,"label":name,"frames_by_round_ms":[[value]*8,[value]*8],
                     "round_means_ms":[value,value],"median_round_mean_ms":value,"late_vs_early_percent":0,
                     "full_count":len(full),"reuse_count":len(reuse),"partial_count":len(part),
                     "full_ms":full,"reuse_ms":reuse,"partial_ms":part})
    return {"schema":"fsr_modes_thermal_v1","render":list(bench.SIZES[size][:2]),"output":list(bench.SIZES[size][2:]),
            "rounds":2,"frames_per_round":8,"low_power_mode":False,"rcas":False,"auto_exposure":True,
            "backend":"native INT8 MPP","metal":"classic command buffers","rows":rows}


class ThermalBenchmarkTests(unittest.TestCase):
    def test_nearest_rank_and_bounds(self):
        self.assertEqual(bench.percentile([3,1,2],0),1)
        self.assertEqual(bench.percentile([3,1,2],.99),3)
        self.assertEqual(bench.percentile([3,1,2],1),3)

    def test_invalid_durations(self):
        for values in ([],[0],[-1],[float("nan")],[float("inf")],[True]):
            with self.subTest(values=values),self.assertRaises(ValueError):bench.percentile(values,.99)

    def test_complete_profile_matrix(self):
        for size in bench.SIZES:bench.validate_run(fixture(size),size,2,8,False)

    def test_forced_simd_report_requires_actual_backend(self):
        run=fixture(); run["backend"]="INT8 SIMD (half matrices, float accumulation)"
        bench.validate_run(run,"1080p",2,8,False,True)
        with self.assertRaises(ValueError):bench.validate_run(run,"1080p",2,8,False,False)
        with self.assertRaises(ValueError):bench.validate_run(fixture(),"1080p",2,8,False,True)

    def test_no_missing_modes_or_diagnostics(self):
        for key,value in (("low_power_mode",True),("rcas",True),("auto_exposure",False),("metal","Metal 4"),("backend","scalar")):
            run=fixture();run[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):bench.validate_run(run,"1080p",2,8,False)
        run=fixture();run["rows"].pop()
        with self.assertRaises(ValueError):bench.validate_run(run,"1080p",2,8,False)

    def test_incomplete_or_wrong_cadence(self):
        for broken in ("block","means","cadence"):
            run=fixture();row=next(r for r in run["rows"] if r["name"]=="int8-lite-lossy")
            if broken=="block":row["frames_by_round_ms"][0].pop()
            elif broken=="means":row["round_means_ms"][0]=99
            else:row.update(full_count=16,reuse_count=0,full_ms=[2]*16,reuse_ms=[])
            with self.subTest(broken=broken),self.assertRaises(ValueError):bench.validate_run(run,"1080p",2,8,False)

    def test_summary_uses_all_rounds_not_best_run(self):
        runs=[{"size":size,"pass":index,"data":fixture(size,value)}
              for index,(size,value) in enumerate((("1080p",2),("1440p",3),("1440p",5),("1080p",4)),1)]
        summary=bench.summarize(runs)
        for size,want in (("1080p",3),("1440p",4)):
            row=summary[size]["models"]["int8-lite-lossy"]
            self.assertEqual(row["gpu_ms"],want)
            self.assertEqual(row["measured_frames"],32)
            self.assertEqual(row["full_count"],16)
            self.assertEqual(row["reuse_count"],16)
            self.assertEqual(row["partial_count"],0)
            self.assertIsNone(row["partial_mean_ms"])
            self.assertEqual(len(row["sessions"]),2)


if __name__=="__main__":unittest.main()
