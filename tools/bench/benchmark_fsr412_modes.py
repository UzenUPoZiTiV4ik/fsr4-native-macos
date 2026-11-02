#!/usr/bin/env python3
"""Interleave native CLI replays of a real capture; no diagnostic GPU copies.

Compilation/data preparation happens once before --no-compile timings.
Whole/reuse/partial distributions stay separate. This is not game FPS.
"""
import argparse
import json
import statistics
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
PLANS=(('lite','int8-lite',0),('lite-post-tiles','int8-lite',1),
       ('lite-lossy','int8-lite-lossy',0),('lite-lossy-tiled','int8-lite-lossy-tiled',0))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture',type=Path)
    p.add_argument('--rounds',type=int,default=6)
    p.add_argument('--sequences',type=int,default=30)
    p.add_argument('--out',type=Path,default=ROOT/'build/lite-lossy-review/replay-ab')
    args=p.parse_args()
    if not 2<=args.rounds<=100 or not 2<=args.sequences<=1000:raise ValueError('invalid benchmark count')
    args.out.mkdir(parents=True,exist_ok=True)
    all_runs={name:[] for name,_,_ in PLANS}
    for round in range(-1,args.rounds):
        plans=PLANS if round%2==0 else tuple(reversed(PLANS))
        for name,profile,tile in plans:
            directory=args.out/name
            command=[str(ROOT/'tools/bin/fsr4mac'),str(args.capture.resolve()),str(directory),
                     '--fsr-version','4.1.2','--profile',profile,'--post-tiling',str(tile),'--bench',str(args.sequences)]
            if round>=0:command.append('--no-compile')
            result=subprocess.run(command,capture_output=True,text=True)
            if result.returncode:raise RuntimeError(result.stderr[-12000:])
            run=json.loads((directory/'run.json').read_text())
            if round<0:continue
            all_runs[name].append(run)
            print(name,round,'mean',round_float(run['gpu_mean_ms']),flush=True)
    report={'capture':str(args.capture.resolve()),'rounds':args.rounds,'sequences':args.sequences,
            'ordering':'reversed every round','timing':'full classic command buffer; no diagnostic copies/counters','rows':{}}
    for name,runs in all_runs.items():
        means=[run['gpu_mean_ms'] for run in runs]
        times=[frame['gpu_ms'] for run in runs for frame in run['measurements']]
        ordered=sorted(times)
        report['rows'][name]={'median_run_mean_ms':statistics.median(means),'run_mean_ms':means,
            'frame_p99_ms':ordered[int((len(ordered)-1)*.99)],'frame_max_ms':max(times),
            'full_times':[f['gpu_ms'] for r in runs for f in r['measurements'] if not f.get('skipped_inference') and not f.get('partial_inference')],
            'reuse_times':[f['gpu_ms'] for r in runs for f in r['measurements'] if f.get('skipped_inference')],
            'partial_times':[f['gpu_ms'] for r in runs for f in r['measurements'] if f.get('partial_inference')]}
    target=args.out/'summary.json';target.write_text(json.dumps(report,indent=2)+'\n')
    print(target)
    for name,row in report['rows'].items():print(name,round_float(row['median_run_mean_ms']))

def round_float(value):return round(value,4)
if __name__=='__main__':main()
