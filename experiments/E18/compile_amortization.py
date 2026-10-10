"""Illustrative startup amortization, conditional on a supported warm saving."""
import argparse
import json
import numpy as np
from .common import HERE,save,sha
from .statistics import paired_ratio


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',required=True,choices=['mps','cuda']);device=parser.parse_args().device
    startup={p:json.loads((HERE/f'{device}_{p}_startup.json').read_text()) for p in ('stock','compiled')}
    observed=[]
    for path in (HERE/'raw').glob(f'{device}_direct_s*.json'):
        report=json.loads(path.read_text())
        if report.get('status')!='COMPLETE':continue
        observed.extend(v for v in report['observations'] if (v['scope'],v['k'],v['batch'],v['length'],v['group'])==('scoring',100,16,512,'primary') and v['path'] in ('stock','compiled'))
    comparison=paired_ratio(observed,'stock','compiled')
    output={'device':device,'plan_sha256':sha(HERE/'plan.json'),'warm_comparison':comparison,
            'scope':'One ordered fresh software-cache startup pair; GPU driver cache unknown; startup CI unavailable. Model setup includes imports and storage effects. Warm saving uses matched K100 requests; not an HTTP or fully cold hardware break-even.',
            'startup_artifact_sha256':{p:sha(HERE/f'{device}_{p}_startup.json') for p in startup}}
    if comparison['ci95'][0]<=1:
        output['status']='NO_SUPPORTED_WARM_SAVING'
    else:
        pairs={}
        for r in observed:pairs.setdefault((r['session'],r['repetition']),{})[r['path']]=r
        savings=[(v['stock']['seconds']-v['compiled']['seconds'])/v['stock']['queries'] for v in pairs.values()]
        saving=float(np.median(savings));assert saving>0
        first=startup['compiled']['first_score_seconds']-startup['stock']['first_score_seconds']
        total=sum(startup['compiled'][k]-startup['stock'][k] for k in ('model_setup_seconds','first_score_seconds'))
        output.update(status='DIAGNOSTIC_ONLY',median_warm_seconds_saved_per_K100_query=saving,
                      first_score_overhead_seconds=first,total_startup_difference_seconds=total,
                      first_score_amortization_queries=max(0,first)/saving,total_startup_amortization_queries=max(0,total)/saving)
    save(HERE/f'{device}_compile_amortization.json',output)


if __name__=='__main__':main()
