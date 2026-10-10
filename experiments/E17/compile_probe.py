"""Bounded first-use compile compatibility probe, run in a separate process."""
import argparse
import json
import time
import traceback
from .common import HERE, rows, sha, save, verify_plan
from .inference import Inference


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['mps','cuda'],required=True);a=p.parse_args()
    verify_plan();r=rows()[0];pairs=[(r['query'],t) for t in r['texts'][:32]]
    started=time.perf_counter()
    report={'device':a.device,'backend':'inductor','fullgraph':True,'dynamic':False,
            'plan_sha256':sha(HERE/'plan.json'), 'scope':'First invocation in a fresh process; disk cache state unknown, not a true cold compile.'}
    try:
        import torch
        from torch._inductor import metrics
        model=Inference(a.device,'compiled'); inputs=model.prepare(model.encode(pairs))
        scores=model.cpu(model.forward(inputs));model.sync()
        report.update(status='compatible',first_use_seconds=time.perf_counter()-started,
                      generated_kernel_count=metrics.generated_kernel_count,scores=scores)
        report['dispatch']=model.dispatch(pairs)
    except Exception as e:
        report.update(status='unsupported_or_failed',error_type=type(e).__name__,error=str(e),
                      diagnostic_traceback=traceback.format_exc(),first_use_seconds=time.perf_counter()-started)
    save(HERE/f'{a.device}_compile.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
