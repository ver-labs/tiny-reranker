"""Diagnostic only: fullgraph compilation over the custom CUDA adapter."""
import time
import traceback
import torch
from .common import HERE,fixtures,pairs,save,sha
from .inference import Inference


def main():
    started=time.perf_counter();report={'path':'cuda+inductor','cache_state':'Unknown compiler/system cache','mode':'fullgraph=True dynamic=True','plan_sha256':sha(HERE/'plan.json'),'scope':'Compatibility diagnostic; never added post-hoc to balanced timing path list'}
    try:
        model=Inference('cuda','cuda');model.call=torch.compile(model.model,backend='inductor',fullgraph=True,dynamic=True)
        scores=model.score(pairs(fixtures()[0],16));model.sync()
        report.update(status='SUPPORTED_UNQUALIFIED',scores=scores)
    except Exception as e:report.update(status='UNSUPPORTED',error=repr(e),traceback=traceback.format_exc())
    report['first_use_seconds']=time.perf_counter()-started;save(HERE/'cuda_combined_probe.json',report);print(report['status'],flush=True)

if __name__=='__main__':main()
