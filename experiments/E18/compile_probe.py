"""Candidate modes declared here before timing; isolated cache and process."""
import argparse
import os
import time
import traceback
from .common import HERE,fixtures,save,environment
from .inference import Inference


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',required=True,choices=['mps','cuda'])
    p.add_argument('--path',required=True,choices=['compiled','mlx_compiled','metal_compiled']);a=p.parse_args()
    report={'device':a.device,'path':a.path,'mode':'Inductor fullgraph=True dynamic=True' if a.path=='compiled' else 'mx.compile(shapeless=False)',
            'cache_state':'Unknown system shader cache; Inductor directory '+os.environ.get('TORCHINDUCTOR_CACHE_DIR','default (not isolated)'),
            'environment':environment()}
    # The runner isolates Inductor disk cache before this process starts.
    started=time.perf_counter()
    try:
        model=Inference(a.device,a.path)
        pairs=[(fixtures()[0]['query'],t) for t in fixtures()[0]['texts'][:16]]
        values=model.score(pairs);model.sync()
        report.update(status='SUPPORTED',first_use_seconds=time.perf_counter()-started,scores=values,
                      dispatch=model.dispatch(pairs))
        if a.path=='compiled':
            from torch._inductor import metrics
            import torch._dynamo as dynamo
            report['generated_kernel_count']=metrics.generated_kernel_count
            report['dynamo_counters']={k:dict(v) for k,v in dynamo.utils.counters.items()}
        else:
            import mlx.core as mx
            inputs=model.prepare(model.encode(pairs))
            graph=HERE/'traces'/f'{a.device}_{a.path}.dot';graph.parent.mkdir(exist_ok=True)
            mx.export_to_dot(str(graph),model.call(**inputs))
            report['graph_file']=str(graph.relative_to(HERE))
    except Exception as error:
        report.update(status='UNSUPPORTED',error_type=type(error).__name__,error=str(error),
                      traceback=traceback.format_exc(),first_use_seconds=time.perf_counter()-started)
    save(HERE/f'{a.device}_{a.path}_probe.json',report)
    print(report['status'],report.get('error',''),flush=True)


if __name__=='__main__':main()
