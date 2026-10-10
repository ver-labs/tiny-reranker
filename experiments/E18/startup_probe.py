"""Separate startup diagnostic with explicitly fresh compiler caches."""
import argparse
import os
import time
from .common import HERE,fixtures,pairs,save,sha,environment
from .inference import Inference


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',required=True,choices=['mps','cuda']);p.add_argument('--path',choices=['stock','compiled'],required=True);a=p.parse_args()
    cache=HERE/'cache'/f'{a.device}_startup_{a.path}'
    if cache.exists():raise RuntimeError('Fresh-cache probe already attempted; retain and version a new attempt')
    (cache/'inductor').mkdir(parents=True);(cache/'triton').mkdir()
    os.environ['TORCHINDUCTOR_CACHE_DIR']=str(cache/'inductor');os.environ['TRITON_CACHE_DIR']=str(cache/'triton')
    report={'device':a.device,'path':a.path,'plan_sha256':sha(HERE/'plan.json'),'environment':environment(),
            'cache_state':'Inductor and Triton directories newly created and empty; GPU driver/system shader cache unknown','scope':'One fresh-process diagnostic; model setup and first score separate; B16 / L≤512; not warm latency'}
    started=time.perf_counter();model=Inference(a.device,a.path);report['model_setup_seconds']=time.perf_counter()-started
    request=pairs(fixtures()[0],16);start=time.perf_counter();scores=model.score(request);model.sync();report['first_score_seconds']=time.perf_counter()-start
    start=time.perf_counter();model.score(request);model.sync();report['second_score_seconds']=time.perf_counter()-start
    report['scores']=scores;report['status']='COMPLETE';save(HERE/f'{a.device}_{a.path}_startup.json',report);print(report['first_score_seconds'],flush=True)

if __name__=='__main__':main()
