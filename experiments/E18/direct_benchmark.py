"""Balanced fresh-process direct sessions; synchronized scopes kept separate."""
import argparse
import json
import random
import time
import traceback
from .common import HERE,rows,pairs,save,sha,verify_plan,environment
from .inference import Inference


def selected(ids):
    lookup={(r['dataset'],r['query_id']):r for r in rows()}
    return [lookup[tuple(key)] for key in ids]


def run(device,path,session):
    plan=verify_plan();dest=HERE/'raw'/f'{device}_direct_s{session}_{path}.json'
    if dest.exists() and json.loads(dest.read_text()).get('status')=='COMPLETE':
        assert json.loads(dest.read_text())['plan_sha256']==sha(HERE/'plan.json');return
    result={'device':device,'path':path,'session':session,'plan_sha256':sha(HERE/'plan.json'),'environment':environment(),'observations':[],'warmups':[],'protocol_addendum_sha256':sha(HERE/'protocol_addendum.json')}
    try:
        model=Inference(device,path)
        result['dispatch']=model.dispatch(pairs(selected(plan['sweep_ids'])[0],10))
        cells=[(*cell,'primary' if cell==[100,16,512] else 'sweep') for cell in plan['direct']['cells']]+[(100,16,512,'sweep')];random.Random(1801+session).shuffle(cells)
        for k,b,l,group in cells:
            model.length=l
            primary=group=='primary'
            data=selected(plan['timing_ids'] if primary else plan['sweep_ids'])
            batches=[]
            for row in data:
                pp=pairs(row,k)
                batches.extend(pp[o:o+b] for o in range(0,k,b))
            encoded=[model.encode(pp) for pp in batches]
            prepared=[model.prepare(e) for e in encoded]
            warm=time.perf_counter();signatures=set()
            for inputs in prepared:
                signature=tuple(inputs['input_ids'].shape)
                if signature not in signatures:
                    model.forward(inputs);model.sync();signatures.add(signature)
            for _ in range(3):model.forward(prepared[0])
            model.sync();result['warmups'].append({'k':k,'batch':b,'length':l,'seconds':time.perf_counter()-warm,'signatures':sorted(signatures)})
            for rep in range(3):
                scopes=['forward','scoring'];random.Random(1801+session*100+rep+k+b+l).shuffle(scopes)
                for scope in scopes:
                    model.sync();start=time.perf_counter()
                    if scope=='forward':
                        for inputs in prepared:model.forward(inputs)
                    else:
                        for pp in batches:model.score(pp)
                    model.sync();seconds=time.perf_counter()-start
                    result['observations'].append({'session':session,'repetition':rep,'path':path,'scope':scope,'group':group,'k':k,'batch':b,'length':l,'queries':len(data),'pairs':len(data)*k,'seconds':seconds,'pairs_per_second':len(data)*k/seconds})
                    save(dest,result)
            del prepared,encoded
        result.update(status='COMPLETE',memory=model.memory())
        if path=='compiled':
            import torch._dynamo as dynamo
            result['compile_counters']={k:dict(v) for k,v in dynamo.utils.counters.items()}
    except Exception as e:result.update(status='FAILED_EXECUTION',error=repr(e),traceback=traceback.format_exc())
    result['end_environment']=environment();save(dest,result);print(device,path,session,result['status'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',required=True);p.add_argument('--path',required=True);p.add_argument('--session',type=int,required=True);a=p.parse_args();run(a.device,a.path,a.session)

if __name__=='__main__':main()
