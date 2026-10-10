"""Recompute every aggregate from per-run observations; no baked-in winners."""
import json
from collections import defaultdict
from .common import HERE,save,sha
from .statistics import paired_ratio,distribution


def main():
    output={'plan_sha256':sha(HERE/'plan.json'),'accuracy':{},'direct':{},'http':{},'operation':{},'optimized_http':{}}
    for p in HERE.glob('*_accuracy.json'):
        r=json.loads(p.read_text());output['accuracy'][p.stem.replace('_accuracy','')]={k:v for k,v in r.items() if k not in ('per_query','environment','traceback')}
    for device in ('mps','cuda'):
        observations=[]
        for p in (HERE/'raw').glob(f'{device}_direct_s*.json'):
            result=json.loads(p.read_text())
            if result.get('status')=='COMPLETE':observations+=result['observations']
        cells=defaultdict(list)
        for r in observations:cells[(r['scope'],r['k'],r['batch'],r['length'],r['group'])].append(r)
        for cell,items in cells.items():
            paths=sorted({r['path'] for r in items});entry={'distribution':{},'comparisons':{}}
            for path in paths:
                rr=[r for r in items if r['path']==path];entry['distribution'][path]={'seconds':distribution([r['seconds'] for r in rr]),'pairs_per_second':distribution([r['pairs_per_second'] for r in rr]),'repetitions':len(rr),'sessions':len({r['session'] for r in rr}),'accuracy':output['accuracy'].get(device+'_'+path,{}).get('status','PENDING') if cell[2:4]==(16,512) else 'DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED'}
            comparisons=[('stock',p) for p in paths if p!='stock']
            if device=='mps':
                comparisons += [('mlx',p) for p in ('mlx_compiled','metal','metal_compiled') if p in paths]
                if 'mlx_compiled' in paths and 'metal_compiled' in paths:comparisons.append(('mlx_compiled','metal_compiled'))
            else:
                comparisons += [('explicit',p) for p in ('cuda','triton') if p in paths]
                if 'compiled' in paths:comparisons += [('compiled',p) for p in ('cuda','triton') if p in paths]
            for baseline,path in comparisons:
                try:entry['comparisons'][baseline+'/'+path]=paired_ratio(items,baseline,path)
                except ValueError as error:entry['comparisons'][baseline+'/'+path]={'status':'INCOMPLETE','reason':str(error)}
            output['direct'][device+'/'+str(cell)]=entry
        ops=[]
        for p in (HERE/'raw').glob(f'{device}_operation_s*.json'):ops+=json.loads(p.read_text())['observations']
        for rows in sorted({r['rows'] for r in ops}):
            selected=[r for r in ops if r['rows']==rows];entry={'distribution':{},'comparisons':{}}
            for path in sorted({r['path'] for r in selected}):
                entry['distribution'][path]=distribution([r['seconds']*1e6 for r in selected if r['path']==path])
                if path!='native':entry['comparisons']['native/'+path]=paired_ratio(selected,'native',path)
            output['operation'][f'{device}/{rows}']=entry
    runs=[]
    for p in (HERE/'raw').glob('*http_s*.json'):
        result=json.loads(p.read_text())
        if result.get('status')=='COMPLETE':runs += [{**r,'study':result.get('study','primary')} for r in result['runs']]
    for study,k,c in sorted({(r['study'],r['k'],r['concurrency']) for r in runs}):
        selected=[r for r in runs if (r['study'],r['k'],r['concurrency'])==(study,k,c)];entry={'distribution':{},'comparisons':{}}
        for path in sorted({r['path'] for r in selected}):
            records=[r for r in selected if r['path']==path]
            entry['distribution'][path]={f:distribution([r[f] for r in records]) for f in ('seconds','requests_per_second','pairs_per_second','validation_seconds','client_cpu_fraction_of_one_core')}
            for percentile in ('p50','p95','p99'):entry['distribution'][path][percentile+'_ms']=distribution([r['latency_ms'][percentile] for r in records])
            for scope in ('queue','processing'):
                entry['distribution'][path][scope+'_ms']={percentile:distribution([r[scope+'_ms'][percentile] for r in records]) for percentile in ('p50','p95','p99')}
            entry['distribution'][path].update(failures=sum(r['failures'] for r in records),timeouts=sum(r['timeouts'] for r in records),requests=sum(r['successful_requests'] for r in records),repetitions=len(records))
        for backend in ('stock','mlx'):
            try:entry['comparisons']['python/'+backend+'/rust/'+backend]=paired_ratio(selected,'python/'+backend,'rust/'+backend)
            except ValueError as e:entry['comparisons'][backend]={'status':'INCOMPLETE','reason':str(e)}
        if study=='primary':output['http'][f'K{k}/C{c}']=entry
        else:
            baseline='python/'+study.removeprefix('optimized_')
            entry['comparisons']={path:paired_ratio(selected,baseline,path) for path in entry['distribution'] if path!=baseline}
            output['optimized_http'][f'{study}/K{k}/C{c}']=entry
    save(HERE/'summary.json',output);print('Summary updated',flush=True)

if __name__=='__main__':main()
