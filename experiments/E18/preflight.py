"""Bounded device feasibility, CPU/mask/layer checks; no headline timing."""
import argparse
import gc
import json
import time
import traceback
import numpy as np
from .common import HERE,fixtures,save,environment
from .inference import Inference


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['mps','cuda'],required=True);a=p.parse_args()
    started=time.perf_counter()
    data=fixtures()
    probes=[('q','short document'),(data[0]['query'],data[0]['texts'][0]),
            (data[32]['query'],data[32]['texts'][0]),('repeated query','long document '*700)]
    cpu=Inference('cpu','stock')
    encoded=cpu.encode(probes)
    assert encoded['input_ids'].shape==(4,512)
    assert np.all(encoded['input_ids'][:,0]==101)
    assert np.all(encoded['token_type_ids'][:,0]==0) and np.any(encoded['token_type_ids']==1)
    import torch
    inputs=cpu.prepare(encoded)
    with torch.inference_mode():
        output=cpu.model(**inputs,output_hidden_states=True)
        pooled=cpu.model.bert(**inputs).pooler_output.numpy()
    cpu_scores=output.logits[:,0].numpy()
    report={'device':a.device,'scope':'Feasibility/diagnostic only; no optimization winner selection',
            'environment':environment(),'paths':{},'CPU_scores':cpu_scores.tolist(),
            'tokenizer_class':type(cpu.tokenizer).__name__,'input_shape':list(encoded['input_ids'].shape),
            'special_token_segment_mask_verified':True,'cells':[]}
    del cpu;gc.collect()
    for path in (('stock','mlx') if a.device=='mps' else ('stock',)):
        model=Inference(a.device,path)
        values=model.cpu(model.forward(model.prepare(encoded)))
        report['paths'][path]={'CPU_score_max_abs':float(np.max(np.abs(np.array(values)-cpu_scores))),
                               'dispatch':model.dispatch(probes),'memory':model.memory()}
        if path=='mlx':
            import mlx.core as mx
            states,pool,scores=model.model(**model.prepare(encoded),trace=True)
            mx.eval(states,pool,scores)
            report['paths'][path]['layer_CPU_max_abs']=[float(np.max(np.abs(np.array(x)-y.numpy()))) for x,y in zip(states,output.hidden_states)]
            report['paths'][path]['pooler_CPU_max_abs']=float(np.max(np.abs(np.array(pool)-pooled)))
        for length in (128,256,512):
            model.length=length
            sample=[(data[0]['query'],t) for t in data[0]['texts'][:64]]
            for batch in (1,8,16,32,64):
                cell={'path':path,'L':length,'B':batch}
                try:
                    chunk=sample[:batch]; inputs=model.prepare(model.encode(chunk))
                    for _ in range(3):model.forward(inputs)
                    model.sync();forward=[];direct=[]
                    for _ in range(3):
                        t=time.perf_counter();model.forward(inputs);model.sync();forward.append(time.perf_counter()-t)
                        t=time.perf_counter();model.score(chunk);model.sync();direct.append(time.perf_counter()-t)
                    cell.update(status='FEASIBLE',forward_ms=float(np.median(forward)*1000),
                        direct_ms=float(np.median(direct)*1000),memory=model.memory(),shape=list(inputs['input_ids'].shape))
                except Exception as error:
                    cell.update(status='FAILED',error_type=type(error).__name__,error=str(error))
                report['cells'].append(cell)
                save(HERE/f'{a.device}_preflight.json',report)
                if time.perf_counter()-started>1800:
                    raise RuntimeError('30-minute feasibility ceiling reached')
        model.length=512
        requests=[]
        for row in (data[0],data[16],data[32],data[48]):
            for k in (10,50,100):
                t=time.perf_counter();model.request(row,k,16);model.sync()
                requests.append({'dataset':row['dataset'],'K':k,'seconds':time.perf_counter()-t})
        report['paths'][path]['request_probes']=requests
        del model;gc.collect()
    report.update(status='COMPLETE',total_seconds=time.perf_counter()-started)
    save(HERE/f'{a.device}_preflight.json',report)
    print(json.dumps({'total_seconds':report['total_seconds'],'paths':report['paths']},indent=2))


if __name__=='__main__':main()
