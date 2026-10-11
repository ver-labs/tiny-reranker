"""Same bounded JSON-lines protocol behind both E18 coordinators."""
import argparse
from contextlib import redirect_stdout
import json
import os
import sys
from pathlib import Path
# Rust invokes this by file path.
if not __package__:
    # File execution otherwise exposes profile.py as the stdlib profile module
    # when Transformers imports cProfile. Match the package launch import path.
    script_dir=Path(__file__).resolve().parent
    sys.path[:]=[entry for entry in sys.path if Path(entry or os.getcwd()).resolve()!=script_dir]
    sys.path.insert(0,str(script_dir.parents[1]));__package__='experiments.E18'
MAX_FRAME=8*1024*1024


def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['mlx','pytorch'],required=True);p.add_argument('--pair-batch-size',type=int,choices=range(1,65),default=16);a=p.parse_args()
    protocol=sys.stdout
    with redirect_stdout(sys.stderr):
        from .common import verify_plan,rows,ranked
        from .inference import Inference
        from .direct_benchmark import selected
        from experiments.S1.serve import Request
        plan=verify_plan()
        path=os.environ.get('E18_INFERENCE_PATH','stock' if a.backend=='pytorch' else 'mlx')
        if (path in ('mlx','mlx_compiled','metal','metal_compiled')) != (a.backend=='mlx'):
            raise ValueError('Backend/path mismatch')
        backend=Inference('mps',path)
        # Warm every HTTP shape before READY; startup includes this cost.
        seen=set()
        for row in selected(plan['timing_ids']):
            for k in plan['document_counts']:
                docs=sorted(zip(row['document_ids'][:k],row['texts'][:k]))
                for offset in range(0,k,a.pair_batch_size):
                    pp=[(row['query'],text) for doc,text in docs[:k][offset:offset+a.pair_batch_size]]
                    encoded=backend.encode(pp);signature=tuple(encoded['input_ids'].shape)
                    if signature not in seen:
                        backend.forward(backend.prepare(encoded));backend.sync();seen.add(signature)
    def send(body):
        wire=json.dumps(body,ensure_ascii=False,allow_nan=False).encode()+b'\n'
        if len(wire)>MAX_FRAME:raise RuntimeError('Oversized output')
        protocol.buffer.write(wire);protocol.buffer.flush()
    send({'type':'ready','protocol_version':1,'backend':a.backend,'pair_batch_size':a.pair_batch_size,
          'max_length':512,'checkpoint_sha256':plan['model']['file_sha256']['model.safetensors']})
    while True:
        wire=sys.stdin.buffer.readline(MAX_FRAME+1)
        if not wire:break
        if len(wire)>MAX_FRAME or not wire.endswith(b'\n'):raise RuntimeError('Invalid frame')
        message=json.loads(wire)
        if message.get('type')!='rerank' or not isinstance(message.get('seq'),int):raise RuntimeError('Invalid protocol')
        request_id=message['request']['request_id']
        try:
            with redirect_stdout(sys.stderr):
                request=Request.model_validate(message['request']);docs=sorted(request.documents,key=lambda d:d.document_id)
                scores=[]
                for offset in range(0,len(docs),a.pair_batch_size):
                    scores+=backend.score([(request.query,d.text) for d in docs[offset:offset+a.pair_batch_size]])
                mapping=dict(zip((d.document_id for d in docs),scores));order=ranked(list(mapping),list(mapping.values()))
                ranking=[{'document_id':doc,'score':mapping[doc]} for doc in order[:request.top_k]]
            send({'type':'result','seq':message['seq'],'request_id':request_id,'ranking':ranking})
        except Exception as e:
            print(repr(e),file=sys.stderr,flush=True)
            send({'type':'result','seq':message['seq'],'request_id':request_id,'error':'Inference failed'})

if __name__=='__main__':main()
