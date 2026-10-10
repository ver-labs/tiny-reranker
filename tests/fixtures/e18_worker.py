"""Deterministic protocol fixture: no model imports or GPU execution."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

p=argparse.ArgumentParser()
p.add_argument('--backend')
p.add_argument('--pair-batch-size',type=int)
a=p.parse_args()
print(json.dumps({'type':'ready','protocol_version':1,'backend':a.backend,
                  'pair_batch_size':a.pair_batch_size,'max_length':512,'checkpoint_sha256':'0'*64}),flush=True)
for line in sys.stdin:
    m=json.loads(line)
    r=m['request']; name=r['request_id']
    if os.environ.get('E18_FIXTURE_EVENTS'):
        with Path(os.environ['E18_FIXTURE_EVENTS']).open('a') as f: f.write(name+'\n')
    if name=='crash': os._exit(9)
    if name=='slow': time.sleep(.3)
    if name=='malformed': print('not-json',flush=True);continue
    if name=='oversized': print('x'*(8*1024*1024+1),flush=True);continue
    docs=sorted(r['documents'],key=lambda d:d['document_id'])
    ranking=[{'document_id':d['document_id'],'score':float(len(docs)-i)} for i,d in enumerate(docs)][:r.get('top_k',10)]
    if name=='badids': ranking[0]['document_id']='unknown'
    payload={'type':'result','seq':m['seq']+(1 if name=='badseq' else 0),'request_id':name,'ranking':ranking}
    if name=='error': payload.pop('ranking');payload['error']='controlled error'
    print(json.dumps(payload),flush=True)
