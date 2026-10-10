"""Lightweight read-only collection status; never imports a GPU framework."""
import gzip
import json
from datetime import datetime,timezone
from .common import HERE


def main():
    statuses={}
    for p in HERE.glob('*_accuracy.json'):
        r=json.loads(p.read_text());statuses[p.stem]=r['status']
    active=[]
    for p in (HERE/'scores').glob('mps_*.jsonl.gz'):
        if any(word in p.name for word in ('reserve','context','interrupted')):continue
        name=p.name.removesuffix('.jsonl.gz')
        if name+'_accuracy' in statuses:continue
        count=0
        try:
            with gzip.open(p,'rt') as f:
                for line in f:
                    if line.endswith('\n'):count+=1
        except EOFError:pass
        active.append(f'{name}: {count}/1455 accuracy records')
    direct=[];http=[]
    for p in (HERE/'raw').glob('mps_direct_s*.json'):
        r=json.loads(p.read_text());direct.append((p.stem,r.get('status','COLLECTING'),len(r['observations'])))
    for p in (HERE/'raw').glob('*http_s*.json'):
        r=json.loads(p.read_text());http.append((p.stem,r.get('status','COLLECTING'),len(r['runs'])))
    print(json.dumps({'utc':datetime.now(timezone.utc).isoformat(timespec='seconds'),'accuracy':statuses,'active':active,'mps_direct_complete':sum(r[1]=='COMPLETE' for r in direct),'mps_direct_active':[r for r in direct if r[1]!='COMPLETE'],'http_complete':sum(r[1]=='COMPLETE' for r in http),'http_active':[r for r in http if r[1]!='COMPLETE']},separators=(',',':')),flush=True)

if __name__=='__main__':main()
