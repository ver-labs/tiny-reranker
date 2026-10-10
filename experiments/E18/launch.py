"""Resumable sequential GPU execution; fresh process for every path/session."""
import argparse
import json
import os
import time
import subprocess
import sys
from .common import ROOT,HERE,verify_plan,sha,save


def run(module,args,timeout=None):
    command=[sys.executable,'-u','-m','experiments.E18.'+module,*args]
    try:
        subprocess.run(command,cwd=ROOT,check=True,timeout=timeout)
    except subprocess.TimeoutExpired:
        save(HERE/'raw'/('timeout_'+module+'_'+'_'.join(args).replace('/','_')+'.json'),{'status':'TIMEOUT','command':command,'timeout_seconds':timeout,'plan_sha256':sha(HERE/'plan.json')})
        raise


def complete(path):
    if not path.exists():return False
    report=json.loads(path.read_text())
    if report.get('plan_sha256')!=sha(HERE/'plan.json'):raise RuntimeError('Existing artifact belongs to a different plan')
    return report.get('status') in ('VERIFIED','FAILED_ACCURACY','FAILED_EXECUTION','COMPLETE')


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',required=True,choices=['mps','cuda']);p.add_argument('--wait-pid',type=int);p.add_argument('--stage',choices=['accuracy','direct','http','all'],default='all');a=p.parse_args()
    if a.wait_pid:
        while True:
            try:os.kill(a.wait_pid,0)
            except ProcessLookupError:break
            time.sleep(5)
    plan=verify_plan();paths=plan['paths'][a.device]
    if a.stage in ('accuracy','all'):
        for path in paths:
            dest=HERE/f'{a.device}_{path}_accuracy.json'
            if not complete(dest):run('accuracy',['--device',a.device,'--path',path])
    if a.stage in ('direct','all'):
        for session in range(plan['direct'][a.device+'_sessions']):
            order=paths[session%len(paths):]+paths[:session%len(paths)]
            for path in order:
                # Execution failures are reported and not silently replaced.
                accuracy=json.loads((HERE/f'{a.device}_{path}_accuracy.json').read_text())
                if accuracy['status']=='FAILED_EXECUTION':continue
                run('direct_benchmark',['--device',a.device,'--path',path,'--session',str(session)])
    if a.device=='mps' and a.stage in ('http','all'):
        paths=plan['http']['paths']
        for session in range(plan['http']['sessions']):
            offset=session%len(paths);order=paths[offset:]+paths[:offset]
            for coordinator,path in order:run('http_benchmark',['--session',str(session),'--coordinator',coordinator,'--path',path])

if __name__=='__main__':main()
