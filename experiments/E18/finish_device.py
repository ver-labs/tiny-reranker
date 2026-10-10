"""Post-study diagnostics and frozen selection; never run beside headline GPU work."""
import argparse
import ast
import json
import os
import time
import subprocess
import sys
from .common import HERE,ROOT,save,verify_plan,sha
from .launch import run


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',required=True);p.add_argument('--wait-pid',type=int);a=p.parse_args()
    if a.wait_pid:
        while True:
            try:os.kill(a.wait_pid,0)
            except ProcessLookupError:break
            time.sleep(5)
    plan=verify_plan();run('summarize',[])
    summary=json.loads((HERE/'summary.json').read_text())
    primary=summary['direct'].get(a.device+"/('scoring', 100, 16, 512, 'primary')",{})
    if not primary:raise RuntimeError('Primary direct matrix missing')
    qualified={path:entry for path,entry in primary['distribution'].items() if entry['accuracy']=='VERIFIED' and entry['sessions']==plan['direct'][a.device+'_sessions'] and entry['repetitions']==3*plan['direct'][a.device+'_sessions']}
    if 'stock' not in qualified:raise RuntimeError('Cannot finalize without the complete qualified stock primary matrix')
    candidates=[(p,r) for p,r in primary['comparisons'].items() if p.startswith('stock/') and 'ratio' in r and p.split('/')[1] in qualified and r['ci95'][0]>1.05]
    best=max(candidates,key=lambda pair:pair[1]['ratio'])[0].split('/')[1] if candidates else 'stock'
    selection={'device':a.device,'plan_sha256':sha(HERE/'plan.json'),'selected':best,'rule':'Qualified primary scoring ratio versus stock, lower unadjusted session CI >1.05; otherwise keep stock. Highest point ratio among eligible candidates; exploratory selection, not multiple-testing confirmation.','comparisons':primary['comparisons'],'reserve_ids':plan['reserve_ids']}
    dest=HERE/f'{a.device}_selection.json'
    if dest.exists() and json.loads(dest.read_text())!=selection:raise RuntimeError('Selection already frozen differently')
    save(dest,selection)
    for session in range(6):
        output=HERE/'raw'/f'{a.device}_operation_s{session}.json'
        if not output.exists():run('operation_benchmark',['--device',a.device,'--session',str(session)])
    for path in (['stock','mlx'] if a.device=='mps' else ['stock']):
        output=HERE/f'{a.device}_{path}_profile.json'
        if not output.exists():run('profile',['--device',a.device,'--path',path])
    if not (HERE/f'{a.device}_context_check.json').exists():run('context_check',['--device',a.device])
    reserve_paths=['stock']+([best] if best!='stock' else [])
    if a.device=='mps':
        mlx_candidates=[(p,r) for p,r in primary['comparisons'].items() if p.startswith('mlx/') and 'ratio' in r and p.split('/')[1] in qualified and r['ci95'][0]>1.05]
        best_mlx=max(mlx_candidates,key=lambda pair:pair[1]['ratio'])[0].split('/')[1] if mlx_candidates else 'mlx'
        reserve_paths += [best_mlx]
        # Whole-request effects of individually eligible and combined paths.
        optimized=sorted({p.split('/')[1] for p,r in primary['comparisons'].items() if p.split('/')[0] in ('stock','mlx') and 'ratio' in r and r['ci95'][0]>1.05 and p.split('/')[1] in qualified and p.split('/')[1] not in ('stock','mlx')})
        families={}
        for path in optimized:
            baseline='mlx' if path in ('mlx_compiled','metal','metal_compiled') else 'stock'
            families.setdefault(baseline,[baseline]).append(path)
        for baseline,paths in families.items():
            sessions=((max(8,len(paths))+len(paths)-1)//len(paths))*len(paths)
            schedule={'baseline':baseline,'paths':paths,'sessions':sessions,'repetitions':2,'k':100,'concurrency':[1,4,8],'reason':'New matched baseline worker in every optimization session; balanced order; no comparison to earlier baseline wave'}
            save(HERE/f'optimized_{baseline}_http_plan.json',schedule)
            for session in range(sessions):
                offset=session%len(paths)
                for path in paths[offset:]+paths[:offset]:
                    run('http_benchmark',['--session',str(session),'--coordinator','python','--path',path,'--study','optimized_'+baseline,'--primary-only'])
    for path in dict.fromkeys(reserve_paths):
        if not (HERE/f'{a.device}_{path}_reserve.json').exists():run('reserve',['--device',a.device,'--path',path])
    run('summarize',[])

if __name__=='__main__':main()
