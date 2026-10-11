"""Scope-labeled figures; variation across repetitions, not request pseudo-CIs."""
import ast
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .common import HERE

COLORS={'stock':'#20639b','mlx':'#e07a22','compiled':'#7c4d9e','mlx_compiled':'#a95324','metal':'#2e8b57','metal_compiled':'#c34868','explicit':'#777777','cuda':'#2e8b57','triton':'#c34868'}


def bars(ax,x,entries,field,label,color):
    if any(e[field]['median'] is None for e in entries):return
    values=[e[field]['median'] for e in entries];low=[v-e[field]['p25'] for v,e in zip(values,entries)];high=[e[field]['p75']-v for v,e in zip(values,entries)]
    ax.errorbar(x,values,yerr=[low,high],label=label,color=color,marker='o',capsize=3,linewidth=1.5)


def main():
    summary=json.loads((HERE/'summary.json').read_text());out=HERE/'plots';out.mkdir(exist_ok=True);plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    if summary['http']:
        fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True)
        for col,c in enumerate((1,4,8)):
            cells=[summary['http'].get(f'K{k}/C{c}') for k in (10,50,100)]
            if any(e is None for e in cells):continue
            for path in ('python/stock','rust/stock','python/mlx','rust/mlx'):
                if not all(path in e['distribution'] for e in cells):continue
                entries=[e['distribution'][path] for e in cells];base=path.split('/')[1];color=COLORS[base] if path.startswith('python') else ('#72a5cf' if base=='stock' else '#eab47e')
                bars(axes[0,col],[10,50,100],entries,'p50_ms',path,color);bars(axes[1,col],[10,50,100],entries,'requests_per_second',path,color)
            axes[0,col].set_title(f'Concurrency {c}');axes[0,col].set_ylabel('Request P50 (ms)');axes[1,col].set_ylabel('Requests / second');axes[1,col].set_xlabel('Documents per request (K)')
            for ax in axes[:,col]:ax.set_ylim(bottom=0);ax.grid(alpha=.2);ax.set_xticks([10,50,100])
        axes[0,0].legend(fontsize=8);fig.suptitle('Local loopback HTTP · FP32 · B=16 · L≤512\nMedian and IQR across 16 repetitions / 8 sessions; 16 requests per repetition',fontsize=12);fig.tight_layout(rect=(0,0,1,.92));fig.savefig(out/'http_workloads.png',dpi=160);plt.close(fig)
    families=sorted({key.split('/')[0] for key in summary.get('optimized_http',{})})
    if families:
        fig,axes=plt.subplots(2,len(families),figsize=(7*len(families),7),squeeze=False)
        for col,family in enumerate(families):
            cells=[summary['optimized_http'].get(f'{family}/K100/C{c}') for c in (1,4,8)]
            if any(e is None for e in cells):continue
            for path in sorted(set.intersection(*(set(e['distribution']) for e in cells))):
                color=COLORS.get(path.split('/')[1],'#777')
                bars(axes[0,col],[1,4,8],[e['distribution'][path] for e in cells],'p50_ms',path,color)
                bars(axes[1,col],[1,4,8],[e['distribution'][path] for e in cells],'requests_per_second',path,color)
            axes[0,col].set_title(family+' · new matched baseline');axes[0,col].set_ylabel('Request P50 (ms)');axes[1,col].set_ylabel('Requests / second');axes[1,col].set_xlabel('Concurrent requests (C)')
            for ax in axes[:,col]:ax.set_ylim(bottom=0);ax.grid(alpha=.2);ax.set_xticks([1,4,8])
            axes[0,col].legend(fontsize=8)
        fig.suptitle('Conditional optimized HTTP follow-up · Python coordinator · K100 / B16 / L≤512\nMedian and IQR across fresh matched sessions; 16 requests per repetition',fontsize=12)
        fig.tight_layout(rect=(0,0,1,.92));fig.savefig(out/'optimized_http.png',dpi=160);plt.close(fig)
    for device in ('mps','cuda'):
        cells={}
        for key,value in summary['direct'].items():
            prefix,cell=key.split('/',1)
            if prefix==device:cells[ast.literal_eval(cell)]=value
        if not cells:continue
        fig,axes=plt.subplots(1,2,figsize=(12,4.8))
        for col,scope in enumerate(('forward','scoring')):
            for path in sorted({p for e in cells.values() for p in e['distribution']}):
                entries=[cells.get((scope,100,b,512,'sweep'),{}).get('distribution',{}).get(path) for b in (1,8,16,32,64)]
                if any(e is None for e in entries):continue
                bars(axes[col],[1,8,16,32,64],entries,'pairs_per_second',path,COLORS[path])
            axes[col].set_title('Prepared forward' if scope=='forward' else 'Complete direct scoring');axes[col].set_xlabel('Pair microbatch size (B)');axes[col].set_ylabel('Pairs / second');axes[col].set_xticks([1,8,16,32,64]);axes[col].set_ylim(bottom=0);axes[col].grid(alpha=.2)
        axes[0].legend(fontsize=8);fig.suptitle(f'{device.upper()} · FP32 · K=100 · L≤512 · same four queries\nMedian / IQR across repeated fresh sessions; accuracy status in tables',fontsize=12);fig.tight_layout(rect=(0,0,1,.89));fig.savefig(out/f'{device}_batching.png',dpi=160);plt.close(fig)
        entries=[]
        for cell,value in cells.items():
            if cell!=('scoring',100,16,512,'primary'):continue
            for name,r in value['comparisons'].items():
                if 'ratio' in r:entries.append((name,r))
        if entries:
            fig,ax=plt.subplots(figsize=(9,max(3,len(entries)*.45)))
            for y,(name,r) in enumerate(entries):
                ax.errorbar(r['ratio'],y,xerr=[[r['ratio']-r['ci95'][0]],[r['ci95'][1]-r['ratio']]],fmt='o',capsize=4,color='#20639b')
            ax.set_yticks(range(len(entries)),[name for name,r in entries]);ax.axvline(1,color='#555',linestyle='--');ax.axvline(1.05,color='#ba6332',linestyle=':');ax.set_xlabel('Paired throughput ratio; unadjusted session-bootstrap 95% interval');fig.suptitle(f'{device.upper()} primary direct endpoint · K100 / B16 / L≤512\nUnadjusted 95% session intervals; accuracy status in tables',fontsize=11);ax.grid(axis='x',alpha=.2);fig.tight_layout(rect=(0,0,1,.85));fig.savefig(out/f'{device}_primary_ratios.png',dpi=160);plt.close(fig)
    if summary['operation']:
        fig,axes=plt.subplots(1,2,figsize=(11,4.5))
        for col,device in enumerate(('mps','cuda')):
            cells=[(int(k.split('/')[1]),v) for k,v in summary['operation'].items() if k.startswith(device+'/')]
            if not cells:axes[col].set_visible(False);continue
            cells.sort()
            for path in sorted({p for n,e in cells for p in e['distribution']}):
                entries=[{path:e['distribution'][path]} for n,e in cells]
                bars(axes[col],[n for n,e in cells],entries,path,path,COLORS.get(path,'#777'))
            axes[col].set_title(device.upper()+(' · synchronized wall' if device=='mps' else ' · CUDA events'));axes[col].set_xlabel('Rows × hidden 384');axes[col].set_ylabel('Operation time (µs)');axes[col].set_ylim(bottom=0);axes[col].legend();axes[col].grid(alpha=.2)
        fig.suptitle('Resident FP32 residual addition + LayerNorm · ε=1e−12\nMedian / IQR across operation repetitions; device scopes differ',fontsize=12);fig.tight_layout(rect=(0,0,1,.88));fig.savefig(out/'operations.png',dpi=160);plt.close(fig)
    print('Figures regenerated',flush=True)

if __name__=='__main__':main()
