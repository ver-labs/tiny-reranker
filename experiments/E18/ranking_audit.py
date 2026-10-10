"""Additional ranking diagnostics derived from saved scores, no new inference."""
import gzip
import json
from .common import HERE,save,sha


def read(path):
    with gzip.open(path,'rt') as f:return [json.loads(line) for line in f]


def main():
    output={}
    for path in sorted(HERE.glob('*_accuracy.json')):
        report=json.loads(path.read_text())
        if 'artifact' not in report:continue
        device=report['device'];reference=read(HERE/'scores'/f'{device}_stock.jsonl.gz')
        refs={(r['dataset'],r['query_id'],r['k']):r for r in reference}
        artifact=HERE/report['artifact'];assert sha(artifact)==report['artifact_sha256']
        cells={}
        for r in read(artifact):
            a=refs[r['dataset'],r['query_id'],r['k']];old=a['ranking'];new=r['ranking']
            cell=cells.setdefault(f"{r['dataset']}/K{r['k']}",{'queries':0,'full_ranking_changed':0,'top10_order_changed':0,'top10_removed_items':0,'top10_added_items':0,'changed_queries':[]})
            removed=sorted(set(old[:10])-set(new[:10]));added=sorted(set(new[:10])-set(old[:10]))
            cell['queries']+=1;cell['full_ranking_changed']+=old!=new;cell['top10_order_changed']+=old[:10]!=new[:10]
            cell['top10_removed_items']+=len(removed);cell['top10_added_items']+=len(added)
            if old!=new:cell['changed_queries'].append({'query_id':r['query_id'],'removed':removed,'added':added,'reference_top10':old[:10],'candidate_top10':new[:10]})
        output[path.stem]={'artifact_sha256':sha(artifact),'cells':cells}
    save(HERE/'ranking_diagnostics.json',output)


if __name__=='__main__':main()
