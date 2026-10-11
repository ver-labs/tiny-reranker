"""Session units for timing; query units for accuracy. No request pseudo-replication."""
import numpy as np
from .common import SEED


def interval(values, draws=10000):
    x=np.asarray(values,dtype=float)
    if not len(x) or not np.isfinite(x).all():raise ValueError('Finite nonempty units required')
    rng=np.random.default_rng(SEED)
    means=x[rng.integers(0,len(x),(draws,len(x)))].mean(axis=1)
    return {'mean':float(x.mean()),'ci95':np.quantile(means,[.025,.975]).tolist(),'units':len(x)}


def paired_ratio(records,baseline,candidate,key='seconds'):
    indexed={}
    for r in records:
        if r['path'] not in (baseline,candidate):continue
        identity=(r['session'],r['repetition'])
        cell=indexed.setdefault(identity,{})
        if r['path'] in cell:raise ValueError('Duplicate paired observation')
        cell[r['path']]=r[key]
    if not indexed or any(set(v)!={baseline,candidate} for v in indexed.values()):
        raise ValueError('Incomplete paired observations')
    sessions={}
    for (s,rep),v in indexed.items():
        if min(v.values())<=0:raise ValueError('Nonpositive measurement')
        # Lower elapsed time means higher throughput.
        sessions.setdefault(s,[]).append(np.log(v[baseline]/v[candidate]))
    result=interval([np.mean(sessions[s]) for s in sorted(sessions)])
    result.update(ratio=float(np.exp(result.pop('mean'))),ci95=np.exp(result['ci95']).tolist(),
                  independent_unit='session mean log ratio',paired_repetitions=len(indexed),
                  interval_interpretation='Unadjusted exploratory 95% interval')
    result['practical_primary_gain']=result['ci95'][0]>1.05
    return result


def distribution(values):
    x=np.asarray(values,dtype=float)
    if not len(x) or not np.isfinite(x).all():raise ValueError('Finite nonempty values required')
    return dict(zip(('min','p25','median','p75','p95','p99','max'),np.quantile(x,[0,.25,.5,.75,.95,.99,1]).tolist()))


def accuracy_gate(reference,candidate):
    from .metrics import metrics
    from .common import ranked
    if len(reference)!=len(candidate):raise ValueError('Record count mismatch')
    errors=[];relative=[];strict=True;cells={}
    for a,b in zip(reference,candidate):
        if (a['dataset'],a['query_id'],a['k'],a['ids'])!=(b['dataset'],b['query_id'],b['k'],b['ids']):
            raise ValueError('Candidate identity/order mismatch')
        x,y=np.asarray(a['scores']),np.asarray(b['scores'])
        if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Nonfinite scores')
        error=np.abs(x-y);errors.extend(error);relative.extend(error/np.maximum(np.abs(x),1e-3))
        strict &= bool(np.all(error<=2e-5+1e-4*np.abs(x)))
        ar,br=ranked(a['ids'],x),ranked(b['ids'],y)
        am,bm=metrics(ar,a['qrels']),metrics(br,a['qrels'])
        cell=cells.setdefault(f"{a['dataset']}/K{a['k']}",[])
        cell.append({'query_id':a['query_id'],'full_ranking_changed':ar!=br,'top1_changed':ar[0]!=br[0],
            'top10_membership_changed':set(ar[:10])!=set(br[:10]),'top10_order_changed':ar[:10]!=br[:10],
            'ndcg_delta':bm['ndcg_at_10']-am['ndcg_at_10'],'mrr_delta':bm['mrr_at_10']-am['mrr_at_10'],
            'reference_metrics':am,'candidate_metrics':bm,
            'min_adjacent_gap':float(np.min(np.abs(np.diff(sorted(x)))))})
    summaries={};passed=max(errors)<=.01
    for key,items in cells.items():
        summary={'queries':len(items)}
        for field in ('full_ranking_changed','top1_changed','top10_membership_changed','top10_order_changed'):
            summary[field]=sum(r[field] for r in items)
        for field in ('ndcg_delta','mrr_delta'):summary[field]=interval([r[field] for r in items])
        summary['application_pass']=summary['top1_changed']/len(items)<=.01 and summary['top10_membership_changed']/len(items)<=.01 and all(summary[f]['ci95'][0]>=-.001 for f in ('ndcg_delta','mrr_delta'))
        passed &= summary['application_pass'];summaries[key]=summary
    return {'strict_numerical_pass':strict,'application_pass':bool(passed),'absolute_error':distribution(errors),
            'relative_error_floor_1e-3':distribution(relative),'cells':summaries,'per_query':cells}
