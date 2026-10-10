"""Correctness-sensitive benchmark gates, input integrity, and statistical units."""
import math
import numpy as np
import pytest
from experiments.E17.common import payload, ranked, errors, acceptance
from experiments.E17.http_benchmark import validate_response, aggregate
from experiments.E17.summarize import paired


def fixture():
    return {'dataset':'scifact','query_id':'q','query':'query',
            'document_ids':[str(i) for i in range(100)],'texts':[f'doc {i}' for i in range(100)]}


@pytest.mark.parametrize('k',[10,50,100])
def test_fixed_bm25_prefix_without_duplicates(k):
    r=fixture();p=payload(r,k)
    assert [d['document_id'] for d in p['documents']]==r['document_ids'][:k]
    assert len({d['text'] for d in p['documents']})==k and p['top_k']==k
    assert p['request_id']==f'scifact:q:{k}'
    assert r['document_ids']==[str(i) for i in range(100)]


def test_ties_finiteness_and_mismatched_records():
    assert ranked(['b','a'],[1.,1.])==['a','b']
    for ids,values in [(['a','b'],[math.nan,1]),(['a','a'],[1,2]),(['a'],[1,2])]:
        with pytest.raises(ValueError):ranked(ids,values)


def test_strict_fail_can_be_app_pass_but_failed_app_stays_diagnostic():
    tol={'application_max_absolute_error':.01,'application_top1_change_fraction':.01,
         'application_top10_membership_change_fraction':.01,'application_metric_loss':.001}
    e=errors([0.,1.],[.001,1.001]);changes={'top1_fraction':0.,'top10_membership_fraction':0.}
    deltas={'ndcg':{'ci95':[-.0005,.001]},'mrr':{'ci95':[0,0]}}
    r=acceptance(e,changes,deltas,tol)
    assert not r['strict_numerical_equivalence'] and r['application_accepted']
    assert e['reference_near_zero_count']==1 and e['relative_max']==1
    deltas['ndcg']['ci95'][0]=-.002
    r=acceptance(e,changes,deltas,tol)
    assert r['diagnostic_only'] and not r['accuracy_qualified']


def test_every_http_score_and_ranking_checked():
    p={'request_id':'q'};ref={'candidate_ids':['a','b'],'scores':{'stock':[1.,2.]},'rankings':{'stock':['b','a']}}
    body={'request_id':'q','ranking':[{'document_id':'b','score':2.},{'document_id':'a','score':1.}],
          'queue_ms':0.,'processing_ms':2.}
    t={'http_atol':2e-5,'http_rtol':1e-4}
    assert validate_response(body,p,ref,'stock',t)==0
    body['ranking'][1]['score']=1.1
    with pytest.raises(ValueError):validate_response(body,p,ref,'stock',t)
    body['ranking'].reverse()
    with pytest.raises(ValueError):validate_response(body,p,ref,'stock',t)


def test_percentiles_pairs_and_failures():
    observations=[{'success':True,'latency_ms':x,'queue_ms':1.,'processing_ms':2.} for x in (1.,2.,10.)]
    observations.append({'success':False,'timeout':True})
    r=aggregate(observations,2.,100)
    assert r['successful_requests']==3 and r['successful_pairs']==300
    assert r['requests_per_second']==1.5 and r['pairs_per_second']==150
    assert r['failures']==r['timeouts']==1
    assert r['p99_ms']==np.percentile([1,2,10],99)


def test_session_bootstrap_not_request_pseudoreplication():
    left=[{'session':s,'repeat':r,'speed':10.} for s in range(4) for r in range(3)]
    right=[{**r,'speed':20.} for r in left]
    result=paired(left,right,'speed')
    assert result['paired_repetitions']==12 and result['independent_sessions']==4
    assert np.allclose(result['ratio_ci95'],[2,2])
    assert result['interpretation']=='right advantage supported'
    with pytest.raises(ValueError):paired(left,right[:-1],'speed')
    with pytest.raises(ValueError):paired(left+[left[0]],right,'speed')
