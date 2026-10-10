import copy
import numpy as np
import pytest
from experiments.E18.common import rows,fixtures,pairs,ranked
from experiments.E18.statistics import paired_ratio,accuracy_gate
from experiments.E18.http_benchmark import payload,validate,aggregate


def test_candidate_contract():
    data=rows();assert len(data)==485 and len(fixtures(True))==421
    for r in fixtures():
        for k in (10,50,100):
            request=payload(r,k,0)
            assert [d['document_id'] for d in request['documents']]==r['document_ids'][:k]
            assert len({d['document_id'] for d in request['documents']})==k
            expected=[(r['query'],t) for d,t in sorted(zip(r['document_ids'][:k],r['texts'][:k]))]
            assert pairs(r,k)==expected


def test_session_units_and_missing_pairs():
    records=[{'session':s,'repetition':r,'path':p,'seconds':(10 if p=='stock' else 5)} for s in range(5) for r in range(3) for p in ('stock','other')]
    result=paired_ratio(records,'stock','other');assert result['units']==5 and result['paired_repetitions']==15 and result['ratio']==2
    with pytest.raises(ValueError):paired_ratio(records[:-1],'stock','other')
    with pytest.raises(ValueError):paired_ratio(records+[records[0]],'stock','other')


def test_accuracy_gate_fails_drift_and_ranking():
    base=[{'dataset':'scifact','query_id':'q','k':10,'ids':['a','b'],'scores':[1.,0.],'qrels':{'a':1}}]
    assert accuracy_gate(base,copy.deepcopy(base))['application_pass']
    drift=copy.deepcopy(base);drift[0]['scores']=[1.1,0.]
    assert not accuracy_gate(base,drift)['application_pass']
    change=copy.deepcopy(base);change[0]['scores']=[0.,1.]
    result=accuracy_gate(base,change);assert not result['application_pass'] and result['cells']['scifact/K10']['top1_changed']==1
    bad=copy.deepcopy(base);bad[0]['scores']=[np.nan,0.]
    with pytest.raises(ValueError):accuracy_gate(base,bad)


def test_ties_response_and_throughput():
    assert ranked(['b','a'],[1.,1.])==['a','b']
    ref={'ranking':['a','b'],'ids':['a','b'],'scores':[1.,0.]}
    request={'request_id':'r'};body={'request_id':'r','ranking':[{'document_id':'a','score':1.},{'document_id':'b','score':0.}],'queue_ms':2.,'processing_ms':3.}
    assert validate(body,request,ref)==0
    bad=copy.deepcopy(body);bad['ranking'].reverse()
    with pytest.raises(ValueError):validate(bad,request,ref)
    obs=[{'success':True,'latency_ms':10.,'queue_ms':2.,'processing_ms':3.},{'success':False,'timeout':True}]
    result=aggregate(obs,2.,100);assert result['requests_per_second']==.5 and result['pairs_per_second']==50 and result['failures']==result['timeouts']==1


def test_no_cuda_fallback_contract():
    import torch
    from experiments.E18.torch_kernel import residual_norm
    r=torch.zeros(2,384);w=torch.ones(384)
    with pytest.raises(RuntimeError,match='no fallback'):residual_norm(r,r,w,w)
    with pytest.raises(ValueError):residual_norm(r[:,:128],r[:,:128],w,w)
    with pytest.raises(ValueError):residual_norm(r,r,w,w,eps=0)


def test_resume_refuses_changed_plan(tmp_path):
    import json
    from experiments.E18.launch import complete
    artifact=tmp_path/'old.json';artifact.write_text(json.dumps({'status':'COMPLETE','plan_sha256':'0'*64}))
    with pytest.raises(RuntimeError,match='different plan'):complete(artifact)


def test_unknown_path_and_cpu_optimization_never_fall_back():
    from experiments.E18.inference import Inference
    with pytest.raises(ValueError,match='no fallback'):Inference('mps','typo')
    with pytest.raises(ValueError,match='correctness diagnostics'):Inference('cpu','compiled')
    with pytest.raises(ValueError,match='cap'):Inference('mps','stock',length=513)


def test_saved_interval_audit_rejects_wrong_uncertainty():
    from experiments.E18.verify import verify_ratio
    records=[{'session':s,'repetition':r,'path':p,'seconds':(10+s if p=='stock' else 5)} for s in range(5) for r in range(3) for p in ('stock','other')]
    reported=paired_ratio(records,'stock','other')
    verify_ratio(records,'stock','other',reported)
    wrong=copy.deepcopy(reported);wrong['ci95']=[reported['ratio'],reported['ratio']]
    with pytest.raises(AssertionError):verify_ratio(records,'stock','other',wrong)


def test_failed_http_cannot_be_a_speed_winner():
    from experiments.E18.summarize import http_ratio,percentiles_or_missing
    records=[{'session':s,'repetition':0,'path':p,'seconds':(10 if p=='baseline' else 1),'failures':(0 if p=='baseline' else 16),'successful_requests':(16 if p=='baseline' else 0)} for s in range(8) for p in ('baseline','broken')]
    assert http_ratio(records,'baseline','broken')['status']=='UNQUALIFIED_HTTP_FAILURES'
    assert percentiles_or_missing([])['median'] is None
