"""User-authorized smaller first study; immutable before headline measurements."""
import json
from .common import HERE,ROOT,fixtures,sha,save,verify_assets


def main():
    if (HERE/'plan.json').exists():raise RuntimeError('Plan already frozen; do not overwrite')
    chosen=fixtures();small=chosen[:8]+chosen[32:40]
    model=verify_assets()
    files={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'data/e16/development.jsonl',HERE/'model_manifest.json',HERE/'data_manifest.json']}
    plan={'version':'E18-small-v1','seed':1801,'authorization':'User selected smaller first study before headline collection; retain K/C, accuracy gates, session repetitions.',
      'model':model,'files':files,'document_counts':[10,50,100],'length':512,'pair_microbatch':16,'concurrency':[1,4,8],
      'precision':'FP32; TF32 disabled; highest float32 matmul precision',
      'tokenization':'Pinned HF fast tokenizer CPU; query first, document second; longest_first truncation, dynamic padding; scoring ordered by ascending document ID after unchanged BM25 prefix selection',
      'timing_ids':[[r['dataset'],r['query_id']] for r in small],
      'sweep_ids':[[r['dataset'],r['query_id']] for r in (small[:2]+small[8:10])],
      'reserve_ids':[[r['dataset'],r['query_id']] for r in fixtures(True)],
      'reduction':'HTTP 16 requests/repetition (one pass of 16), replacing 256/64-query waves; secondary direct sweep 4 queries; full accuracy 485 queries unchanged. No original-matrix headline claim.',
      'accuracy':{'document_counts':[10,50,100],'queries':485,'pair_occurrences_per_path':77600,'absolute_tolerance':2e-5,'relative_tolerance':1e-4,'application_max_error':.01,'top1_fraction_per_dataset_k':.01,'top10_membership_fraction_per_dataset_k':.01,'quality_lower_ci':-.001,'bootstrap_draws':10000,'relative_floor':1e-3,'tie_break':'descending raw logit then ascending document ID','http_atol':2e-5,'http_rtol':1e-4},
      'paths':{'mps':['stock','mlx','compiled','mlx_compiled','metal','metal_compiled'],'cuda':['stock','explicit','compiled','cuda','triton']},
      'direct':{'mps_sessions':6,'cuda_sessions':5,'repetitions_per_session':3,'primary':{'k':100,'batch':16,'length':512,'scope':'scoring'},'primary_queries':16,'secondary_queries':4,'cells':[[k,16,512] for k in [10,50,100]]+[[100,b,512] for b in [1,8,32,64]]+[[100,16,l] for l in [128,256]],'warmup':'Every measured tokenized shape signature once before each scope, plus three primary forwards; record seconds/signatures; compile excluded'},
      'http':{'sessions':8,'repetitions_per_session':2,'requests_per_repetition':16,'queue_capacity':32,'workers':1,'cross_request_batching':False,'request_timeout_seconds':120,'worker_timeout_seconds':120,'startup_timeout_seconds':900,'load':'fixed-concurrency closed-loop local loopback persistent connections','paths':[['python','stock'],['rust','stock'],['python','mlx'],['rust','mlx']],'primary':[100,4],'response':'All K IDs and raw scores; validated against same-path direct audit after full-response latency timer; failures retained','optimized':'Python coordinator for qualified paths with direct primary interval lower bound >1.05 versus own framework; individual/combined separately; no NVIDIA HTTP speedup claim'},
      'statistics':{'unit':'session mean matched log-time ratio','draws':10000,'seed':1801,'interval':'Unadjusted exploratory 95%; no confirmatory family discovery claims','meaningful_gain_lower_ratio':1.05,'outliers':'retain all; exclude only declared technical failure, retaining evidence'},
      'ordering':'Balanced cyclic path order across N fresh sessions, 3 repetitions each; HTTP four startup slots each twice; seed 1801 shuffles cell and paired fixture order',
      'scopes':{'operation':'GPU events on CUDA; MLX benchmark device timing on Apple if supported, otherwise synchronized amortized wall labeled explicitly','forward':'Prepared resident inputs, synchronization excludes CPU conversion','scoring':'CPU tokenizer + construction/transfers + forward + CPU-readable scores; excludes ranking sort','http':'Client full-response latency including queue/IPC/scoring/sorting/serialization'},
      'compile':'Inductor fullgraph=True dynamic=True; MLX compile shapeless=False; 900-second startup ceiling; first-use cache unknown unless isolated; all measured signatures warmed; failures retained',
      'reserve':'All 421 untouched reserve queries K100/B16/L512 once per frozen qualified chosen framework and stock baseline, confirmation only, no timing CI',
      'estimate':{'original_http_gpu_hours':45,'small_http_gpu_hours':2.8125,'total_apple_hours_estimate':[4,8],'storage_gb_estimate':2,'caveat':'Preflight probes, estimates exclude new compile shape overhead; bounded failures recorded'}}
    save(HERE/'plan.json',plan);print('Frozen',sha(HERE/'plan.json'))

if __name__=='__main__':main()
