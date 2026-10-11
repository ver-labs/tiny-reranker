"""Ranking metrics over all official qrels, including unretrieved positives."""
import math


def metrics(ranked_ids, qrels, k=10):
    if len(set(ranked_ids)) != len(ranked_ids): raise ValueError('Duplicate retrieved document')
    relevant = {doc: grade for doc, grade in qrels.items() if grade > 0}
    if not relevant: raise ValueError('Query has no positive qrels')
    top = ranked_ids[:k]
    gains = [qrels.get(doc, 0) for doc in top]
    ideal = sorted(relevant.values(), reverse=True)[:k]
    dcg = sum(grade / math.log2(i + 2) for i, grade in enumerate(gains))
    idcg = sum(grade / math.log2(i + 2) for i, grade in enumerate(ideal))
    first = next((i + 1 for i, grade in enumerate(gains) if grade > 0), None)
    return {'ndcg_at_10': dcg / idcg, 'mrr_at_10': 1 / first if first else 0.,
            'recall_at_10': sum(doc in relevant for doc in top) / len(relevant),
            'recall_at_100': sum(doc in relevant for doc in ranked_ids[:100]) / len(relevant)}


def assert_trec_parity(qrels, ranked_ids_by_query):
    import pytrec_eval
    # Force the documented document-ID tie break through unique ordinal scores.
    runs = {qid: {doc: float(len(ids) - rank) for rank, doc in enumerate(ids)}
            for qid, ids in ranked_ids_by_query.items()}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg_cut.10', 'recall.10,100'})
    evaluated = evaluator.evaluate(runs)
    for qid, ids in ranked_ids_by_query.items():
        ours = metrics(ids, qrels[qid])
        for name, key in [('ndcg_at_10', 'ndcg_cut_10'), ('recall_at_10', 'recall_10'), ('recall_at_100', 'recall_100')]:
            if not math.isclose(ours[name], evaluated[qid][key], abs_tol=1e-12):
                raise RuntimeError(f'Metric mismatch with trec_eval: {qid}/{name}')

