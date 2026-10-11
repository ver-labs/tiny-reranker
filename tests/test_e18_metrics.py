import math
import pytest
from experiments.E18.metrics import metrics


def test_unretrieved_positives_reduce_ndcg_and_recall():
    result = metrics(['a', 'negative'], {'a': 1, 'missing': 1})
    assert result['ndcg_at_10'] == pytest.approx(1 / (1 + 1 / math.log2(3)))
    assert result['recall_at_100'] == .5
    assert result['mrr_at_10'] == 1


def test_graded_qrels_use_linear_trec_gain():
    result = metrics(['b', 'a'], {'a': 2, 'b': 1})
    assert result['ndcg_at_10'] == pytest.approx((1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3)))


def test_no_retrieved_positive_is_retained_as_zero():
    assert all(value == 0 for value in metrics(['negative'], {'missing': 1}).values())
    with pytest.raises(ValueError): metrics(['a', 'a'], {'a': 1})
