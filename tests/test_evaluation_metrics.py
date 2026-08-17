from agentic_rag.evaluation.metrics import evaluate_ranking


def test_deterministic_retrieval_metrics_and_media_recall():
    results = [
        {"id": "a", "media_refs": [{"media_id": "img_1"}]},
        {"id": "b", "media_refs": []},
        {"id": "c", "media_refs": []},
    ]
    metrics = evaluate_ranking(results, ["b", "missing"], ["img_1", "img_2"], k=2)
    assert metrics["precision_at_k"] == .5
    assert metrics["recall_at_k"] == .5
    assert metrics["mrr"] == .5
    assert 0 < metrics["ndcg_at_k"] < 1
    assert metrics["media_recall_at_k"] == .5
