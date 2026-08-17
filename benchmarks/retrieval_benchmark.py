"""Fixed-scale keyword retrieval benchmark; run from repository root."""

from statistics import median
from time import perf_counter
import json

from agentic_rag.processing.reranker import ScoredDocument
from agentic_rag.rag.hybrid_retriever import HybridRetriever


def percentile(values, fraction):
    values = sorted(values)
    return values[min(len(values) - 1, round((len(values) - 1) * fraction))]


def main(document_count=1000, query_count=100):
    docs = [ScoredDocument(doc_id=f"chunk_{index}", content=f"多模态检索工业基准 文档编号 {index} 区域收入 {index % 17}", score=1.0) for index in range(document_count)]
    retriever = HybridRetriever(vector_store=None, graph_store=None, embedder=None)
    started = perf_counter(); retriever.build_index(docs); index_ms = (perf_counter() - started) * 1000
    latencies = []
    for index in range(query_count):
        started = perf_counter(); retriever.retrieve(f"区域收入 {index % 17}", top_k=5, use_vector=False); latencies.append((perf_counter() - started) * 1000)
    print(json.dumps({
        "documents": document_count, "queries": query_count, "index_ms": round(index_ms, 3),
        "latency_ms": {"p50": round(median(latencies), 3), "p95": round(percentile(latencies, .95), 3), "max": round(max(latencies), 3)},
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
