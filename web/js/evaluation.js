const relevance = (expected, items) => items.map((item) => expected.has(item.id) || expected.has(item.documentId) ? 1 : 0);
export function precisionAtK(items, expected, k = 5) { if (!k) return 0; return relevance(new Set(expected), items.slice(0, k)).reduce((a, b) => a + b, 0) / k; }
export function recallAtK(items, expected, k = 5) { const wanted = new Set(expected); if (!wanted.size) return null; return relevance(wanted, items.slice(0, k)).reduce((a, b) => a + b, 0) / wanted.size; }
export function reciprocalRank(items, expected) { const wanted = new Set(expected); const rank = items.findIndex((item) => wanted.has(item.id) || wanted.has(item.documentId)); return rank < 0 ? 0 : 1 / (rank + 1); }
export function ndcgAtK(items, expected, k = 5) {
  const wanted = new Set(expected); if (!wanted.size) return null;
  const dcg = relevance(wanted, items.slice(0, k)).reduce((sum, rel, index) => sum + rel / Math.log2(index + 2), 0);
  const ideal = Array.from({ length: Math.min(k, wanted.size) }, (_, index) => 1 / Math.log2(index + 2)).reduce((a, b) => a + b, 0);
  return ideal ? dcg / ideal : 0;
}
export function mediaRecallAtK(items, expectedMedia, k = 5) {
  const actual = new Set(items.slice(0, k).flatMap((item) => [item.provenance?.mediaId, ...(item.mediaRefs || []).map((ref) => ref.mediaId)].filter(Boolean)));
  const expected = new Set(expectedMedia); if (!expected.size) return null;
  return [...expected].filter((id) => actual.has(id)).length / expected.size;
}
export async function runEvaluation(dataset, retrieveFn, { k = 5, onProgress = () => {} } = {}) {
  const cases = [];
  for (const [index, item] of dataset.entries()) {
    const started = performance.now();
    const output = await retrieveFn(item.query);
    const results = output.results || output;
    cases.push({ id: item.id || `case_${index + 1}`, query: item.query, precisionAtK: precisionAtK(results, item.expected || [], k), recallAtK: recallAtK(results, item.expected || [], k), mrr: reciprocalRank(results, item.expected || []), ndcgAtK: ndcgAtK(results, item.expected || [], k), mediaRecallAtK: mediaRecallAtK(results, item.expectedMedia || [], k), latencyMs: performance.now() - started, trace: output.trace || null });
    onProgress({ current: index + 1, total: dataset.length });
  }
  const mean = (field) => { const values = cases.map((item) => item[field]).filter((value) => value !== null); return values.length ? values.reduce((a, b) => a + b, 0) / values.length : null; };
  return { schemaVersion: 1, generatedAt: new Date().toISOString(), k, count: cases.length, summary: { precisionAtK: mean("precisionAtK"), recallAtK: mean("recallAtK"), mrr: mean("mrr"), ndcgAtK: mean("ndcgAtK"), mediaRecallAtK: mean("mediaRecallAtK"), latencyMs: mean("latencyMs") }, cases };
}
export function exportEvaluation(report) { return new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }); }
