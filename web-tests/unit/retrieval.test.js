import test from "node:test";
import assert from "node:assert/strict";
import { bm25Search, reciprocalRankFusion, retrieve, tokenize, vectorSearch } from "../../web/js/retrieval.js";

const corpus = [
  { id: "text_a", documentId: "doc", content: "季度营收增长", mediaRefs: [{ mediaId: "chart_a" }] },
  { id: "chart_a", documentId: "doc", content: "华东区域收入趋势图", modality: "image", provenance: { mediaId: "chart_a" } },
];
test("BM25 retrieves media-derived text", () => assert.equal(bm25Search("华东收入", corpus, 2)[0].id, "chart_a"));
test("RRF deduplicates while preserving references", () => {
  const fused = reciprocalRankFusion({ keyword: [{ ...corpus[0], score: 1 }], vector: [{ ...corpus[0], score: .8 }] });
  assert.equal(fused.length, 1); assert.equal(fused[0].mediaRefs[0].mediaId, "chart_a");
});
test("vector dimension failure degrades without losing keyword results", async () => {
  const output = await retrieve("季度营收", [{ ...corpus[0], embedding: [1, 0] }], { queryVector: [1, 0, 0] });
  assert.equal(output.degraded, true); assert.equal(output.results[0].id, "text_a");
});
test("tokenizer handles Chinese and English", () => { const terms = tokenize("RAG季度"); assert.ok(terms.includes("rag")); assert.ok(terms.includes("季度")); });
