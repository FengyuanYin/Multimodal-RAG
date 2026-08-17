import { LIMITS } from "./constants.js";

export function tokenize(text) {
  const normalized = String(text || "").normalize("NFKC").toLowerCase();
  const latin = normalized.match(/[a-z0-9][a-z0-9._-]*/g) || [];
  const han = [...normalized.matchAll(/[\p{Script=Han}]+/gu)].flatMap(([term]) => term.length === 1 ? [term] : [...term].flatMap((char, index, chars) => index < chars.length - 1 ? [char, char + chars[index + 1]] : [char]));
  return [...latin, ...han];
}

export function bm25Search(query, corpus, topK = LIMITS.topK, filter = () => true) {
  const docs = corpus.filter(filter);
  if (!docs.length) return [];
  const terms = docs.map((doc) => tokenize(doc.content));
  const queryTerms = [...new Set(tokenize(query))];
  const avgdl = terms.reduce((sum, row) => sum + row.length, 0) / docs.length || 1;
  const df = new Map();
  queryTerms.forEach((term) => df.set(term, terms.filter((row) => row.includes(term)).length));
  return docs.map((doc, index) => {
    const frequencies = new Map(); terms[index].forEach((term) => frequencies.set(term, (frequencies.get(term) || 0) + 1));
    let score = 0;
    for (const term of queryTerms) {
      const freq = frequencies.get(term) || 0; if (!freq) continue;
      const idf = Math.log(1 + (docs.length - (df.get(term) || 0) + .5) / ((df.get(term) || 0) + .5));
      score += idf * (freq * 2.2) / (freq + 1.2 * (.25 + .75 * terms[index].length / avgdl));
    }
    return { ...doc, score, channelScores: { keyword: score }, source: "keyword" };
  }).filter((item) => item.score > 0).sort((a, b) => b.score - a.score || a.id.localeCompare(b.id)).slice(0, topK);
}

export function vectorSearch(queryVector, corpus, topK = LIMITS.topK) {
  if (!Array.isArray(queryVector) || !queryVector.length) throw new Error("查询向量为空");
  const norm = Math.hypot(...queryVector) || 1;
  return corpus.filter((item) => Array.isArray(item.embedding)).map((item) => {
    if (item.embedding.length !== queryVector.length) throw new Error(`向量维度不一致: ${item.embedding.length}/${queryVector.length}`);
    const denominator = norm * (Math.hypot(...item.embedding) || 1);
    const score = item.embedding.reduce((sum, value, index) => sum + value * queryVector[index], 0) / denominator;
    return { ...item, score, channelScores: { vector: score }, source: "vector" };
  }).sort((a, b) => b.score - a.score || a.id.localeCompare(b.id)).slice(0, topK);
}

export function reciprocalRankFusion(channels, topK = LIMITS.topK, k = LIMITS.rrfK) {
  const fused = new Map();
  for (const [channel, items] of Object.entries(channels)) items.forEach((item, rank) => {
    const current = fused.get(item.id) || { ...item, channelScores: {}, fusedScore: 0, mediaRefs: item.mediaRefs || [], provenance: item.provenance || {} };
    current.channelScores[channel] = item.channelScores?.[channel] ?? item.score;
    current.fusedScore += 1 / (k + rank + 1);
    current.mediaRefs = current.mediaRefs.length ? current.mediaRefs : (item.mediaRefs || []);
    fused.set(item.id, current);
  });
  return [...fused.values()].sort((a, b) => b.fusedScore - a.fusedScore || a.id.localeCompare(b.id)).slice(0, topK);
}

export async function retrieve(query, corpus, options = {}) {
  const started = performance.now();
  const trace = { requestId: crypto.randomUUID(), requestedMode: options.mode || "hybrid", activeChannels: [], degradedChannels: [], candidateCounts: {}, stageLatencyMs: {}, warnings: [] };
  const channels = {};
  const keywordStart = performance.now();
  channels.keyword = bm25Search(query, corpus, options.topK);
  trace.activeChannels.push("keyword"); trace.candidateCounts.keyword = channels.keyword.length; trace.stageLatencyMs.keyword = performance.now() - keywordStart;
  if (options.queryVector) {
    const vectorStart = performance.now();
    try { channels.vector = vectorSearch(options.queryVector, corpus, options.topK); trace.activeChannels.push("vector"); trace.candidateCounts.vector = channels.vector.length; }
    catch (error) { trace.degradedChannels.push({ channel: "vector", code: "VECTOR_UNAVAILABLE", reason: error.message, recoverable: true }); }
    trace.stageLatencyMs.vector = performance.now() - vectorStart;
  }
  const results = reciprocalRankFusion(channels, options.topK);
  trace.stageLatencyMs.total = performance.now() - started;
  return { results, trace, degraded: trace.degradedChannels.length > 0 };
}
