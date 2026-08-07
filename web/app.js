/* =========================================================================
 * PDF Chat — 浏览器本地 RAG（侧边栏导航版）
 * -------------------------------------------------------------------------
 * 核心原则：
 *   1. 所有文档解析与检索均在浏览器本地完成，不上传任何服务器
 *      （除非用户显式选择 MinerU 服务解析器，此时 PDF 会发送到其自配服务）。
 *   2. API Key 仅保存在浏览器 localStorage，仅发送到用户自配的 API 地址。
 *   3. 检索默认使用本地 BM25（中文 bigram 分词），可选 Embedding API 增强。
 *   4. 支持文档分类管理、按分类检索、无需检索问题直接回答。
 * ========================================================================= */
"use strict";

/* ───────────────────────── 工具函数 ───────────────────────── */
const $ = (id) => document.getElementById(id);

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function addMessage(role, text, extra) {
  const box = $("messages");
  const msg = el("div", `msg ${role}`);
  msg.textContent = text;
  if (extra) {
    const meta = el("div", "meta");
    if (typeof extra === "string") {
      meta.textContent = extra;
    } else {
      meta.append(...extra);
    }
    msg.appendChild(meta);
  }
  box.appendChild(msg);
  box.scrollTop = box.scrollHeight;
  return msg;
}

function setProgress(pct) {
  $("progress").classList.remove("hidden");
  $("progressBar").style.width = `${Math.max(2, pct)}%`;
}
function hideProgress() {
  $("progress").classList.add("hidden");
  $("progressBar").style.width = "0";
}

function uid(prefix) {
  return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

function fmtPages(n) {
  return `${n} 页`;
}

/* ───────────────────────── 设置管理 ───────────────────────── */
const Settings = {
  KEY: "pdfchat.settings.v1",
  load() {
    try {
      return JSON.parse(localStorage.getItem(this.KEY) || "{}");
    } catch {
      return {};
    }
  },
  save(s) {
    localStorage.setItem(this.KEY, JSON.stringify(s));
  },
};

/* ───────────────────────── 分类 / 文档持久化 ───────────────────────── */
const Store = {
  CATS_KEY: "pdfchat.cats.v1",
  DOCS_KEY: "pdfchat.docs.v1",
  loadCats() {
    try {
      const v = JSON.parse(localStorage.getItem(this.CATS_KEY) || "[]");
      return Array.isArray(v) ? v : [];
    } catch {
      return [];
    }
  },
  saveCats(cats) {
    localStorage.setItem(this.CATS_KEY, JSON.stringify(cats));
  },
  loadDocs() {
    try {
      const v = JSON.parse(localStorage.getItem(this.DOCS_KEY) || "[]");
      return Array.isArray(v) ? v : [];
    } catch {
      return [];
    }
  },
  saveDocs(docs) {
    try {
      localStorage.setItem(this.DOCS_KEY, JSON.stringify(docs));
    } catch (e) {
      // 文档文本可能超过 localStorage 配额：降级为仅会话内存并提示
      console.warn("文档持久化失败（可能超出本地存储配额），本次仅保存在内存中。", e);
    }
  },
  clearAll() {
    localStorage.removeItem(this.KEY);
    localStorage.removeItem(this.CATS_KEY);
    localStorage.removeItem(this.DOCS_KEY);
  },
};

/* ───────────────────────── 状态 ───────────────────────── */
const State = {
  // 文档列表：[{ id, name, cat, pages: [{page, text}], parser, addedAt }]
  docs: [],
  // 分块结果：[{ id, docId, file, page, text, tokens }]
  chunks: [],
  // BM25 数据结构
  bm25: null,
  // 可选向量数据：{ vectors: Float32Array, norm: Float32Array, dim: number }
  vectors: null,
};

/* ───────────────────────── 分词（中文 bigram + 英文单词） ───────────────────────── */
function tokenize(text) {
  const tokens = [];
  const en = text.toLowerCase().match(/[a-z0-9]+/g) || [];
  tokens.push(...en);
  const zh = text.match(/[\u4e00-\u9fa5]+/g) || [];
  for (const seg of zh) {
    if (seg.length === 1) {
      tokens.push(seg);
    } else {
      for (let i = 0; i < seg.length - 1; i++) {
        tokens.push(seg.slice(i, i + 2));
      }
    }
  }
  return tokens;
}

/* ───────────────────────── BM25 实现（支持分类过滤） ───────────────────────── */
function buildBM25(chunks) {
  const docTokens = chunks.map((c) => c.tokens);
  const N = docTokens.length;
  const df = new Map();
  const k1 = 1.5;
  const b = 0.75;
  const avgdl = docTokens.reduce((sum, t) => sum + t.length, 0) / Math.max(1, N);

  for (const tokens of docTokens) {
    const seen = new Set(tokens);
    for (const t of seen) df.set(t, (df.get(t) || 0) + 1);
  }

  function idf(t) {
    const n = df.get(t) || 0;
    return Math.log((N - n + 0.5) / (n + 0.5) + 1);
  }

  return {
    search(queryTokens, topK, filter) {
      const scores = new Array(N).fill(0);
      const qf = new Map();
      for (const t of queryTokens) qf.set(t, (qf.get(t) || 0) + 1);

      for (const [t, qcount] of qf) {
        const w = idf(t);
        if (w === 0) continue;
        for (let i = 0; i < N; i++) {
          const tf = docTokens[i].filter((x) => x === t).length;
          if (tf === 0) continue;
          const len = docTokens[i].length;
          scores[i] += w * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (len / avgdl))));
        }
      }

      const ranked = chunks
        .map((c, i) => ({ chunk: c, score: scores[i] }))
        .filter((r) => r.score > 0 && (!filter || filter(r.chunk)))
        .sort((a, b) => b.score - a.score)
        .slice(0, topK);
      return ranked;
    },
  };
}

/* ───────────────────────── 向量检索（可选） ───────────────────────── */
async function embedTexts(texts, settings) {
  const body = { model: settings.embedModel, input: texts };
  const resp = await fetch(apiUrl("/embeddings", settings), {
    method: "POST",
    headers: apiHeaders(settings),
    body: JSON.stringify(apiBody(body, settings)),
  });
  if (!resp.ok) {
    const err = await resp.text().catch(() => "");
    throw new Error(`Embedding API ${resp.status}: ${err.slice(0, 200)}`);
  }
  const data = await resp.json();
  return data.data.map((d) => d.embedding);
}

async function buildVectors(chunks, settings) {
  const batchSize = 64;
  const all = [];
  for (let i = 0; i < chunks.length; i += batchSize) {
    const batch = chunks.slice(i, i + batchSize).map((c) => c.text);
    const vecs = await embedTexts(batch, settings);
    all.push(...vecs);
  }
  const dim = all[0]?.length || 0;
  const vectors = new Float32Array(chunks.length * dim);
  const norm = new Float32Array(chunks.length);
  for (let i = 0; i < chunks.length; i++) {
    let sum = 0;
    for (let j = 0; j < dim; j++) {
      const v = all[i][j];
      vectors[i * dim + j] = v;
      sum += v * v;
    }
    norm[i] = Math.sqrt(sum) || 1;
  }
  return { vectors, norm, dim };
}

function vectorSearch(queryVec, state, topK, filter) {
  const { vectors, norm, dim } = state.vectors;
  const scores = [];
  let qsum = 0;
  for (let j = 0; j < dim; j++) qsum += queryVec[j] * queryVec[j];
  const qnorm = Math.sqrt(qsum) || 1;

  for (let i = 0; i < state.chunks.length; i++) {
    if (filter && !filter(state.chunks[i])) continue;
    let dot = 0;
    for (let j = 0; j < dim; j++) dot += vectors[i * dim + j] * queryVec[j];
    scores.push({ i, score: dot / (norm[i] * qnorm) });
  }
  return scores
    .filter((r) => r.score > 0.1)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
    .map((r) => ({ chunk: state.chunks[r.i], score: r.score }));
}

/* ───────────────────────── RRF 融合 ───────────────────────── */
function rrfMerge(lists, topK, k = 60) {
  const scoreMap = new Map();
  for (const list of lists) {
    list.forEach((item, rank) => {
      const id = item.chunk.id;
      const cur = scoreMap.get(id) || { chunk: item.chunk, score: 0 };
      cur.score += 1 / (k + rank + 1);
      scoreMap.set(id, cur);
    });
  }
  return [...scoreMap.values()]
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
    .map((r) => r.chunk);
}

/* ───────────────────────── API 请求封装（直连 / 同源代理） ───────────────────────── */
function apiUrl(path, settings) {
  if (settings.callMode === "proxy") return `/proxy${path}`;
  return `${settings.baseUrl}${path}`;
}
function apiHeaders(settings) {
  const headers = { "Content-Type": "application/json" };
  if (settings.callMode === "proxy") {
    if (settings.apiKey) headers["X-API-Key"] = settings.apiKey;
  } else if (settings.apiKey) {
    headers["Authorization"] = `Bearer ${settings.apiKey}`;
  }
  return headers;
}
function apiBody(body, settings) {
  if (settings.callMode === "proxy") return { ...body, base_url: settings.baseUrl };
  return body;
}

/* ───────────────────────── 解析器插件（可扩展） ───────────────────────── */
const Parsers = {
  local: {
    name: "本地解析",
    async parse(file) {
      const buf = await file.arrayBuffer();
      const pdf = await pdfjsLib.getDocument({ data: buf }).promise;
      const pages = [];
      for (let p = 1; p <= pdf.numPages; p++) {
        const page = await pdf.getPage(p);
        const content = await page.getTextContent();
        const lines = new Map();
        for (const item of content.items) {
          if (!("str" in item) || !item.str.trim()) continue;
          const y = Math.round(item.transform[5] / 4) * 4;
          const line = lines.get(y) || "";
          lines.set(y, line + item.str);
        }
        const text = [...lines.values()].join("\n");
        if (text.trim()) pages.push({ page: p, text });
      }
      return pages;
    },
  },
  mineru: {
    name: "MinerU 服务",
    async parse(file, settings) {
      const base = (settings.mineruUrl || "").replace(/\/+$/, "");
      if (!base) throw new Error("未配置 MinerU API 地址");
      const fd = new FormData();
      fd.append("file", file);
      const resp = await fetch(`${base}/file2text`, { method: "POST", body: fd });
      if (!resp.ok) {
        const err = await resp.text().catch(() => "");
        throw new Error(`MinerU ${resp.status}: ${err.slice(0, 200)}`);
      }
      const data = await resp.json();
      // 兼容两种返回格式：
      //   1) { text: "整篇文本" }
      //   2) { pages: [{ page: 1, text: "..." }] }
      if (Array.isArray(data.pages) && data.pages.length) {
        return data.pages.map((p) => ({
          page: p.page || 1,
          text: p.text || p.content || "",
        })).filter((p) => p.text.trim());
      }
      if (typeof data.text === "string" && data.text.trim()) {
        return [{ page: 1, text: data.text }];
      }
      throw new Error("MinerU 返回格式无法识别（期望 {text} 或 {pages:[{page,text}]}）");
    },
  },
};

/* ───────────────────────── 分块 ───────────────────────── */
function splitChunks(text, size, overlap) {
  const out = [];
  const step = Math.max(1, size - overlap);
  for (let i = 0; i < text.length; i += step) {
    out.push(text.slice(i, i + size));
  }
  return out;
}

function buildChunks(docs, chunkSize) {
  const overlap = Math.min(100, Math.floor(chunkSize * 0.15));
  const chunks = [];
  let seq = 0;
  for (const doc of docs) {
    for (const p of doc.pages) {
      for (const part of splitChunks(p.text, chunkSize, overlap)) {
        const text = part.trim();
        if (!text) continue;
        chunks.push({
          id: `c${seq++}`,
          docId: doc.id,
          file: doc.name,
          cat: doc.cat || "",
          page: p.page,
          text,
          tokens: tokenize(text),
        });
      }
    }
  }
  return chunks;
}

/* ───────────────────────── 无需检索判断 ───────────────────────── */
function isDirectAnswerable(q) {
  const t = q.trim().toLowerCase();
  if (t.length < 2) return false;
  // 问候 / 感谢 / 自我介绍 / 告别 / 简单能力询问
  const patterns = [
    /^(你好|您好|hi|hello|hey|嗨|哈喽|早上好|晚上好|下午好)[!！。.\s]*$/,
    /^(谢谢|感谢|多谢|thanks|thank you|thx)[!！。.\s]*$/i,
    /^(再见|拜拜|bye|goodbye)[!！。.\s]*$/i,
    /^(你是谁|你是什么|介绍一下你自己|介绍下你自己|what are you|who are you)\b?/i,
    /^(你能做什么|你能干什么|你会什么|可以做什么|what can you do)\b?/i,
  ];
  return patterns.some((re) => re.test(t));
}

/* ───────────────────────── 对话（OpenAI 兼容） ───────────────────────── */
async function askLLM(settings, systemPrompt, userContent) {
  const resp = await fetch(apiUrl("/chat/completions", settings), {
    method: "POST",
    headers: apiHeaders(settings),
    body: JSON.stringify(
      apiBody(
        {
          model: settings.model,
          messages: [
            { role: "system", content: systemPrompt },
            { role: "user", content: userContent },
          ],
          temperature: 0.2,
        },
        settings
      )
    ),
  });
  if (!resp.ok) {
    const err = await resp.text().catch(() => "");
    throw new Error(`LLM API ${resp.status}: ${err.slice(0, 300)}`);
  }
  const data = await resp.json();
  return data.choices?.[0]?.message?.content ?? "";
}

/* ───────────────────────── 设置表单 ───────────────────────── */
function loadSettingsIntoForm() {
  const s = Settings.load();
  $("apiKey").value = s.apiKey || "";
  $("callMode").value = s.callMode || "direct";
  $("baseUrl").value = s.baseUrl || "https://api.openai.com/v1";
  $("model").value = s.model || "gpt-4o-mini";
  $("embedModel").value = s.embedModel || "";
  $("topK").value = s.topK || 5;
  $("chunkSize").value = s.chunkSize || 800;
  $("parser").value = s.parser || "local";
  $("mineruUrl").value = s.mineruUrl || "";
  syncParserUI();
  syncModelTag();
  syncRetrievalStat();
}

function collectSettings() {
  return {
    apiKey: $("apiKey").value.trim(),
    callMode: $("callMode").value,
    baseUrl: $("baseUrl").value.trim().replace(/\/+$/, ""),
    model: $("model").value.trim(),
    embedModel: $("embedModel").value.trim(),
    topK: parseInt($("topK").value, 10) || 5,
    chunkSize: parseInt($("chunkSize").value, 10) || 800,
    parser: $("parser").value,
    mineruUrl: $("mineruUrl").value.trim().replace(/\/+$/, ""),
  };
}

function syncParserUI() {
  const useMineru = $("parser").value === "mineru";
  $("mineruConfig").classList.toggle("hidden", !useMineru);
  $("dzSub").textContent = useMineru
    ? "将由 MinerU 服务解析 · PDF 会发送到该服务"
    : "本地解析 · 不经过任何服务器";
}

function syncModelTag() {
  const tag = $("modelTag");
  if (!tag) return;
  const s = Settings.load();
  tag.textContent = s.model ? `模型 · ${s.model}` : "未配置";
}

function syncRetrievalStat() {
  const s = Settings.load();
  const parts = [s.parser === "mineru" ? "MinerU 解析" : "本地解析"];
  parts.push(s.embedModel ? "BM25 + 向量检索" : "BM25 关键词检索");
  $("retrievalStat").textContent = parts.join(" · ");
}

/* ───────────────────────── 视图切换（侧边栏） ───────────────────────── */
function switchView(name) {
  document.querySelectorAll(".nav-item").forEach((b) => {
    b.classList.toggle("active", b.dataset.view === name);
  });
  document.querySelectorAll(".view").forEach((v) => {
    v.classList.toggle("active", v.id === `view-${name}`);
  });
}

/* ───────────────────────── 分类渲染 ───────────────────────── */
function currentCats() {
  return State.cats || [];
}

function catName(id) {
  if (!id) return "未分类";
  const c = currentCats().find((x) => x.id === id);
  return c ? c.name : "未分类";
}

function renderCatList() {
  const list = $("catList");
  list.innerHTML = "";
  for (const cat of currentCats()) {
    const li = el("li");
    li.appendChild(el("span", "cat-icon", "📁"));
    li.appendChild(el("span", "cat-name", cat.name));
    const count = State.docs.filter((d) => d.cat === cat.id).length;
    li.appendChild(el("span", "cat-count", String(count)));
    const del = el("button", "cat-del", "✕");
    del.title = "删除分类（文档保留）";
    del.addEventListener("click", () => {
      State.cats = currentCats().filter((c) => c.id !== cat.id);
      State.docs.forEach((d) => {
        if (d.cat === cat.id) d.cat = "";
      });
      Store.saveCats(State.cats);
      Store.saveDocs(State.docs);
      renderAll();
    });
    li.appendChild(del);
    list.appendChild(li);
  }
}

function renderCatSelects() {
  // 上传目标分类
  const uploadCat = $("uploadCat");
  uploadCat.innerHTML = "";
  uploadCat.appendChild(new Option("未分类", ""));
  for (const cat of currentCats()) {
    uploadCat.appendChild(new Option(cat.name, cat.id));
  }

  // 对话检索范围
  const chatCat = $("chatCat");
  chatCat.innerHTML = "";
  chatCat.appendChild(new Option("全部文档", "all"));
  chatCat.appendChild(new Option("未分类", "__none__"));
  for (const cat of currentCats()) {
    chatCat.appendChild(new Option(`分类：${cat.name}`, `cat:${cat.id}`));
  }
}

function chatCatFilter() {
  const v = $("chatCat").value;
  if (!v || v === "all") return null;
  if (v === "__none__") return (chunk) => !chunk.cat;
  const catId = v.replace(/^cat:/, "");
  return (chunk) => chunk.cat === catId;
}

/* ───────────────────────── 文档渲染 ───────────────────────── */
function renderDocList() {
  const list = $("docList");
  list.innerHTML = "";
  for (const doc of State.docs) {
    const li = el("li");
    li.appendChild(el("span", "", doc.name));
    li.appendChild(el("span", "pages", fmtPages(doc.pages.length)));

    const sel = el("select", "cat-select");
    sel.appendChild(new Option("未分类", ""));
    for (const cat of currentCats()) {
      sel.appendChild(new Option(cat.name, cat.id));
    }
    sel.value = doc.cat || "";
    sel.addEventListener("change", () => {
      doc.cat = sel.value;
      Store.saveDocs(State.docs);
      renderCatList();
      renderCatSelects();
      invalidateIndex();
    });
    li.appendChild(sel);

    const del = el("button", "del-doc", "🗑");
    del.title = "删除文档";
    del.addEventListener("click", () => {
      State.docs = State.docs.filter((d) => d.id !== doc.id);
      Store.saveDocs(State.docs);
      renderAll();
    });
    li.appendChild(del);
    list.appendChild(li);
  }
  $("docStats").textContent = `共 ${State.docs.length} 个文档 / ${State.docs.reduce((s, d) => s + d.pages.length, 0)} 页`;
  $("docBadge").textContent = String(State.docs.length);
  $("buildIndex").disabled = State.docs.length === 0;
}

function invalidateIndex() {
  if (State.chunks.length || State.bm25) {
    State.chunks = [];
    State.bm25 = null;
    State.vectors = null;
    addMessage("system", "文档或分类已变化，请重新建立索引。");
  }
}

function renderAll() {
  renderCatList();
  renderCatSelects();
  renderDocList();
}

/* ───────────────────────── 上传处理 ───────────────────────── */
async function handleFiles(files) {
  if (!files.length) {
    addMessage("system", "请选择 PDF 文件。");
    return;
  }
  const settings = collectSettings();
  const parser = Parsers[settings.parser] || Parsers.local;
  const targetCat = $("uploadCat").value;

  for (const file of files) {
    addMessage("system", `正在解析：${file.name}（${parser.name}）…`);
    try {
      const pages = await parser.parse(file, settings);
      if (!pages.length) {
        addMessage("error", `解析 ${file.name} 未提取到文本内容。`);
        continue;
      }
      const doc = {
        id: uid("doc"),
        name: file.name,
        cat: targetCat,
        pages,
        parser: settings.parser,
        addedAt: Date.now(),
      };
      State.docs.push(doc);
      Store.saveDocs(State.docs);
      addMessage("system", `已添加 ${file.name}：${pages.length} 页${targetCat ? `（分类：${catName(targetCat)}）` : ""}。`);
    } catch (e) {
      addMessage("error", `解析 ${file.name} 失败：${e.message}`);
    }
  }
  invalidateIndex();
  renderAll();
}

/* ───────────────────────── 建立索引 ───────────────────────── */
async function handleBuildIndex() {
  if (!State.docs.length) {
    addMessage("system", "请先上传 PDF 文档。");
    return;
  }
  const settings = collectSettings();
  if (settings.embedModel && !settings.apiKey) {
    addMessage("error", "使用 Embedding 增强需要填写 API Key。");
    return;
  }

  $("buildIndex").disabled = true;
  try {
    State.chunks = buildChunks(State.docs, settings.chunkSize);
    setProgress(25);
    State.bm25 = buildBM25(State.chunks);
    State.vectors = null;
    setProgress(50);

    if (settings.embedModel && settings.apiKey) {
      addMessage("system", `正在生成向量索引（${State.chunks.length} 块）…`);
      State.vectors = await buildVectors(State.chunks, settings);
    }
    setProgress(100);
    addMessage(
      "system",
      `索引完成：${State.chunks.length} 个分块` +
        (State.vectors ? "（含向量检索）" : "（关键词 BM25 检索）") +
        "。可以开始提问。"
    );
  } catch (e) {
    addMessage("error", `建立索引失败：${e.message}`);
  } finally {
    hideProgress();
    $("buildIndex").disabled = false;
  }
}

/* ───────────────────────── 提问处理 ───────────────────────── */
async function handleAsk(question) {
  const settings = collectSettings();
  if (!settings.apiKey) {
    addMessage("error", "请先在「设置」中填写 API Key。");
    return;
  }
  if (!settings.model) {
    addMessage("error", "请填写模型名称。");
    return;
  }

  addMessage("user", question);
  const thinking = addMessage("assistant", "思考中…");
  $("askBtn").disabled = true;

  try {
    // 1. 无需检索的问题：直接回答，并明确告知未使用检索
    if (isDirectAnswerable(question)) {
      const system = "你是一个友好、简洁的 AI 助手。用户问的是问候或自我介绍类问题，不需要检索文档，请直接、简短地回答（1-3 句话）。";
      const answer = await askLLM(settings, system, question);
      thinking.textContent = answer;
      thinking.className = "msg assistant";
      const chip = el("span", "chip no-retrieval", "⚡ 未使用检索 · 直接回答");
      thinking.appendChild(el("div", "meta")).appendChild(chip);
      return;
    }

    // 2. 需要检索：检查索引
    if (!State.chunks.length) {
      thinking.textContent = "尚未建立索引。请先到「文档」页上传 PDF 并建立索引。";
      thinking.className = "msg error";
      return;
    }

    // 3. 检索（按对话页选择的分类过滤）
    const filter = chatCatFilter();
    const scopeName = $("chatCat").selectedOptions[0]?.text || "全部文档";
    const qTokens = tokenize(question);
    const bm25Hits = State.bm25 ? State.bm25.search(qTokens, settings.topK * 2, filter) : [];

    let vectorHits = [];
    if (State.vectors && settings.embedModel) {
      const [qvec] = await embedTexts([question], settings);
      vectorHits = vectorSearch(qvec, State, settings.topK * 2, filter);
    }

    let topChunks;
    if (vectorHits.length) {
      topChunks = rrfMerge(
        [bm25Hits.map((r) => ({ chunk: r.chunk })), vectorHits.map((r) => ({ chunk: r.chunk }))],
        settings.topK
      );
    } else {
      topChunks = bm25Hits.slice(0, settings.topK).map((r) => r.chunk);
    }

    if (!topChunks.length) {
      thinking.textContent = `在「${scopeName}」中未检索到相关内容，请尝试换一种问法或扩大检索范围。`;
      thinking.className = "msg assistant";
      return;
    }

    // 4. 组装上下文并生成
    const context = topChunks
      .map((c, i) => `[${i + 1}]（来源：${c.file} 第${c.page}页）\n${c.text}`)
      .join("\n\n---\n\n");
    const systemPrompt =
      "你是一个严谨的文档问答助手。请仅依据提供的参考片段回答用户问题。\n" +
      "要求：\n" +
      "1. 只使用参考片段中的信息，不要编造\n" +
      "2. 若参考片段信息不足，明确说明“根据现有文档无法回答”\n" +
      "3. 在回答末尾用 [1][2] 形式标注引用的片段编号\n" +
      "4. 使用与用户问题相同的语言回答";

    const answer = await askLLM(settings, systemPrompt, `参考片段：\n${context}\n\n用户问题：${question}`);

    thinking.textContent = answer;
    thinking.className = "msg assistant";

    // 5. 元信息：检索范围
    const meta = el("div", "meta");
    meta.appendChild(el("span", "chip", `🔍 检索范围：${scopeName}`));
    thinking.appendChild(meta);

    // 6. 引用片段
    const src = el("div", "src");
    const summary = el("summary", "", "查看引用片段");
    src.appendChild(summary);
    topChunks.forEach((c, i) => {
      src.appendChild(el("div", "", `[${i + 1}] ${c.file} 第${c.page}页：${c.text.slice(0, 120)}…`));
    });
    thinking.appendChild(src);
  } catch (e) {
    thinking.textContent = `请求失败：${e.message}`;
    thinking.className = "msg error";
  } finally {
    $("askBtn").disabled = false;
  }
}

/* ───────────────────────── 事件绑定 ───────────────────────── */
function bindEvents() {
  // 侧边栏导航
  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.addEventListener("click", () => switchView(btn.dataset.view));
  });

  // 设置
  $("parser").addEventListener("change", syncParserUI);
  $("saveSettings").addEventListener("click", () => {
    Settings.save(collectSettings());
    syncModelTag();
    syncRetrievalStat();
    addMessage("system", "设置已保存到本浏览器。");
  });
  $("clearData").addEventListener("click", () => {
    Store.clearAll();
    State.docs = [];
    State.chunks = [];
    State.bm25 = null;
    State.vectors = null;
    State.cats = [];
    loadSettingsIntoForm();
    renderAll();
    addMessage("system", "本地数据已全部清除。");
  });
  $("toggleKey").addEventListener("click", () => {
    const input = $("apiKey");
    input.type = input.type === "password" ? "text" : "password";
  });

  // 分类
  $("addCat").addEventListener("click", () => {
    const name = $("newCatName").value.trim();
    if (!name) return;
    if (currentCats().some((c) => c.name === name)) {
      addMessage("system", `分类「${name}」已存在。`);
      return;
    }
    State.cats.push({ id: uid("cat"), name });
    Store.saveCats(State.cats);
    $("newCatName").value = "";
    renderCatList();
    renderCatSelects();
  });

  // 上传
  const dz = $("dropZone");
  const fileInput = $("fileInput");
  dz.addEventListener("click", () => fileInput.click());
  dz.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });
  dz.addEventListener("dragover", (e) => {
    e.preventDefault();
    dz.classList.add("drag");
  });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("drag");
    handleFiles([...e.dataTransfer.files].filter((f) => f.type === "application/pdf" || f.name.endsWith(".pdf")));
  });
  fileInput.addEventListener("change", () => handleFiles([...fileInput.files]));
  fileInput.value = "";

  $("buildIndex").addEventListener("click", handleBuildIndex);

  // 对话
  $("chatForm").addEventListener("submit", (e) => {
    e.preventDefault();
    const q = $("question").value.trim();
    if (!q) return;
    $("question").value = "";
    handleAsk(q);
  });
  $("question").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      $("chatForm").requestSubmit();
    }
  });
}

/* ───────────────────────── 初始化 ───────────────────────── */
function init() {
  State.cats = Store.loadCats();
  State.docs = Store.loadDocs();
  loadSettingsIntoForm();
  renderAll();
  bindEvents();
  if (State.docs.length) {
    addMessage("system", `已从本地恢复 ${State.docs.length} 个文档，请建立索引后提问。`);
  }
}

init();



