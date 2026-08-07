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
    } catch (e) {
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
    } catch (e) {
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
    } catch (e) {
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
    localStorage.removeItem("pdfchat.convs.v1");
    localStorage.removeItem("pdfchat.memories.v1");
    localStorage.removeItem("pdfchat.activeconv.v1");
  },
};

/* ───────────────────────── 会话存储（第 2 层：跨刷新持久化） ───────────────────────── */
const ConvStore = {
  KEY: "pdfchat.convs.v1",
  ACTIVE_KEY: "pdfchat.activeconv.v1",
  load() {
    try {
      const v = JSON.parse(localStorage.getItem(this.KEY) || "[]");
      return Array.isArray(v) ? v : [];
    } catch (e) {
      return [];
    }
  },
  save(convs) {
    try {
      localStorage.setItem(this.KEY, JSON.stringify(convs));
    } catch (e) {
      console.warn("会话持久化失败（可能超出配额）", e);
    }
  },
  loadActive() {
    try {
      return localStorage.getItem(this.ACTIVE_KEY) || "";
    } catch (e) {
      return "";
    }
  },
  saveActive(id) {
    try {
      localStorage.setItem(this.ACTIVE_KEY, id || "");
    } catch (e) { /* ignore */ }
  },
};

/* ───────────────────────── 长期记忆存储（第 3 层） ───────────────────────── */
const MemStore = {
  KEY: "pdfchat.memories.v1",
  load() {
    try {
      const v = JSON.parse(localStorage.getItem(this.KEY) || "[]");
      return Array.isArray(v) ? v : [];
    } catch (e) {
      return [];
    }
  },
  save(mems) {
    try {
      localStorage.setItem(this.KEY, JSON.stringify(mems));
    } catch (e) {
      console.warn("长期记忆持久化失败", e);
    }
  },
  clear() {
    localStorage.removeItem(this.KEY);
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
  // 会话列表：[{ id, title, messages: [{role, content, ts}], createdAt, updatedAt }]
  convs: [],
  // 当前会话 ID
  activeConvId: "",
  // 长期记忆条目：[{ id, type, content, ts }]
  memories: [],
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
  const dim = (all[0] && all[0].length) || 0;
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
      if (settings.mineruMode === "official") {
        return await mineruOfficialParse(file, settings);
      }
      return await mineruSelfhostParse(file, settings);
    },
  },
};

/* MinerU 自托管：POST /file2text */
async function mineruSelfhostParse(file, settings) {
  const base = (settings.mineruUrl || "").replace(/\/+$/, "");
  if (!base) throw new Error("未配置 MinerU 服务地址（请选择「自托管服务」并填写地址）");
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
}

/* MinerU 官方 API：https://mineru.net —— 上传 → 轮询 → 获取结果 */
const MINERU_API_BASE = "https://mineru.net/api/v4";

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function mineruFetch(path, settings, options) {
  const isProxy = settings.callMode === "proxy";
  const url = isProxy ? `/proxy/mineru${path}` : `${MINERU_API_BASE}${path}`;
  const headers = { ...(options.headers || {}) };
  if (isProxy) {
    headers["X-API-Key"] = settings.mineruApiKey || "";
  } else {
    headers["Authorization"] = `Bearer ${settings.mineruApiKey || ""}`;
  }
  const resp = await fetch(url, { ...options, headers });
  if (!resp.ok) {
    const err = await resp.text().catch(() => "");
    throw new Error(`MinerU API ${resp.status}: ${err.slice(0, 200)}`);
  }
  return resp.json();
}

async function mineruOfficialParse(file, settings) {
  if (!settings.mineruApiKey) {
    throw new Error("使用 MinerU 官方 API 需要填写 API Key");
  }

  // 1. 创建解析任务（multipart 上传文件）
  const fd = new FormData();
  fd.append("file", file);
  fd.append("is_ocr", "true");
  fd.append("enable_formula", "true");
  fd.append("enable_table", "true");
  fd.append("language", "ch");
  const task = await mineruFetch("/extract/task", settings, { method: "POST", body: fd });
  if (!task.data || !task.data.task_id) {
    throw new Error(`MinerU 任务创建失败：${task.msg || "未知错误"}`);
  }
  const taskId = task.data.task_id;

  // 2. 轮询任务状态（最多 5 分钟）
  const deadline = Date.now() + 5 * 60 * 1000;
  while (Date.now() < deadline) {
    await sleep(2000);
    const status = await mineruFetch(`/extract/task/${taskId}`, settings, { method: "GET" });
    const st = status.data && status.data.state;
    if (st === "done") break;
    if (st === "failed") {
      throw new Error(`MinerU 解析失败：${(status.data && status.data.err_msg) || "未知错误"}`);
    }
    // waiting / running / pending 继续等待
  }

  // 3. 获取解析结果
  const result = await mineruFetch(`/extract/result/${taskId}`, settings, { method: "GET" });
  const extractResult = (result.data && result.data.extract_result) || [];
  const contents = (extractResult[0] && extractResult[0].extract_content) || [];
  const pages = contents
    .map((c) => ({
      page: (c.page_idx || 0) + 1,
      text: c.content || c.markdown || "",
    }))
    .filter((p) => p.text.trim());
  if (!pages.length) {
    throw new Error("MinerU 未返回可用的解析文本");
  }
  return pages;
}

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
    /^(你是谁|你是什么|介绍一下你自己|介绍下你自己|what are you|who are you)/i,
    /^(你能做什么|你能干什么|你会什么|可以做什么|what can you do)/i,
  ];
  return patterns.some((re) => re.test(t));
}

/* ───────────────────────── 对话（OpenAI 兼容） ───────────────────────── */
async function askLLM(settings, systemPrompt, userContent, history) {
  const messages = [{ role: "system", content: systemPrompt }];
  if (history && history.length) {
    messages.push(...history);
  }
  messages.push({ role: "user", content: userContent });

  const resp = await fetch(apiUrl("/chat/completions", settings), {
    method: "POST",
    headers: apiHeaders(settings),
    body: JSON.stringify(
      apiBody(
        {
          model: settings.model,
          messages: messages,
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
  const choice = data.choices && data.choices[0];
  const msg = choice && choice.message;
  return (msg && msg.content) || "";
}

/* ───────────────────────── 设置表单 ───────────────────────── */
function loadSettingsIntoForm() {
  const s = Settings.load();
  $("apiKey").value = s.apiKey || "";
  $("callMode").value = s.callMode || "direct";
  $("baseUrl").value = s.baseUrl || "https://api.openai.com/v1";
  $("model").value = s.model || "gpt-4o-mini";
  $("embedModel").value = s.embedModel || "";
  $("retrievalMode").value = s.retrievalMode || "keyword";
  $("topK").value = s.topK || 5;
  $("chunkSize").value = s.chunkSize || 800;
  $("parser").value = s.parser || "local";
  $("mineruMode").value = s.mineruMode || "official";
  $("mineruApiKey").value = s.mineruApiKey || "";
  $("mineruUrl").value = s.mineruUrl || "";
  $("enableMemory").checked = s.enableMemory !== false;
  $("memoryRounds").value = s.memoryRounds || 10;
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
    retrievalMode: $("retrievalMode").value,
    topK: parseInt($("topK").value, 10) || 5,
    chunkSize: parseInt($("chunkSize").value, 10) || 800,
    parser: $("parser").value,
    mineruMode: $("mineruMode").value,
    mineruApiKey: $("mineruApiKey").value.trim(),
    mineruUrl: $("mineruUrl").value.trim().replace(/\/+$/, ""),
    enableMemory: $("enableMemory").checked,
    memoryRounds: parseInt($("memoryRounds").value, 10) || 10,
  };
}

function syncParserUI() {
  const useMineru = $("parser").value === "mineru";
  $("mineruConfig").classList.toggle("hidden", !useMineru);
  if (useMineru) syncMineruModeUI();
  $("dzSub").textContent = useMineru
    ? (mineruModeSelected() === "official"
        ? "将由 MinerU 官方 API 解析 · PDF 会发送到 mineru.net"
        : "将由 MinerU 服务解析 · PDF 会发送到该服务")
    : "本地解析 · 不经过任何服务器";
}

function mineruModeSelected() {
  return $("mineruMode").value;
}

function syncMineruModeUI() {
  const official = mineruModeSelected() === "official";
  $("mineruOfficialConfig").classList.toggle("hidden", !official);
  $("mineruSelfConfig").classList.toggle("hidden", official);
}

function syncModelTag() {
  const tag = $("modelTag");
  if (!tag) return;
  const s = Settings.load();
  tag.textContent = s.model ? `模型 · ${s.model}` : "未配置";
}

function syncRetrievalStat() {
  const s = Settings.load();
  const modeMap = { keyword: "关键词检索", vector: "向量检索", hybrid: "混合检索（关键词 + 向量）" };
  const mode = modeMap[s.retrievalMode] || "关键词检索";
  const parts = [s.parser === "mineru" ? "MinerU 解析" : "本地解析", mode];
  if ((s.retrievalMode === "vector" || s.retrievalMode === "hybrid") && !s.embedModel) {
    parts.push("⚠️ 未配置 Embedding 模型，将自动降级为关键词检索");
  }
  $("retrievalStat").textContent = parts.join(" · ");
}

/* ───────────────────────── 视图切换（侧边栏） ───────────────────────── */
function switchView(name) {
  const target = document.getElementById(`view-${name}`);
  if (!target) {
    console.warn(`视图不存在: view-${name}`);
    return;
  }
  document.querySelectorAll(".nav-item").forEach((b) => {
    b.classList.toggle("active", b.dataset.view === name);
  });
  document.querySelectorAll(".view").forEach((v) => {
    v.classList.toggle("active", v === target);
  });
}

// 导航事件委托放在顶层，即使后续初始化失败也能保证切换可用
document.addEventListener("click", (e) => {
  const btn = e.target.closest(".nav-item");
  if (btn && btn.dataset.view) switchView(btn.dataset.view);
});

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
    li.appendChild(el("span", "", doc.name || "未命名文档"));
    li.appendChild(el("span", "pages", fmtPages(doc.pages ? doc.pages.length : 0)));

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
  $("docStats").textContent = `共 ${State.docs.length} 个文档 / ${State.docs.reduce((s, d) => s + (d.pages ? d.pages.length : 0), 0)} 页`;
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
      }      const doc = {
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

/* ───────────────────────── 会话管理（第 2 层） ───────────────────────── */
function currentConv() {
  return State.convs.find((c) => c.id === State.activeConvId) || null;
}

function saveConversation(conv) {
  if (!conv) return;
  conv.updatedAt = Date.now();
  ConvStore.save(State.convs);
}

function newConversation() {
  const conv = {
    id: uid("conv"),
    title: "新对话",
    messages: [],
    createdAt: Date.now(),
    updatedAt: Date.now(),
  };
  State.convs.unshift(conv);
  State.activeConvId = conv.id;
  ConvStore.save(State.convs);
  ConvStore.saveActive(conv.id);
  renderConvList();
  renderMessagesFromConv(conv);
  return conv;
}

/* 清空当前会话（保留会话，清空消息） */
function clearCurrentConversation() {
  const conv = currentConv();
  if (!conv) return;
  conv.messages = [];
  conv.title = "新对话";
  conv.renamed = false;
  conv.updatedAt = Date.now();
  ConvStore.save(State.convs);
  renderConvList();
  renderMessagesFromConv(conv);
}

function switchConversation(id) {
  const conv = State.convs.find((c) => c.id === id);
  if (!conv) return;
  State.activeConvId = id;
  ConvStore.saveActive(id);
  renderConvList();
  renderMessagesFromConv(conv);
}

function deleteConversation(id) {
  State.convs = State.convs.filter((c) => c.id !== id);
  if (State.activeConvId === id) {
    State.activeConvId = "";
    if (State.convs.length) {
      State.activeConvId = State.convs[0].id;
      ConvStore.saveActive(State.activeConvId);
      renderMessagesFromConv(State.convs[0]);
    } else {
      ConvStore.saveActive("");
      $("messages").innerHTML = "";
      addMessage("system", "已删除会话。点击「新建会话」开始新的对话。");
    }
  }
  ConvStore.save(State.convs);
  renderConvList();
}

function renderConvList() {
  const list = $("convList");
  list.innerHTML = "";
  $("convCount").textContent = String(State.convs.length);
  if (!State.convs.length) {
    const empty = el("li", "conv-empty", "暂无会话，点击「＋ 新建」开始");
    list.appendChild(empty);
    renderActiveConvInfo();
    return;
  }
  for (const conv of State.convs) {
    const li = el("li", "conv-item" + (conv.id === State.activeConvId ? " active" : ""));
    const label = el("div", "conv-label");
    const title = el("div", "conv-title", conv.title || "新对话");
    label.appendChild(title);
    const subRow = el("div", "conv-sub");
    const msgCount = conv.messages ? conv.messages.length : 0;
    subRow.appendChild(el("span", "conv-sub-time", relTime(conv.updatedAt)));
    if (msgCount > 0) {
      subRow.appendChild(el("span", "conv-sub-count", `${msgCount} 条`));
    }
    label.appendChild(subRow);
    li.appendChild(label);

    const rename = el("button", "conv-rename", "✏️");
    rename.title = "重命名会话";
    rename.addEventListener("click", (e) => {
      e.stopPropagation();
      renameConversation(conv.id);
    });
    li.appendChild(rename);

    const del = el("button", "conv-del", "🗑");
    del.title = "删除会话";
    del.addEventListener("click", (e) => {
      e.stopPropagation();
      deleteConversation(conv.id);
    });
    li.appendChild(del);
    li.addEventListener("click", () => switchConversation(conv.id));
    list.appendChild(li);
  }
  renderActiveConvInfo();
}

/* 重命名会话：使用浏览器原生 prompt（兼容简单、无需额外 UI） */
function renameConversation(id) {
  const conv = State.convs.find((c) => c.id === id);
  if (!conv) return;
  const newName = prompt("请输入新的会话名称：", conv.title || "");
  if (newName === null) return; // 用户取消
  const trimmed = newName.trim();
  if (!trimmed) {
    addMessage("system", "会话名称不能为空。");
    return;
  }
  conv.title = trimmed.slice(0, 50); // 限制长度
  conv.renamed = true; // 标记用户已手动命名，避免自动标题覆盖
  conv.updatedAt = Date.now();
  ConvStore.save(State.convs);
  renderConvList();
  addMessage("system", `会话已重命名为「${conv.title}」。`);
}

/* 相对时间显示：刚刚 / N 分钟前 / N 小时前 / 昨天 / 日期 */
function relTime(ts) {
  if (!ts) return "";
  const diff = Date.now() - ts;
  const min = 60 * 1000;
  const hour = 60 * min;
  const day = 24 * hour;
  if (diff < min) return "刚刚";
  if (diff < hour) return `${Math.floor(diff / min)} 分钟前`;
  if (diff < day) return `${Math.floor(diff / hour)} 小时前`;
  if (diff < 2 * day) return "昨天";
  const d = new Date(ts);
  return `${d.getMonth() + 1}/${d.getDate()}`;
}

/* 对话页头部显示当前会话名 */
function renderActiveConvInfo() {
  const info = $("activeConvInfo");
  if (!info) return;
  const conv = currentConv();
  info.textContent = conv
    ? `当前会话：${conv.title || "新对话"}${conv.messages ? ` · ${conv.messages.length} 条消息` : ""}`
    : "选择检索范围，基于文档提问";
}

function renderMessagesFromConv(conv) {
  const box = $("messages");
  box.innerHTML = "";
  if (!conv || !conv.messages.length) {
    addMessage("system", "👋 上传 PDF 并建立索引后，即可基于文档提问。无需检索的问题（如问候）会直接回答。");
    return;
  }
  for (const m of conv.messages) {
    const msg = el("div", `msg ${m.role}`);
    msg.textContent = m.content;
    if (m.meta) {
      const meta = el("div", "meta");
      if (typeof m.meta === "string") meta.textContent = m.meta;
      else meta.append(...m.meta);
      msg.appendChild(meta);
    }
    box.appendChild(msg);
  }
  box.scrollTop = box.scrollHeight;
}

function pushConvMessage(role, content, meta) {
  let conv = currentConv();
  if (!conv) {
    conv = newConversation();
  }
  conv.messages.push({ role, content, meta: meta || null, ts: Date.now() });
  // 控制单会话消息上限（避免 localStorage 过大）：最多保留 60 条
  if (conv.messages.length > 60) {
    conv.messages = conv.messages.slice(-60);
  }
  // 自动更新标题：仅当用户未手动重命名且标题仍是默认值时
  const firstUser = conv.messages.find((m) => m.role === "user");
  if (firstUser && !conv.renamed && (conv.title === "新对话" || !conv.title)) {
    conv.title = firstUser.content.slice(0, 24) || "新对话";
  }
  saveConversation(conv);
  renderConvList();
}

/* ───────────────────────── 会话区折叠（状态持久化） ───────────────────────── */
const CONV_PANEL_KEY = "pdfchat.convpanel.collapsed";

function toggleConvPanel() {
  const panel = $("sideConv");
  const collapsed = panel.classList.toggle("collapsed");
  try {
    localStorage.setItem(CONV_PANEL_KEY, collapsed ? "1" : "0");
  } catch (e) { /* ignore */ }
  if (!collapsed && panel.scrollIntoView) {
    // 展开时滚动侧边栏到会话区，方便用户看到列表
    try { $("convHeader").scrollIntoView({ block: "nearest" }); } catch (e) { /* ignore */ }
  }
}

function restoreConvPanel() {
  const panel = $("sideConv");
  if (!panel) return;
  try {
    if (localStorage.getItem(CONV_PANEL_KEY) === "1") {
      panel.classList.add("collapsed");
    }
  } catch (e) { /* ignore */ }
}

/* ───────────────────────── 第 1 层：短期记忆（最近 N 轮） ───────────────────────── */
function getHistoryForPrompt(settings) {
  if (!settings.enableMemory) return [];
  const conv = currentConv();
  if (!conv) return [];
  const rounds = settings.memoryRounds || 10;
  // 取最近 rounds 轮（一问一答 = 2 条），但排除最后一条 assistant（通常未完成）
  const msgs = conv.messages.slice(-(rounds * 2));
  const history = [];
  for (const m of msgs) {
    if (m.role === "user" || m.role === "assistant") {
      history.push({ role: m.role, content: m.content });
    }
  }
  return history;
}

/* ───────────────────────── 第 4 层：检索式记忆 ───────────────────────── */
function memorySearch(query, settings) {
  if (!settings.enableMemory) return [];
  const qTokens = tokenize(query);
  const qSet = new Set(qTokens);
  const cands = [];

  // 候选 1：当前会话历史（去掉最近一条 user，避免重复）
  const conv = currentConv();
  if (conv && conv.messages.length) {
    const historyMsgs = conv.messages.slice(0, -1);
    for (const m of historyMsgs) {
      if (m.role === "user" || m.role === "assistant") {
        cands.push({ text: `${m.role === "user" ? "用户" : "助手"}: ${m.content}`, kind: "history" });
      }
    }
  }

  // 候选 2：长期记忆
  for (const mem of State.memories) {
    cands.push({ text: `[记忆] ${mem.content}`, kind: "memory" });
  }

  // 关键词重叠打分
  const scored = cands
    .map((c) => {
      const cTokens = tokenize(c.text);
      const cSet = new Set(cTokens);
      let hit = 0;
      for (const t of qSet) if (cSet.has(t)) hit++;
      return { c, score: hit / Math.max(1, qSet.size) };
    })
    .filter((r) => r.score >= 0.25 && r.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 3);

  return scored.map((r) => r.c.text);
}

/* ───────────────────────── 第 3 层：长期记忆提取 ───────────────────────── */
async function extractMemories(question, answer, settings) {
  if (!settings.enableMemory || !settings.apiKey) return;
  try {
    const system =
      "你是记忆提取助手。从用户问题与助手回答中，提取值得长期记住的信息，例如：\n" +
      "- 用户偏好（语言、风格、内容偏好）\n" +
      "- 关键事实（用户身份、项目背景、重要结论）\n" +
      "- 上下文（用户正在做的事、目标）\n" +
      "请以 JSON 数组输出，每项形如 {\"type\": \"preference|fact|context\", \"content\": \"一句话描述\"}。\n" +
      "如果没有值得记住的内容，输出 []。不要输出其他内容。";
    const resp = await fetch(apiUrl("/chat/completions", settings), {
      method: "POST",
      headers: apiHeaders(settings),
      body: JSON.stringify(
        apiBody(
          {
            model: settings.model,
            messages: [
              { role: "system", content: system },
              { role: "user", content: `用户问题：${question}\n\n助手回答：${answer}` },
            ],
            temperature: 0.1,
            max_tokens: 300,
          },
          settings
        )
      ),
    });
    if (!resp.ok) return;
    const data = await resp.json();
    const text = (data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content) || "";
    let items = [];
    try {
      items = JSON.parse(text);
    } catch (e) {
      const m = text.match(/\[[\s\S]*\]/);
      if (m) items = JSON.parse(m[0]);
    }
    if (!Array.isArray(items)) return;

    let added = 0;
    for (const it of items) {
      const content = String(it.content || "").trim();
      if (!content) continue;
      // 去重：内容完全相同或高度相似则跳过
      const dup = State.memories.some((m) => m.content === content);
      if (dup) continue;
      State.memories.push({
        id: uid("mem"),
        type: it.type || "fact",
        content,
        ts: Date.now(),
      });
      added++;
    }
    // 上限 50 条
    if (State.memories.length > 50) {
      State.memories = State.memories.slice(-50);
    }
    if (added > 0) {
      MemStore.save(State.memories);
      renderMemories();
    }
  } catch (e) {
    // 记忆提取失败不影响主流程
    console.warn("记忆提取失败（已忽略）:", e);
  }
}

function renderMemories() {
  const list = $("memList");
  list.innerHTML = "";
  $("memCount").textContent = String(State.memories.length);
  $("memStat").textContent = `${State.memories.length} 条`;
  if (!State.memories.length) {
    const empty = el("li", "mem-empty", "暂无长期记忆");
    list.appendChild(empty);
    return;
  }
  for (const mem of State.memories.slice(-8).reverse()) {
    const li = el("li", "mem-item");
    li.textContent = mem.content;
    list.appendChild(li);
  }
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

    const wantVector = settings.retrievalMode === "vector" || settings.retrievalMode === "hybrid";
    if (wantVector) {
      if (!settings.embedModel || !settings.apiKey) {
        addMessage("system", "检索模式为向量/混合，但未配置 Embedding 模型或 API Key，本次使用关键词检索。");
      } else {
        addMessage("system", `正在生成向量索引（${State.chunks.length} 块）…`);
        State.vectors = await buildVectors(State.chunks, settings);
      }
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

  // 确保存在当前会话（第 2 层）
  let conv = currentConv();
  if (!conv) conv = newConversation();

  addMessage("user", question);
  pushConvMessage("user", question);
  const thinking = addMessage("assistant", "思考中…");
  $("askBtn").disabled = true;

  try {
    // 第 1 层：短期记忆（最近 N 轮）
    const history = getHistoryForPrompt(settings);

    // 第 3 层：长期记忆注入 system
    let memoryBlock = "";
    if (settings.enableMemory && State.memories.length) {
      const lines = State.memories.map((m, i) => `${i + 1}. ${m.content}`).join("\n");
      memoryBlock = `\n\n【关于用户的长期记忆】\n${lines}\n（仅作参考，若与文档信息冲突以文档为准）`;
    }

    // 1. 无需检索的问题：直接回答，并明确告知未使用检索
    if (isDirectAnswerable(question)) {
      const system =
        "你是一个友好、简洁的 AI 助手。用户问的是问候或自我介绍类问题，不需要检索文档，请直接、简短地回答（1-3 句话）。" +
        memoryBlock;
      const answer = await askLLM(settings, system, question, history);
      thinking.textContent = answer;
      thinking.className = "msg assistant";
      const chip = el("span", "chip no-retrieval", "⚡ 未使用检索 · 直接回答");
      thinking.appendChild(el("div", "meta")).appendChild(chip);
      pushConvMessage("assistant", answer, [chip.textContent]);
      renderMemories();
      extractMemories(question, answer, settings); // 第 3 层：后台提取记忆
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
    const scopeSel = $("chatCat");
    const scopeName = (scopeSel.selectedOptions && scopeSel.selectedOptions[0] && scopeSel.selectedOptions[0].text) || "全部文档";
    const qTokens = tokenize(question);
    const bm25Hits = State.bm25 ? State.bm25.search(qTokens, settings.topK * 2, filter) : [];

    const wantVector = settings.retrievalMode === "vector" || settings.retrievalMode === "hybrid";
    let vectorHits = [];
    if (wantVector && State.vectors && settings.embedModel) {
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
      pushConvMessage("assistant", thinking.textContent);
      return;
    }

    // 4. 组装上下文并生成
    const context = topChunks
      .map((c, i) => `[${i + 1}]（来源：${c.file} 第${c.page}页）\n${c.text}`)
      .join("\n\n---\n\n");

    // 第 4 层：检索式记忆（从历史与长期记忆中检索相关片段）
    const memHits = memorySearch(question, settings);
    let memContext = "";
    if (memHits.length) {
      memContext = "\n\n【相关历史对话】\n" + memHits.map((t) => "- " + t).join("\n");
    }

    const systemPrompt =
      "你是一个严谨的文档问答助手。请仅依据提供的参考片段回答用户问题。\n" +
      "要求：\n" +
      "1. 只使用参考片段中的信息，不要编造\n" +
      "2. 若参考片段信息不足，明确说明“根据现有文档无法回答”\n" +
      "3. 在回答末尾用 [1][2] 形式标注引用的片段编号\n" +
      "4. 使用与用户问题相同的语言回答" +
      memoryBlock;

    const userContent =
      `参考片段：\n${context}${memContext}\n\n用户问题：${question}`;

    const answer = await askLLM(settings, systemPrompt, userContent, history);

    thinking.textContent = answer;
    thinking.className = "msg assistant";

    // 5. 元信息：检索范围 + 记忆标签
    const meta = el("div", "meta");
    meta.appendChild(el("span", "chip", `🔍 检索范围：${scopeName}`));
    if (memHits.length) meta.appendChild(el("span", "chip", `🧠 相关历史：${memHits.length} 条`));
    thinking.appendChild(meta);

    // 6. 引用片段
    const src = el("div", "src");
    const summary = el("summary", "", "查看引用片段");
    src.appendChild(summary);
    topChunks.forEach((c, i) => {
      src.appendChild(el("div", "", `[${i + 1}] ${c.file} 第${c.page}页：${c.text.slice(0, 120)}…`));
    });
    thinking.appendChild(src);

    // 保存回答到会话（第 2 层）
    pushConvMessage("assistant", answer, [meta.textContent || "assistant"]);
    renderMemories();
    extractMemories(question, answer, settings); // 第 3 层：后台提取记忆
  } catch (e) {
    thinking.textContent = `请求失败：${e.message}`;
    thinking.className = "msg error";
  } finally {
    $("askBtn").disabled = false;
  }
}

/* ───────────────────────── 事件绑定 ───────────────────────── */
function bindEvents() {
  // 设置
  $("parser").addEventListener("change", syncParserUI);
  $("mineruMode").addEventListener("change", syncMineruModeUI);
  $("retrievalMode").addEventListener("change", () => {
    syncRetrievalStat();
    addMessage("system", "检索模式已切换，请重新建立索引后生效。");
  });
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
    State.convs = [];
    State.activeConvId = "";
    State.memories = [];
    loadSettingsIntoForm();
    renderAll();
    renderConvList();
    renderMemories();
    $("messages").innerHTML = "";
    addMessage("system", "本地数据已全部清除。");
  });
  $("clearMemories").addEventListener("click", () => {
    State.memories = [];
    MemStore.clear();
    renderMemories();
    addMessage("system", "长期记忆已清除。");
  });
  $("toggleKey").addEventListener("click", () => {
    const input = $("apiKey");
    input.type = input.type === "password" ? "text" : "password";
  });

  // 会话（侧边栏可折叠区）
  $("newConv").addEventListener("click", (e) => {
    e.stopPropagation(); // 防止冒泡触发折叠切换
    newConversation();
  });
  $("convToggle").addEventListener("click", (e) => {
    e.stopPropagation();
    toggleConvPanel();
  });
  $("convHeader").addEventListener("click", () => {
    toggleConvPanel();
  });

  // 对话头部操作
  $("newConvTop").addEventListener("click", () => {
    newConversation();
  });
  $("clearConv").addEventListener("click", () => {
    const conv = currentConv();
    if (!conv) {
      addMessage("system", "当前没有会话。");
      return;
    }
    if (conv.messages && conv.messages.length) {
      clearCurrentConversation();
      addMessage("system", "当前会话已清空。");
    } else {
      addMessage("system", "当前会话已经是空的。");
    }
  });

  // 输入框自适应高度（随内容增高，最多 6 行）
  const questionBox = $("question");
  function autoResize() {
    questionBox.style.height = "auto";
    questionBox.style.height = Math.min(questionBox.scrollHeight, 132) + "px";
  }
  questionBox.addEventListener("input", autoResize);
  questionBox.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      $("chatForm").requestSubmit();
    }
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
    autoResize();
    handleAsk(q);
  });
}

/* ───────────────────────── 初始化 ───────────────────────── */
function init() {
  try {
    State.cats = Store.loadCats();
    State.docs = Store.loadDocs();
    State.convs = ConvStore.load();
    State.memories = MemStore.load();
    const savedActive = ConvStore.loadActive();
    State.activeConvId = savedActive && State.convs.some((c) => c.id === savedActive)
      ? savedActive
      : (State.convs[0] ? State.convs[0].id : "");
    loadSettingsIntoForm();
    renderAll();
    restoreConvPanel();
    renderConvList();
    renderMemories();
    bindEvents();
    const conv = currentConv();
    if (conv) {
      renderMessagesFromConv(conv);
    } else if (State.docs.length) {
      addMessage("system", `已从本地恢复 ${State.docs.length} 个文档，请建立索引后提问。`);
    }
    renderActiveConvInfo();
  } catch (e) {
    console.error("初始化失败（已降级继续运行）:", e);
    // 尽力保证导航可用（事件委托已在顶层注册），并重新绑定基础事件
    try { bindEvents(); } catch (_) {}
  }
}

init();



