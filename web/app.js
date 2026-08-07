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

import { initializeIndustrialWorkspace } from "./js/bootstrap.js";
import { detectMediaReferences, resolveMediaReference, sha256, stableId } from "./js/media-association.js";
import { readOpenAIStream } from "./js/streaming.js";

initializeIndustrialWorkspace();

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
  SECRET_FIELDS: ["apiKey", "vlmApiKey", "mineruApiKey", "webTavilyKey"],
  load() {
    try {
      const settings = JSON.parse(localStorage.getItem(this.KEY) || "{}");
      for (const field of this.SECRET_FIELDS) {
        settings[field] = sessionStorage.getItem(`mmrag.secret.${field}`) || settings[field] || "";
      }
      return settings;
    } catch (e) {
      return {};
    }
  },
  save(s) {
    const safe = { ...s };
    for (const field of this.SECRET_FIELDS) {
      const value = safe[field] || "";
      delete safe[field];
      if (value) sessionStorage.setItem(`mmrag.secret.${field}`, value);
      else sessionStorage.removeItem(`mmrag.secret.${field}`);
    }
    localStorage.setItem(this.KEY, JSON.stringify(safe));
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
  // 文档列表：[{ id, name, cat, pages: [{page, text}], media: [{id,type,label,page,dataUrl,caption}], parser, addedAt }]
  docs: [],
  // 分块结果：[{ id, docId, file, page, text, tokens, refs }]
  chunks: [],
  // BM25 数据结构
  bm25: null,
  // 可选向量数据：{ vectors: Float32Array, norm: Float32Array, dim: number }
  vectors: null,
  // 媒体资产（图片/表格，多模态检索用）：[{ id, docId, file, page, label, type, num, dataUrl, caption }]
  media: [],
  // 会话列表：[{ id, title, messages: [{role, content, ts}], createdAt, updatedAt }]
  convs: [],
  // 当前会话 ID
  activeConvId: "",
  // 长期记忆条目：[{ id, type, content, ts }]
  memories: [],
  generationController: null,
};

/* ───────────────────────── 图/表引用检测（RAG-Anything 风格） ───────────────────────── */
// 匹配 "图1" / "图 1" / "Figure 1" / "Fig. 1" / "表2" / "表 2" / "Table 2"
function detectMediaRefsInText(text) {
  const refs = [];
  const re = /(?:图\s*|Figure\s+|Fig\.?\s*)(\d{1,3})|(?:表\s*|表格\s*|Table\s+)(\d{1,3})/gi;
  let m;
  while ((m = re.exec(text))) {
    let type, num;
    if (m[1] !== undefined) {
      type = "image";
      num = m[1];
    } else {
      type = "table";
      num = m[2];
    }
    refs.push({
      label: type === "image" ? `图${num}` : `表${num}`,
      type,
      num,
      offset: m.index,
    });
  }
  return refs;
}

// 渲染 PDF 页面为图片（用于展示/交给 VLM 理解）
async function renderPageToDataUrl(page, scale) {
  const viewport = page.getViewport({ scale: scale || 1.5 });
  const canvas = document.createElement("canvas");
  canvas.width = viewport.width;
  canvas.height = viewport.height;
  const ctx = canvas.getContext("2d");
  await page.render({ canvasContext: ctx, viewport }).promise;
  return canvas.toDataURL("image/jpeg", 0.7);
}

// 从检索命中的分块收集关联媒体（按 文档+类型+编号 匹配）
function collectMediaForChunks(chunks, mediaList) {
  const out = [];
  const seen = new Set();
  for (const c of chunks || []) {
    for (const ref of c.refs || []) {
      const key = `${c.docId}_${ref.type}_${ref.num}_${c.page}`;
      if (seen.has(key)) continue;
      seen.add(key);
      // num 可能为字符串（正则捕获）或数字（自增计数），统一 String 比较
      let found = mediaList.find(
        (m) => m.docId === c.docId && m.type === ref.type && String(m.num) === String(ref.num) && m.page === c.page
      );
      if (!found) {
        found = mediaList.find(
          (m) => m.docId === c.docId && m.type === ref.type && String(m.num) === String(ref.num)
        );
      }
      if (found) out.push({ ...found, ref });
    }
  }
  return out;
}

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
async function embedTexts(texts, settings, options = {}) {
  const body = { model: settings.embedModel, input: texts };
  const resp = await fetch(apiUrl("/embeddings", settings), {
    method: "POST",
    headers: apiHeaders(settings),
    body: JSON.stringify(apiBody(body, settings)),
    signal: options.signal,
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
      const media = [];
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

        // 渲染页面图像（供引用展示 / VLM 理解）
        let dataUrl = "";
        try {
          dataUrl = await renderPageToDataUrl(page, 1.5);
        } catch (e) {
          /* 渲染失败则跳过页面图像 */
        }
        // pdf.js 无法可靠给出独立图表边界：整页仅作为 fallback 快照，绝不冒充“图 N”。
        if (dataUrl) media.push({
          id: `page_snapshot_${p}`,
          file: file.name,
          page: p,
          label: `第${p}页`,
          type: "page_snapshot",
          dataUrl,
          caption: text.slice(0, 400),
          searchText: text.slice(0, 1200),
          quality: "fallback",
          extractionMethod: "pdfjs_page_render",
        });
      }
      return { pages, media };
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

/* MinerU 解析：从 markdown 文本中抽取图片与表格媒体资产（多模态检索用） */
function parseMediaFromMarkdown(markdown, page) {
  const media = [];
  let imgSeq = 0;
  let tableSeq = 0;

  // 图片：![label](url) —— url 可为 base64(data:image/...) 或 http(s) 链接
  const imgRe = /!\[([^\]]*)\]\(([^)]+)\)/g;
  let m;
  while ((m = imgRe.exec(markdown || ""))) {
    const labelRaw = (m[1] || "").trim();
    const url = (m[2] || "").trim();
    let type = "image";
    let num;
    let label;
    const numMatch = labelRaw.match(/^(?:图|Figure|Fig\.?)\s*(\d{1,3})$/i);
    if (numMatch) {
      num = numMatch[1];
      label = `图${num}`;
    } else {
      imgSeq++;
      num = imgSeq;
      label = `图${imgSeq}`;
    }
    let dataUrl = "";
    let urlOnly = "";
    if (/^data:image\//i.test(url)) dataUrl = url;
    else if (/^https?:\/\//i.test(url)) urlOnly = url;
    else continue; // 忽略相对路径等无法直接引用的图片
    media.push({
      id: `image_${num}_p${page}`,
      page,
      label,
      type,
      num,
      dataUrl,
      url: urlOnly,
      caption: labelRaw,
    });
  }

  // 表格：连续以 | 开头的 markdown 表格行
  const lines = (markdown || "").split("\n");
  let i = 0;
  while (i < lines.length) {
    if (lines[i].trim().startsWith("|") && lines[i].includes("|")) {
      const rows = [];
      while (i < lines.length && lines[i].trim().startsWith("|") && lines[i].includes("|")) {
        rows.push(lines[i].trim());
        i++;
      }
      if (rows.length >= 2) {
        tableSeq++;
        media.push({
          id: `table_${tableSeq}_p${page}`,
          page,
          label: `表${tableSeq}`,
          type: "table",
          num: tableSeq,
          dataUrl: "",
          url: "",
          caption: rows.join("\n"),
        });
      }
    } else {
      i++;
    }
  }
  return media;
}

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
  // 返回 { pages, media }，media 从 markdown 中抽取图片/表格
  if (Array.isArray(data.pages) && data.pages.length) {
    const pages = data.pages.map((p) => ({
      page: p.page || 1,
      text: p.text || p.content || "",
    })).filter((p) => p.text.trim());
    const media = pages.flatMap((p) => parseMediaFromMarkdown(p.text, p.page));
    return { pages, media };
  }
  if (typeof data.text === "string" && data.text.trim()) {
    const pages = [{ page: 1, text: data.text }];
    return { pages, media: parseMediaFromMarkdown(data.text, 1) };
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

  // 逐项处理：文本/markdown 累积，图片项转成 ![label](url) 语法，统一交给 parseMediaFromMarkdown
  const pages = [];
  const markdownParts = [];
  let imgSeq = 0;
  for (const c of contents) {
    const pageIdx = (c.page_idx || 0) + 1;
    const text = c.content || c.markdown || "";
    const imgUrl = c.img_url || c.image_url || "";
    if (imgUrl && (/^data:image\//i.test(imgUrl) || /^https?:\/\//i.test(imgUrl))) {
      imgSeq++;
      markdownParts.push({ pageIdx, text: `${text ? text + "\n" : ""}![图${imgSeq}](${imgUrl})` });
    } else if (text && text.trim()) {
      markdownParts.push({ pageIdx, text });
    }
  }
  const media = [];
  for (const { pageIdx, text } of markdownParts) {
    media.push(...parseMediaFromMarkdown(text, pageIdx));
    pages.push({ page: pageIdx, text });
  }
  if (!pages.length) {
    throw new Error("MinerU 未返回可用的解析文本");
  }
  return { pages, media };
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
        // 记录本分块内对「图N / 表N」的引用位置（RAG-Anything 风格）
        const refs = detectMediaRefsInText(text).map((r) => ({
          ...r,
          page: p.page,
          docId: doc.id,
        }));
        chunks.push({
          id: `c${seq++}`,
          docId: doc.id,
          file: doc.name,
          cat: doc.cat || "",
          page: p.page,
          text,
          tokens: tokenize(text),
          refs,
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
async function askLLM(settings, systemPrompt, userContent, history, options = {}) {
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
          stream: true,
        },
        settings
      )
    ),
    signal: options.signal,
  });
  if (!resp.ok) {
    const err = await resp.text().catch(() => "");
    throw new Error(`LLM API ${resp.status}: ${err.slice(0, 300)}`);
  }
  return readOpenAIStream(resp, options.onDelta);
}

function setGenerationUI(active) {
  $("askBtn").disabled = active;
  $("stopBtn").classList.toggle("hidden", !active);
  $("stopBtn").disabled = !active;
  $("streamStatus").classList.toggle("active", active);
  $("streamStatusText").textContent = active ? "正在生成" : "流式响应已就绪";
  $("liveRegion").textContent = active ? "正在生成回答" : "回答生成结束";
}

function streamIntoMessage(message) {
  let answer = "";
  let frame = 0;
  const paint = () => {
    frame = 0;
    message.textContent = answer || "正在连接模型…";
    const box = $("messages");
    if (box.scrollHeight - box.scrollTop - box.clientHeight < 120) box.scrollTop = box.scrollHeight;
  };
  return {
    append(delta) {
      answer += delta;
      if (!frame) frame = requestAnimationFrame(paint);
    },
    flush() {
      if (frame) cancelAnimationFrame(frame);
      paint();
      return answer;
    },
  };
}

/* ───────────────────────── VLM 多模态理解（图片/表格） ───────────────────────── */
async function askVLM(settings, imageMedia, options = {}) {
  // OpenAI 兼容多模态输入：content 为 text + image_url 数组
  const content = [
    {
      type: "text",
      text:
        "以下是文档中与用户问题相关的图片（可能包含整页截图）。请用中文逐张简要描述核心内容（每张 1-3 句），" +
        "说明图片里有哪些关键信息（文字、数据、图表、表格等），以及它们可能用于回答什么问题。",
    },
    ...imageMedia.map((m) => ({
      type: "image_url",
      image_url: { url: m.dataUrl || m.url },
    })),
  ];
  const resp = await fetch(apiUrl("/chat/completions", settings), {
    method: "POST",
    headers: apiHeaders(settings),
    body: JSON.stringify(
      apiBody(
        {
          model: settings.vlmModel || settings.model,
          messages: [{ role: "user", content }],
          temperature: 0.2,
          max_tokens: 800,
        },
        settings
      )
    ),
    signal: options.signal,
  });
  if (!resp.ok) {
    const err = await resp.text().catch(() => "");
    throw new Error(`VLM API ${resp.status}: ${err.slice(0, 300)}`);
  }
  const data = await resp.json();
  const choice = data.choices && data.choices[0];
  const msg = choice && choice.message;
  return (msg && msg.content) || "";
}

/* ───────────────────────── VLM 未配置弹窗提醒 ───────────────────────── */
let vlmModalShown = false;
function showVlmModal(msg) {
  if (vlmModalShown) return;
  vlmModalShown = true;
  const box = $("vlmModal");
  if (!box) return;
  if (msg) $("vlmModalMsg").textContent = msg;
  box.classList.remove("hidden");
}
function closeVlmModal() {
  const box = $("vlmModal");
  if (box) box.classList.add("hidden");
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
  // VLM 配置（多模态检索）
  $("vlmApiKey").value = s.vlmApiKey || "";
  $("vlmBaseUrl").value = s.vlmBaseUrl || "";
  $("vlmModel").value = s.vlmModel || "";
  // Web 搜索配置（知识库 Web 抓取）
  $("webProvider").value = s.webProvider || "duckduckgo";
  $("webTavilyKey").value = s.webTavilyKey || "";
  syncWebProviderUI();
  $("enableMemory").checked = s.enableMemory !== false;
  $("memoryRounds").value = s.memoryRounds || 10;
  syncParserUI();
  syncModelTag();
  syncRetrievalStat();
  syncVlmStat();
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
    vlmApiKey: $("vlmApiKey").value.trim(),
    vlmBaseUrl: $("vlmBaseUrl").value.trim().replace(/\/+$/, ""),
    vlmModel: $("vlmModel").value.trim(),
    webProvider: $("webProvider").value,
    webTavilyKey: $("webTavilyKey").value.trim(),
    enableMemory: $("enableMemory").checked,
    memoryRounds: parseInt($("memoryRounds").value, 10) || 10,
  };
}

function syncWebProviderUI() {
  const useTavily = $("webProvider").value === "tavily";
  $("webTavilyConfig").classList.toggle("hidden", !useTavily);
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
  const modeMap = {
    keyword: "关键词检索",
    vector: "向量检索",
    hybrid: "混合检索（关键词 + 向量）",
    multimodal: "多模态检索（文本 + 图片/表格引用 + VLM）",
  };
  const mode = modeMap[s.retrievalMode] || "关键词检索";
  const parts = [s.parser === "mineru" ? "MinerU 解析" : "本地解析", mode];
  if ((s.retrievalMode === "vector" || s.retrievalMode === "hybrid") && !s.embedModel) {
    parts.push("警告：未配置 Embedding 模型，将自动降级为关键词检索");
  }
  if (s.retrievalMode === "multimodal" && !(s.vlmApiKey && s.vlmModel)) {
    parts.push("警告：未配置 VLM，图片将仅展示引用");
  }
  $("retrievalStat").textContent = parts.join(" · ");
}

function syncVlmStat() {
  const s = Settings.load();
  const stat = $("vlmStat");
  if (!stat) return;
  const ok = s.vlmApiKey && s.vlmModel;
  stat.textContent = ok ? `已配置 · ${s.vlmModel}` : "未配置（多模态图片将无 VLM 描述）";
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
    li.appendChild(el("span", "cat-icon", "分类"));
    li.appendChild(el("span", "cat-name", cat.name));
    const count = State.docs.filter((d) => d.cat === cat.id).length;
    li.appendChild(el("span", "cat-count", String(count)));
    const del = el("button", "cat-del", "删除");
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

/* ───────────────────────── 文档渲染（知识库条目） ───────────────────────── */
function renderDocList() {
  const list = $("docList");
  list.innerHTML = "";
  for (const doc of State.docs) {
    const li = el("li");
    const type = doc.parser === "web" ? "网页" : doc.parser === "mineru" ? "MinerU" : "PDF";
    const titleWrap = el("div", "doc-title-wrap");
    const titleRow = el("div", "doc-title-row");
    titleRow.appendChild(el("span", "doc-type", type));
    titleRow.appendChild(el("span", "", doc.name || "未命名"));
    titleWrap.appendChild(titleRow);
    if (doc.url) {
      const urlLink = el("a", "doc-url", doc.url);
      urlLink.href = doc.url;
      urlLink.target = "_blank";
      urlLink.rel = "noopener noreferrer";
      titleWrap.appendChild(urlLink);
    }
    titleWrap.appendChild(el("span", "pages", fmtPages(doc.pages ? doc.pages.length : 0)));
    li.appendChild(titleWrap);

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

    const del = el("button", "del-doc", "删除");
    del.title = "删除条目";
    del.addEventListener("click", () => {
      State.docs = State.docs.filter((d) => d.id !== doc.id);
      Store.saveDocs(State.docs);
      if (doc.repositoryId) window.MultimodalRAG?.repository.deleteDocument(doc.repositoryId).catch((error) => console.warn("IndexedDB 删除失败", error));
      renderAll();
    });
    li.appendChild(del);
    list.appendChild(li);
  }
  const webCount = State.docs.filter((d) => d.parser === "web").length;
  const pdfCount = State.docs.length - webCount;
  $("docStats").textContent = `共 ${State.docs.length} 个条目（PDF ${pdfCount} / 网页 ${webCount}） / ${State.docs.reduce((s, d) => s + (d.pages ? d.pages.length : 0), 0)} 页`;
  $("docBadge").textContent = String(State.docs.length);
  $("buildIndex").disabled = State.docs.length === 0;
}

function invalidateIndex() {
  if (State.chunks.length || State.bm25) {
    State.chunks = [];
    State.bm25 = null;
    State.vectors = null;
    State.media = [];
    addMessage("system", "文档或分类已变化，请重新建立索引。");
  }
}

function renderAll() {
  renderCatList();
  renderCatSelects();
  renderDocList();
}

/* ───────────────────────── Web 搜索与网页抓取（知识库） ───────────────────────── */
function webApiBase(settings) {
  // 同源代理模式下所有 Web 抓取/搜索都经 /proxy/web/*
  return settings.callMode === "proxy" ? "/proxy/web" : null;
}

async function webFetchUrl(url, settings) {
  // 1) 优先经同源代理（自托管 web/proxy.py 时可用，绕过 CORS）
  const base = webApiBase(settings);
  if (base) {
    const resp = await fetch(`${base}/fetch?url=${encodeURIComponent(url)}`);
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data.error || `抓取失败 ${resp.status}`);
    return data;
  }
  // 2) 直连尝试（仅当目标网站允许 CORS 时可用，多数情况会被浏览器拦截）
  const resp = await fetch(url, { method: "GET", mode: "cors" });
  if (!resp.ok) throw new Error(`网页请求失败 ${resp.status}`);
  const html = await resp.text();
  const doc = new DOMParser().parseFromString(html, "text/html");
  const titleEl = doc.querySelector("title");
  const title = (titleEl && titleEl.textContent ? titleEl.textContent : "").trim() || url;
  doc.querySelectorAll("script,style,noscript,nav,footer,header,aside,iframe,svg").forEach((n) => n.remove());
  const body = doc.body;
  const text = (body && body.innerText ? body.innerText : "").replace(/\n{3,}/g, "\n\n").trim();
  if (!text) throw new Error("网页未提取到正文（可能为动态渲染页面）");
  return { url, title, text };
}

async function webSearch(query, settings) {
  const provider = settings.webProvider || "duckduckgo";
  const base = webApiBase(settings);
  if (base) {
    const resp = await fetch(`${base}/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, provider, api_key: settings.webTavilyKey || "", max_results: 6 }),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data.error || `搜索失败 ${resp.status}`);
    return data.results || [];
  }
  // 直连：仅 Tavily 支持浏览器 CORS 时可用
  if (provider === "tavily" && settings.webTavilyKey) {
    const resp = await fetch("https://api.tavily.com/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ api_key: settings.webTavilyKey, query, max_results: 6 }),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data.error || `搜索失败 ${resp.status}`);
    return (data.results || []).map((r) => ({ title: r.title, url: r.url, snippet: r.content || "" }));
  }
  throw new Error("直连模式搜索需要服务支持 CORS，请改用「同源代理」模式（自托管 web/proxy.py）");
}

function renderWebResults(results, settings) {
  const box = $("webResults");
  box.innerHTML = "";
  if (!results || !results.length) {
    box.appendChild(el("p", "hint", "未找到相关结果。"));
    return;
  }
  const list = el("div", "web-result-list");
  results.forEach((r, i) => {
    const item = el("div", "web-result");
    const head = el("div", "web-result-head");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.className = "web-result-check";
    cb.dataset.url = r.url;
    cb.checked = i === 0; // 默认勾选第一个
    head.appendChild(cb);
    const link = document.createElement("a");
    link.href = r.url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = r.title || r.url;
    head.appendChild(link);
    item.appendChild(head);
    item.appendChild(el("div", "web-result-url", r.url));
    if (r.snippet) item.appendChild(el("div", "web-result-snippet", r.snippet));
    const fetchBtn = el("button", "btn small ghost web-result-fetch", "＋ 抓取此页");
    fetchBtn.addEventListener("click", () => fetchWebDocs([r.url], collectSettings()));
    item.appendChild(fetchBtn);
    list.appendChild(item);
  });
  box.appendChild(list);
  const fetchSel = el("button", "btn primary web-fetch-selected", "⬇ 抓取选中页面");
  fetchSel.addEventListener("click", handleWebFetchSelected);
  box.appendChild(fetchSel);
}

async function handleWebSearch() {
  const settings = collectSettings();
  const q = $("webSearchQuery").value.trim();
  if (!q) {
    addMessage("system", "请输入搜索关键词。");
    return;
  }
  const box = $("webResults");
  box.innerHTML = "";
  box.appendChild(el("p", "hint", "搜索中…"));
  try {
    const results = await webSearch(q, settings);
    renderWebResults(results, settings);
  } catch (e) {
    box.innerHTML = "";
    box.appendChild(el("p", "hint error-text", `搜索失败：${e.message}`));
    addMessage("error", `Web 搜索失败：${e.message}`);
  }
}

async function handleWebFetch() {
  const settings = collectSettings();
  const raw = $("webUrlInput").value;
  const urls = raw.split("\n").map((s) => s.trim()).filter((s) => /^https?:\/\//i.test(s));
  if (!urls.length) {
    addMessage("system", "请粘贴一个或多个 http/https 网址（每行一个）。");
    return;
  }
  await fetchWebDocs(urls, settings);
}

async function handleWebFetchSelected() {
  const settings = collectSettings();
  const urls = [...document.querySelectorAll(".web-result-check:checked")].map((cb) => cb.dataset.url);
  if (!urls.length) {
    addMessage("system", "请先勾选要抓取的搜索结果。");
    return;
  }
  await fetchWebDocs(urls, settings);
}

async function fetchWebDocs(urls, settings) {
  const targetCat = $("uploadCat").value;
  let ok = 0;
  for (const url of urls) {
    addMessage("system", `正在抓取：${url}…`);
    try {
      const data = await webFetchUrl(url, settings);
      if (!data.text || !data.text.trim()) {
        addMessage("error", `抓取 ${url} 未提取到正文。`);
        continue;
      }
      addWebDoc({ title: data.title || url, url: data.url || url, text: data.text, cat: targetCat });
      ok++;
    } catch (e) {
      addMessage("error", `抓取 ${url} 失败：${e.message}`);
    }
  }
  if (ok) addMessage("system", `已抓取 ${ok} 个网页加入知识库。`);
}

function addWebDoc({ title, url, text, cat }) {
  const doc = {
    id: uid("doc"),
    name: title || url,
    cat: cat || "",
    pages: [{ page: 1, text }],
    parser: "web",
    url,
    addedAt: Date.now(),
    media: [],
  };
  State.docs.push(doc);
  Store.saveDocs(State.docs);
  invalidateIndex();
  renderAll();
  return doc;
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
      const parsed = await parser.parse(file, settings);
      // 兼容两种返回：数组（旧逻辑，仅 pages）或 { pages, media }
      const pages = Array.isArray(parsed) ? parsed : parsed.pages;
      const parsedMedia = (!Array.isArray(parsed) && parsed.media) || [];
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
      // 媒体资产补全 docId 前缀，便于多模态检索时按文档定位
      doc.media = parsedMedia.map((m) => ({
        ...m,
        id: `${doc.id}_${m.id}`,
        docId: doc.id,
      }));
      State.docs.push(doc);
      Store.saveDocs(State.docs);
      const mediaTip = doc.media.length ? `，检测到 ${doc.media.length} 处图/表引用` : "";
      addMessage("system", `已添加 ${file.name}：${pages.length} 页${targetCat ? `（分类：${catName(targetCat)}）` : ""}${mediaTip}。`);
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

    const del = el("button", "conv-del", "删除");
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
    addMessage("system", "上传 PDF 并建立索引后，即可基于文档提问。无需检索的问题会直接回答。");
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

/* ───────────────────────── RAGAS 评估（第 5 大功能） ───────────────────────── */
const EvalStore = {
  KEY: "pdfchat.evalset.v1",
  load() {
    try {
      const v = JSON.parse(localStorage.getItem(this.KEY) || "{}");
      return (v && typeof v === "object") ? v : {};
    } catch (e) {
      return {};
    }
  },
  save(q, gt) {
    try {
      localStorage.setItem(this.KEY, JSON.stringify({ questions: q, groundTruth: gt }));
    } catch (e) {
      console.warn("测试集保存失败", e);
    }
  },
};

const EVAL_METRICS = [
  { key: "faithfulness", name: "忠实度", needTruth: false,
    desc: "答案是否忠于检索上下文" },
  { key: "answer_relevancy", name: "答案相关性", needTruth: false,
    desc: "答案是否充分回答问题" },
  { key: "context_precision", name: "上下文精确率", needTruth: false,
    desc: "检索片段中相关信息占比" },
  { key: "context_recall", name: "上下文召回率", needTruth: true,
    desc: "检索是否覆盖参考答案要点" },
  { key: "answer_correctness", name: "答案正确性", needTruth: true,
    desc: "答案与参考答案的一致程度" },
];

function buildMetricPrompt(metric, sample) {
  const contexts = (sample.contexts && sample.contexts.length)
    ? sample.contexts.map((c, i) => `[${i + 1}] ${c}`).join("\n")
    : "（无）";
  const base = "你是 RAG 系统评估器。请根据要求给出 0 到 1 之间的分数（0=差，1=优）。\n只输出 JSON：{\"score\": 数字}，不要输出其他内容。\n\n";
  switch (metric) {
    case "faithfulness":
      return base + `判断答案是否忠实于给定上下文：答案中的每个事实都应能从上下文中找到依据，不包含上下文之外的信息。\n\n上下文：\n${contexts}\n\n答案：\n${sample.answer}`;
    case "answer_relevancy":
      return base + `评估答案与问题的相关性：答案是否直接、充分、完整地回答了问题？\n\n问题：\n${sample.question}\n\n答案：\n${sample.answer}`;
    case "context_precision":
      return base + `给定问题与检索到的上下文片段，评估片段中真正与问题相关的信息所占比例（精确率）。\n\n问题：\n${sample.question}\n\n上下文：\n${contexts}`;
    case "context_recall":
      return base + `给定问题、参考答案与检索上下文，评估上下文中是否包含回答该问题所需的全部关键信息（召回率）。\n\n问题：\n${sample.question}\n\n参考答案：\n${sample.ground_truth}\n\n上下文：\n${contexts}`;
    case "answer_correctness":
      return base + `将模型答案与参考答案对比，评估答案的正确性与完整性（事实一致性、关键点覆盖）。\n\n问题：\n${sample.question}\n\n模型答案：\n${sample.answer}\n\n参考答案：\n${sample.ground_truth}`;
    default:
      return base + "请给出分数。";
  }
}

async function judgeMetric(settings, metric, sample) {
  const prompt = buildMetricPrompt(metric, sample);
  try {
    const resp = await fetch(apiUrl("/chat/completions", settings), {
      method: "POST",
      headers: apiHeaders(settings),
      body: JSON.stringify(
        apiBody(
          {
            model: settings.model,
            messages: [
              { role: "system", content: "你是一个严谨、客观的 RAG 评估打分器。" },
              { role: "user", content: prompt },
            ],
            temperature: 0.0,
            max_tokens: 60,
          },
          settings
        )
      ),
    });
    if (!resp.ok) return null;
    const data = await resp.json();
    const text = (data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content) || "";
    let score = null;
    try {
      const parsed = JSON.parse(text);
      score = parseFloat(parsed.score);
    } catch (e) {
      const m = text.match(/\d+(\.\d+)?/);
      if (m) score = parseFloat(m[0]);
    }
    if (score === null || isNaN(score)) return null;
    return Math.max(0, Math.min(1, score));
  } catch (e) {
    console.warn(`指标 ${metric} 评判失败（已忽略）:`, e);
    return null;
  }
}

/* 评估用检索：复用 BM25 + 可选向量 */
async function evalRetrieve(question, settings) {
  const filter = chatCatFilter();
  const qTokens = tokenize(question);
  const bm25Hits = State.bm25 ? State.bm25.search(qTokens, settings.topK * 2, filter) : [];
  let vectorHits = [];
  if ((settings.retrievalMode === "vector" || settings.retrievalMode === "hybrid") &&
      State.vectors && settings.embedModel) {
    try {
      const [qvec] = await embedTexts([question], settings, { signal: controller.signal });
      vectorHits = vectorSearch(qvec, State, settings.topK * 2, filter);
    } catch (e) {
      console.warn("评估向量检索失败（忽略）:", e);
    }
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
  return topChunks;
}

function setEvalProgress(pct, status) {
  $("evalProgress").classList.remove("hidden");
  $("evalProgressBar").style.width = `${Math.max(2, pct)}%`;
  if (status !== undefined) $("evalStatus").textContent = status;
}

function scoreClass(v) {
  if (v === null || v === undefined || isNaN(v)) return "na";
  if (v >= 0.8) return "good";
  if (v >= 0.6) return "mid";
  return "bad";
}

function renderEvalResults(results) {
  // 汇总分数
  const scoresBox = $("evalScores");
  scoresBox.innerHTML = "";
  for (const m of EVAL_METRICS) {
    const vals = results
      .map((r) => r.scores[m.key])
      .filter((v) => v !== null && v !== undefined && !isNaN(v));
    const avg = vals.length ? (vals.reduce((a, b) => a + b, 0) / vals.length) : null;
    const card = el("div", "eval-score-card " + scoreClass(avg));
    card.appendChild(el("div", "name", m.name));
    card.appendChild(el("div", "val", avg === null ? "—" : avg.toFixed(2)));
    card.appendChild(el("div", "sub", m.desc));
    scoresBox.appendChild(card);
  }

  // 明细表
  const detail = $("evalDetail");
  detail.innerHTML = "";
  const table = el("table");
  const thead = el("thead");
  const headTr = el("tr");
  headTr.appendChild(el("th", "", "问题"));
  headTr.appendChild(el("th", "", "模型答案"));
  for (const m of EVAL_METRICS) headTr.appendChild(el("th", "", m.name));
  thead.appendChild(headTr);
  table.appendChild(thead);

  const tbody = el("tbody");
  for (const r of results) {
    const tr = el("tr");
    const qTd = el("td", "q-text", r.sample.question);
    const aTd = el("td", "a-text", (r.sample.answer || "").slice(0, 300));
    tr.appendChild(qTd);
    tr.appendChild(aTd);
    for (const m of EVAL_METRICS) {
      const v = r.scores[m.key];
      const td = el("td", "num " + scoreClass(v), v === null || v === undefined || isNaN(v) ? "—" : v.toFixed(2));
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  detail.appendChild(table);

  $("evalStatus").textContent = `评估完成：${results.length} 个样本。分数基于 LLM 评判，仅供参考。`;
}

/* 运行 RAGAS 评估：生成答案 → LLM 评判各指标 */
async function runEvaluation() {
  const settings = collectSettings();
  if (!settings.apiKey || !settings.model) {
    $("evalStatus").textContent = "❌ 请先在「设置」中配置 API Key 与模型。";
    return;
  }
  if (!State.chunks.length) {
    $("evalStatus").textContent = "❌ 请先在「文档」页上传 PDF 并建立索引。";
    return;
  }

  const questions = $("evalQuestions").value.split("\n").map((s) => s.trim()).filter(Boolean);
  if (!questions.length) {
    $("evalStatus").textContent = "❌ 请至少输入一个测试问题。";
    return;
  }
  const truths = $("evalGroundTruth").value.split("\n").map((s) => s.trim());

  $("runEval").disabled = true;
  $("evalScores").innerHTML = "";
  $("evalDetail").innerHTML = "";

  try {
    // 阶段 1：对每个问题检索 + 生成答案
    const samples = [];
    const genSystem =
      "你是一个严谨的文档问答助手。请仅依据提供的参考片段回答用户问题。\n" +
      "要求：只使用参考片段中的信息，不要编造；若信息不足请说明；使用与问题相同的语言回答。";
    for (let i = 0; i < questions.length; i++) {
      const q = questions[i];
      setEvalProgress(3 + ((i + 1) / questions.length) * 27, `生成答案 ${i + 1}/${questions.length}…`);
      const chunks = await evalRetrieve(q, settings);
      const contexts = chunks.map((c) => c.text);
      let answer;
      if (contexts.length) {
        const context = chunks.map((c, j) => `[${j + 1}]（来源：${c.file} 第${c.page}页）\n${c.text}`).join("\n\n---\n\n");
        answer = await askLLM(settings, genSystem, `参考片段：\n${context}\n\n用户问题：${q}`);
      } else {
        answer = "（未检索到相关内容）";
      }
      samples.push({
        question: q,
        answer: answer || "",
        contexts,
        ground_truth: truths[i] || "",
      });
    }

    // 阶段 2：LLM 评判各指标
    const results = [];
    const total = samples.length * EVAL_METRICS.length;
    let done = 0;
    for (let i = 0; i < samples.length; i++) {
      const s = samples[i];
      const scores = {};
      for (const m of EVAL_METRICS) {
        done++;
        setEvalProgress(30 + (done / total) * 68, `评判指标 ${done}/${total}…`);
        if (m.needTruth && !s.ground_truth) {
          scores[m.key] = null; // 未提供参考答案，无法计算
          continue;
        }
        scores[m.key] = await judgeMetric(settings, m.key, s);
      }
      results.push({ sample: s, scores });
    }

    setEvalProgress(100, "评估完成，正在渲染结果…");
    renderEvalResults(results);
  } catch (e) {
    $("evalStatus").textContent = `❌ 评估失败：${e.message}`;
  } finally {
    $("runEval").disabled = false;
  }
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

async function persistWorkspaceToIndexedDB() {
  const repository = window.MultimodalRAG?.repository;
  if (!repository) return { documents: 0, warning: "IndexedDB 不可用，仍使用旧版会话存储" };
  let documents = 0;
  for (const doc of State.docs) {
    const fullText = (doc.pages || []).map((page) => page.text || "").join("\n");
    const fingerprint = await sha256(`${doc.url || doc.name || "document"}\u241f${fullText}`);
    const documentId = await stableId("doc", fingerprint);
    const media = [];
    for (const [index, item] of (doc.media || []).entries()) {
      media.push({
        ...item,
        id: await stableId("media", documentId, item.page || 1, item.type || "image", item.label || index, item.caption || item.searchText || ""),
        documentId,
        searchText: item.searchText || item.caption || "",
        data: item.dataUrl || item.data || null,
        mimeType: item.dataUrl?.match(/^data:([^;,]+)/)?.[1] || item.mimeType || "",
        quality: item.quality || (doc.parser === "local" ? "fallback" : "exact"),
        extractionMethod: item.extractionMethod || doc.parser || "legacy",
      });
    }
    const chunks = [];
    const references = [];
    for (const oldChunk of State.chunks.filter((item) => item.docId === doc.id)) {
      const chunkId = await stableId("chunk", documentId, oldChunk.page || 1, oldChunk.text);
      chunks.push({ ...oldChunk, id: chunkId, documentId, content: oldChunk.text, modality: "text" });
      for (const ref of detectMediaReferences(oldChunk.text, documentId, oldChunk.page || 1)) {
        const decision = resolveMediaReference(ref, media);
        references.push({
          id: await stableId("ref", chunkId, ref.label, ref.offset), chunkId, documentId,
          mediaId: decision.mediaId, mediaType: ref.mediaType, label: ref.label,
          page: ref.page, offset: ref.offset, confidence: decision.confidence,
          resolution: decision.resolution, reason: decision.reason,
        });
      }
    }
    await repository.upsertDocument({
      document: { id: documentId, fingerprint, name: doc.name || "未命名文档", categoryId: doc.cat || "", source: doc.url || doc.name || "", sourceType: doc.parser === "web" ? "web" : "pdf", parser: doc.parser || "local", pageCount: doc.pages?.length || 1, status: doc.parser === "local" ? "degraded" : "ready" },
      chunks, media, references,
    });
    doc.repositoryId = documentId;
    documents += 1;
  }
  Store.saveDocs(State.docs);
  return { documents };
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
    const persisted = await persistWorkspaceToIndexedDB();
    setProgress(25);
    State.bm25 = buildBM25(State.chunks);
    State.vectors = null;
    setProgress(50);

    // 多模态检索：汇总媒体引用图谱（文本块 → 图片/表格引用位置）
    const wantMultimodal = settings.retrievalMode === "multimodal";
    if (wantMultimodal) {
      State.media = State.docs.flatMap((d) => d.media || []);
      addMessage(
        "system",
        `已建立媒体引用图谱：${State.media.length} 个图片/表格引用` +
          (settings.vlmApiKey && settings.vlmModel ? "（VLM 已配置）" : "（未配置 VLM，图片将仅展示引用）")
      );
    }

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
        (wantMultimodal ? `，含 ${State.media.length} 个图片/表格引用` : "") +
        `；${persisted.documents || 0} 个文档已事务写入 IndexedDB。可以开始提问。`
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
  const streamView = streamIntoMessage(thinking);
  const controller = new AbortController();
  State.generationController = controller;
  setGenerationUI(true);

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
      const answer = await askLLM(settings, system, question, history, {
        signal: controller.signal,
        onDelta: (delta) => streamView.append(delta),
      });
      streamView.flush();
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

    // 4.1 多模态检索扩展：通过引用位置找到关联图片/表格（RAG-Anything 风格）
    let mediaBlock = "";
    let relatedMedia = [];
    const wantMultimodal = settings.retrievalMode === "multimodal";
    if (wantMultimodal) {
      relatedMedia = collectMediaForChunks(topChunks, State.media);
      const tables = relatedMedia.filter((m) => m.type === "table");
      const images = relatedMedia.filter((m) => m.type === "image" && (m.dataUrl || m.url));
      if (tables.length) {
        mediaBlock +=
          "\n\n【关联表格】\n" +
          tables
            .map((t) => `[${t.label} 第${t.page}页]\n${(t.caption || "").slice(0, 500)}`)
            .join("\n\n");
      }
      if (images.length) {
        if (settings.vlmApiKey && settings.vlmModel) {
          addMessage("system", `正在用 VLM（${settings.vlmModel}）理解 ${images.length} 张图片…`);
          try {
            const desc = await askVLM(settings, images, { signal: controller.signal });
            mediaBlock += "\n\n【关联图片（VLM 描述）】\n" + desc;
          } catch (e) {
            mediaBlock +=
              "\n\n【关联图片】\n" +
              images.map((m) => `[${m.label} 第${m.page}页]（VLM 描述失败：${e.message}）`).join("\n");
          }
        } else {
          // 未配置 VLM：弹窗提醒（仅一次）
          showVlmModal(
            "多模态检索命中了文档中的图片，但尚未配置 VLM 模型。图片将仅展示引用，无法生成图片描述。"
          );
          mediaBlock +=
            "\n\n【关联图片】\n" +
            images.map((m) => `[${m.label} 第${m.page}页]（未配置 VLM，请查看引用片段中的图片）`).join("\n");
        }
      }
    }

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
      `参考片段：\n${context}${mediaBlock}${memContext}\n\n用户问题：${question}`;

    const answer = await askLLM(settings, systemPrompt, userContent, history, {
      signal: controller.signal,
      onDelta: (delta) => streamView.append(delta),
    });

    streamView.flush();
    thinking.className = "msg assistant";

    // 5. 元信息：检索范围 + 记忆标签 + 媒体引用
    const meta = el("div", "meta");
    meta.appendChild(el("span", "chip", `检索范围：${scopeName}`));
    if (memHits.length) meta.appendChild(el("span", "chip", `相关历史：${memHits.length} 条`));
    if (wantMultimodal && relatedMedia.length) {
      meta.appendChild(el("span", "chip", `关联媒体：${relatedMedia.length} 个`));
    }
    thinking.appendChild(meta);

    // 6. 引用片段
    const src = el("div", "src");
    const summary = el("summary", "", "查看引用片段");
    src.appendChild(summary);
    topChunks.forEach((c, i) => {
      src.appendChild(el("div", "", `[${i + 1}] ${c.file} 第${c.page}页：${c.text.slice(0, 120)}…`));
      // 多模态：展示本分块引用的图片缩略图 / 表格文本
      if (wantMultimodal) {
        collectMediaForChunks([c], State.media).forEach((m) => {
          const row = el("div", "src-media");
          row.appendChild(el("span", "src-media-label", `${m.label}（第${m.page}页）`));
          if (m.type === "image" && (m.dataUrl || m.url)) {
            const img = document.createElement("img");
            img.src = m.dataUrl || m.url;
            img.alt = m.label;
            img.className = "src-media-img";
            img.loading = "lazy";
            img.referrerPolicy = "no-referrer";
            row.appendChild(img);
          } else if (m.type === "table") {
            row.appendChild(el("pre", "src-media-table", (m.caption || "").slice(0, 300)));
          }
          src.appendChild(row);
        });
      }
    });
    thinking.appendChild(src);

    // 保存回答到会话（第 2 层）
    pushConvMessage("assistant", answer, [meta.textContent || "assistant"]);
    renderMemories();
    extractMemories(question, answer, settings); // 第 3 层：后台提取记忆
  } catch (e) {
    const partial = streamView.flush();
    if (e.name === "AbortError") {
      thinking.textContent = partial || "已停止生成。";
      thinking.className = "msg assistant interrupted";
      thinking.appendChild(el("div", "meta", "生成已停止"));
      if (partial) pushConvMessage("assistant", partial, ["生成已停止"]);
    } else {
      thinking.textContent = partial ? `${partial}\n\n[流式连接中断：${e.message}]` : `请求失败：${e.message}`;
      thinking.className = partial ? "msg assistant interrupted" : "msg error";
      if (partial) pushConvMessage("assistant", partial, ["流式连接中断"]);
    }
  } finally {
    if (State.generationController === controller) State.generationController = null;
    setGenerationUI(false);
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
    syncVlmStat();
    addMessage("system", "设置已保存到本浏览器。");
  });
  $("useLlmKey").addEventListener("click", () => {
    $("vlmApiKey").value = $("apiKey").value;
    addMessage("system", "已复用 LLM API Key 到 VLM 配置。");
  });
  $("vlmModalClose").addEventListener("click", closeVlmModal);
  $("vlmModalGoto").addEventListener("click", () => {
    closeVlmModal();
    switchView("settings");
  });
  $("vlmModal").addEventListener("click", (e) => {
    if (e.target === $("vlmModal")) closeVlmModal();
  });
  $("clearData").addEventListener("click", async () => {
    const repository = window.MultimodalRAG?.repository;
    if (repository) {
      for (const doc of State.docs) if (doc.repositoryId) await repository.deleteDocument(doc.repositoryId);
    }
    Store.clearAll();
    State.docs = [];
    State.chunks = [];
    State.bm25 = null;
    State.vectors = null;
    State.media = [];
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
  $("stopBtn").addEventListener("click", () => {
    if (State.generationController) State.generationController.abort();
  });

  // 评估
  $("saveEvalSet").addEventListener("click", () => {
    const q = $("evalQuestions").value;
    const gt = $("evalGroundTruth").value;
    EvalStore.save(q, gt);
    $("evalStatus").textContent = "测试集已保存到本浏览器。";
  });
  $("runEval").addEventListener("click", runEvaluation);

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

  // Web 搜索与抓取（知识库）
  $("webProvider").addEventListener("change", syncWebProviderUI);
  $("webSearchBtn").addEventListener("click", handleWebSearch);
  $("webSearchQuery").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      handleWebSearch();
    }
  });
  $("webFetchBtn").addEventListener("click", handleWebFetch);

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
    // 恢复媒体引用图谱（多模态检索用）
    State.media = State.docs.flatMap((d) => d.media || []);
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
    // 恢复测试集
    const ev = EvalStore.load();
    if (ev.questions) $("evalQuestions").value = ev.questions;
    if (ev.groundTruth) $("evalGroundTruth").value = ev.groundTruth;
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



