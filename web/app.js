/* =========================================================================
 * PDF Chat — 浏览器本地 RAG
 * -------------------------------------------------------------------------
 * 核心原则：
 *   1. 所有文档解析与检索均在浏览器本地完成，不上传任何服务器。
 *   2. API Key 仅保存在浏览器 localStorage，仅发送到用户自配的 API 地址。
 *   3. 检索默认使用本地 BM25（中文 bigram 分词），可选 Embedding API 增强。
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
    meta.textContent = extra;
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

function loadSettingsIntoForm() {
  const s = Settings.load();
  $("apiKey").value = s.apiKey || "";
  $("callMode").value = s.callMode || "direct";
  $("baseUrl").value = s.baseUrl || "https://api.openai.com/v1";
  $("model").value = s.model || "gpt-4o-mini";
  $("embedModel").value = s.embedModel || "";
  $("topK").value = s.topK || 5;
  $("chunkSize").value = s.chunkSize || 800;
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
  };
}

/* ───────────────────────── 状态 ───────────────────────── */
const State = {
  // 解析后的原始文本（按页）：[{ file, page, text }]
  pages: [],
  // 分块结果：[{ id, file, page, text, tokens }]
  chunks: [],
  // BM25 数据结构
  bm25: null,
  // 可选向量数据：{ vectors: Float32Array[], norm: number[], index: number[] }
  vectors: null,
};

/* ───────────────────────── 分词（中文 bigram + 英文单词） ───────────────────────── */
function tokenize(text) {
  const tokens = [];
  // 英文/数字单词
  const en = text.toLowerCase().match(/[a-z0-9]+/g) || [];
  tokens.push(...en);
  // 中文：bigram 分词（如 “人工智能” -> 人工、工智、智能）
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

/* ───────────────────────── BM25 实现 ───────────────────────── */
function buildBM25(chunks) {
  const docTokens = chunks.map((c) => c.tokens);
  const N = docTokens.length;
  const df = new Map(); // 文档频率
  const k1 = 1.5;
  const b = 0.75;

  const avgdl = docTokens.reduce((sum, t) => sum + t.length, 0) / Math.max(1, N);

  for (const tokens of docTokens) {
    const seen = new Set(tokens);
    for (const t of seen) df.set(t, (df.get(t) || 0) + 1);
  }

  function idf(t) {
    const n = df.get(t) || 0;
    // 平滑：避免分母为 0；df=N 时 idf 为 0（此时不贡献排序）
    return Math.log((N - n + 0.5) / (n + 0.5) + 1);
  }

  return {
    search(queryTokens, topK) {
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
        .filter((r) => r.score > 0)
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

/* ───────────────────────── API 请求封装（直连 / 同源代理） ───────────────────────── */
function apiUrl(path, settings) {
  if (settings.callMode === "proxy") {
    return `/proxy${path}`;
  }
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
  if (settings.callMode === "proxy") {
    // 代理需要知道真实目标地址；Key 已在请求头中
    return { ...body, base_url: settings.baseUrl };
  }
  return body;
}

async function buildVectors(chunks, settings) {
  // 分批调用，每批 64 块
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

function vectorSearch(queryVec, state, topK) {
  const { vectors, norm, dim } = state.vectors;
  const scores = [];
  let qsum = 0;
  for (let j = 0; j < dim; j++) qsum += queryVec[j] * queryVec[j];
  const qnorm = Math.sqrt(qsum) || 1;

  for (let i = 0; i < state.chunks.length; i++) {
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

/* ───────────────────────── PDF 解析 ───────────────────────── */
async function parsePDF(file) {
  const buf = await file.arrayBuffer();
  const pdf = await pdfjsLib.getDocument({ data: buf }).promise;
  const pages = [];
  for (let p = 1; p <= pdf.numPages; p++) {
    const page = await pdf.getPage(p);
    const content = await page.getTextContent();
    // 按行还原文本（y 坐标相近的项归为同一行）
    const lines = new Map();
    for (const item of content.items) {
      if (!("str" in item) || !item.str.trim()) continue;
      const y = Math.round(item.transform[5] / 4) * 4; // 容差 4pt 视为同一行
      const line = lines.get(y) || "";
      lines.set(y, line + item.str);
    }
    const text = [...lines.values()].join("\n");
    if (text.trim()) {
      pages.push({ file: file.name, page: p, text });
    }
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

function buildChunks(pages, chunkSize) {
  const overlap = Math.min(100, Math.floor(chunkSize * 0.15));
  const chunks = [];
  let seq = 0;
  for (const p of pages) {
    for (const part of splitChunks(p.text, chunkSize, overlap)) {
      const text = part.trim();
      if (!text) continue;
      chunks.push({
        id: `c${seq++}`,
        file: p.file,
        page: p.page,
        text,
        tokens: tokenize(text),
      });
    }
  }
  return chunks;
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

/* ───────────────────────── 主流程：建立索引 ───────────────────────── */
async function handleBuildIndex() {
  if (State.pages.length === 0) {
    addMessage("system", "请先上传 PDF 文档。");
    return;
  }
  const settings = collectSettings();
  if (!settings.apiKey && settings.embedModel) {
    addMessage("error", "使用 Embedding 增强需要填写 API Key。");
    return;
  }

  $("buildIndex").disabled = true;
  try {
    // 1. 分块
    State.chunks = buildChunks(State.pages, settings.chunkSize);
    setProgress(25);

    // 2. BM25（总是建立，无需网络）
    State.bm25 = buildBM25(State.chunks);
    State.vectors = null;
    setProgress(50);

    // 3. 可选向量
    if (settings.embedModel && settings.apiKey) {
      addMessage("system", `正在生成向量索引（${State.chunks.length} 块）…`);
      State.vectors = await buildVectors(State.chunks, settings);
    }
    setProgress(100);

    addMessage(
      "system",
      `索引完成：${State.chunks.length} 个分块` +
        (State.vectors ? "（含向量检索）" : "（关键词 BM25 检索）") +
        `。可以开始提问。`
    );
  } catch (e) {
    addMessage("error", `建立索引失败：${e.message}`);
  } finally {
    hideProgress();
    $("buildIndex").disabled = false;
  }
}

/* ───────────────────────── 主流程：提问 ───────────────────────── */
async function handleAsk(question) {
  const settings = collectSettings();
  if (!settings.apiKey) {
    addMessage("error", "请先在上方填写 API Key。");
    return;
  }
  if (!settings.model) {
    addMessage("error", "请填写模型名称。");
    return;
  }
  if (!State.chunks.length) {
    addMessage("error", "请先上传 PDF 并建立索引。");
    return;
  }

  addMessage("user", question);
  const thinking = addMessage("assistant", "思考中…");
  $("askBtn").disabled = true;

  try {
    // 1. 检索
    const qTokens = tokenize(question);
    let bm25Hits = State.bm25 ? State.bm25.search(qTokens, settings.topK * 2) : [];

    let vectorHits = [];
    if (State.vectors && settings.embedModel) {
      const [qvec] = await embedTexts([question], settings);
      vectorHits = vectorSearch(qvec, State, settings.topK * 2);
    }

    // 2. 融合（有向量时 RRF 融合，否则直接用 BM25）
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
      thinking.textContent = "未检索到相关内容，请尝试换一种问法。";
      thinking.className = "msg assistant";
      return;
    }

    // 3. 组装上下文
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

    // 4. 调用 LLM
    const answer = await askLLM(settings, systemPrompt, `参考片段：\n${context}\n\n用户问题：${question}`);

    thinking.textContent = answer;
    thinking.className = "msg assistant";

    // 5. 展示来源
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
  // 设置
  $("saveSettings").addEventListener("click", () => {
    Settings.save(collectSettings());
    addMessage("system", "设置已保存到本浏览器。");
  });
  $("clearData").addEventListener("click", () => {
    localStorage.removeItem(Settings.KEY);
    loadSettingsIntoForm();
    addMessage("system", "本地设置已清除。");
  });
  $("toggleKey").addEventListener("click", () => {
    const input = $("apiKey");
    input.type = input.type === "password" ? "text" : "password";
  });

  // 上传
  const dz = $("dropZone");
  const fileInput = $("fileInput");
  dz.addEventListener("click", () => fileInput.click());
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

async function handleFiles(files) {
  if (!files.length) {
    addMessage("system", "请选择 PDF 文件。");
    return;
  }
  const totalStart = State.pages.length;
  for (const file of files) {
    addMessage("system", `正在解析：${file.name} …`);
    try {
      const pages = await parsePDF(file);
      State.pages.push(...pages);
      addMessage("system", `已解析 ${file.name}：${pages.length} 页。`);
    } catch (e) {
      addMessage("error", `解析 ${file.name} 失败：${e.message}`);
    }
  }

  // 更新文档列表与统计
  const added = State.pages.length - totalStart;
  $("docStats").textContent = `共 ${State.pages.length} 页 / ${new Set(State.pages.map((p) => p.file)).size} 个文件`;
  renderDocList();
  $("buildIndex").disabled = State.pages.length === 0;

  // 新文件加入后旧索引失效，提示重建
  if (added > 0 && (State.chunks.length || State.bm25)) {
    State.chunks = [];
    State.bm25 = null;
    State.vectors = null;
    addMessage("system", "文档已更新，请重新建立索引。");
  }
}

function renderDocList() {
  const list = $("docList");
  list.innerHTML = "";
  const byFile = new Map();
  for (const p of State.pages) {
    byFile.set(p.file, (byFile.get(p.file) || 0) + 1);
  }
  for (const [file, pages] of byFile) {
    const li = el("li");
    li.appendChild(el("span", "", file));
    li.appendChild(el("span", "pages", `${pages} 页`));
    const del = el("button", "", "移除");
    del.addEventListener("click", () => {
      State.pages = State.pages.filter((p) => p.file !== file);
      State.chunks = [];
      State.bm25 = null;
      State.vectors = null;
      $("docStats").textContent = State.pages.length
        ? `共 ${State.pages.length} 页 / ${new Set(State.pages.map((p) => p.file)).size} 个文件`
        : "未加载文档";
      renderDocList();
      $("buildIndex").disabled = State.pages.length === 0;
      addMessage("system", `已移除 ${file}，请重新建立索引。`);
    });
    li.appendChild(del);
    list.appendChild(li);
  }
}

/* ───────────────────────── 初始化 ───────────────────────── */
loadSettingsIntoForm();
bindEvents();
