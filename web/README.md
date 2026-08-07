# 📄 PDF Chat — 浏览器本地 PDF 问答

一个**纯前端**的 PDF 问答工具。访问者打开页面后：

1. 上传自己的 PDF（在浏览器本地解析，不上传任何服务器）
2. 填写自己的 API Key 与 LLM 配置（Key 只存本浏览器 localStorage，只发往自己填写的 API 地址）
3. 建立本地索引（BM25 关键词检索，可选 Embedding 向量增强）
4. 基于文档提问，答案附引用片段

**文档与密钥均不出浏览器**，因此代码可以放心公开。

---

## ✨ 功能

- 🌐 **Web 搜索与抓取（知识库）**：侧边栏「知识库」页支持搜索互联网（DuckDuckGo 免费 / Tavily）或直接粘贴网址，把网页正文抓取进知识库，与 PDF 一起建立检索索引
- 📄 PDF 解析：pdf.js 在浏览器本地完成，支持多文件、分页
- 🔍 本地 BM25 检索：中文 bigram 分词，无需任何模型即可用
- 🧠 向量增强（可选）：配置 Embedding 模型后自动启用向量检索，并与 BM25 做 RRF 融合
- 🖼 **多模态检索（可选）**：检测文档中的「图N / 表N」引用位置，命中文本时自动附带关联图片/表格；配置 **VLM** 后，图片会由视觉语言模型理解后参与回答（RAG-Anything 风格）
- 💬 OpenAI 兼容 Chat Completions：适配 OpenAI / DeepSeek / OpenRouter / Groq / Moonshot 等
- 🔐 密钥仅存 localStorage，绝不发送到第三方服务器

> 多模态检索使用方式：在「设置」→ 检索模式中选择「多模态检索」，并在「VLM 配置」中填写 API Key 与模型（如 `gpt-4o` / `qwen-vl-max` / `glm-4v`）。未配置 VLM 时，检索仍会返回图片/表格引用，但不会生成图片描述，并会弹窗提醒。

### Web 搜索与抓取

- 搜索：默认使用 **DuckDuckGo**（免费、无需 Key）；也可切换 **Tavily**（专为 RAG 设计，返回正文片段，需要 API Key）。
- 抓取：搜索结果可勾选后批量抓取，也可直接粘贴网址（每行一个）抓取。
- 网页抓取会经「同源代理」`web/proxy.py` 的 `/proxy/web/fetch` 与 `/proxy/web/search` 端点完成（直连模式受浏览器 CORS 限制，多数网站无法直接抓取，建议自托管代理）。

---

## 🚀 部署到 GitHub Pages

1. 将 `web/` 目录（或整个仓库）推送到 GitHub 公开仓库。
2. 仓库 Settings → Pages → 选择分支（如 `main`）与目录（`/web` 或根目录）→ Save。
3. 打开 `https://<用户名>.github.io/<仓库名>/web/` 即可使用。

> 提示：GitHub Pages 不支持自定义服务器逻辑，纯前端模式需要 LLM 提供商支持 CORS。
> 多数 OpenAI 兼容服务（DeepSeek、OpenRouter、Groq、Moonshot 等）支持浏览器直连；
> 若你使用的服务不支持 CORS，请改用下方「同源代理」模式自托管。

---

## 🖥 同源代理模式（可选，自托管）

部分 LLM 提供商不支持浏览器 CORS，会被浏览器拦截。此时可自托管一个轻量代理：

```bash
# 使用本项目虚拟环境（或任意含 fastapi + httpx 的环境）
agenticrag\Scripts\python.exe -m uvicorn web.proxy:app --host 0.0.0.0 --port 8000
```

前端页面选择「API 调用方式 → 同源代理」即可。代理只做请求转发：
- **不存储**任何 API Key；
- Key 由浏览器放在 `X-API-Key` 头中，代理原样转发给用户填写的 `base_url`；
- 生产环境请自行加 HTTPS 与访问控制。

---

## 🔒 安全说明

| 事项 | 说明 |
|------|------|
| API Key 存储 | 仅浏览器 `localStorage`，页面关闭后仍保留（可手动清除） |
| API Key 流向 | 直连模式仅发往用户填写的 Base URL；代理模式经你的自托管服务转发 |
| 文档数据 | 全部在浏览器本地解析/索引，不经过任何服务器 |
| 公开仓库 | 仓库不含任何密钥；`.gitignore` 已排除 `data/`、`agenticrag/` 等敏感/大体积目录 |

---

## 🧩 目录结构

```
web/
├── index.html    # 页面结构
├── style.css     # 样式
├── app.js        # 核心逻辑（解析/分词/BM25/向量/对话）
├── proxy.py      # 可选同源代理（FastAPI）
└── README.md     # 本文档
```

---

## ⚠️ 免责声明

本项目为开源示例，用于演示 BYOK（Bring Your Own Key）模式。请勿在页面中泄露他人密钥；
自行评估所使用 LLM 提供商的服务条款与数据合规要求。
