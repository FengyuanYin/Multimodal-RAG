import { LIMITS } from "./constants.js";

export class ValidationError extends Error {
  constructor(code, message, details = {}) { super(message); this.name = "ValidationError"; this.code = code; this.details = details; }
}

export function validateHttpUrl(value, { allowLocalhost = true } = {}) {
  let url;
  try { url = new URL(value); } catch { throw new ValidationError("INVALID_URL", "请输入有效的 HTTP(S) 地址"); }
  if (!['http:', 'https:'].includes(url.protocol)) throw new ValidationError("UNSAFE_SCHEME", "仅允许 HTTP(S) 地址");
  if (url.username || url.password) throw new ValidationError("URL_CREDENTIALS", "地址中不能包含用户名或密码");
  if (!allowLocalhost && /^(localhost|127\.|\[::1\]$)/i.test(url.hostname)) throw new ValidationError("LOCAL_ADDRESS", "不允许本地地址");
  return url;
}

export function validateCapacity({ fileBytes = 0, documentChars = 0, mediaBytes = 0 }) {
  const checks = [[fileBytes, LIMITS.fileBytes, "FILE_TOO_LARGE"], [documentChars, LIMITS.documentChars, "DOCUMENT_TOO_LARGE"], [mediaBytes, LIMITS.mediaBytes, "MEDIA_TOO_LARGE"]];
  for (const [actual, limit, code] of checks) if (actual > limit) throw new ValidationError(code, `内容超过容量上限（${actual}/${limit}）`, { actual, limit });
  return true;
}

export function redact(value) {
  let text = typeof value === "string" ? value : JSON.stringify(value ?? "");
  text = text.replace(/\b(sk-|mineru_|tvly-)[A-Za-z0-9_\-.]{6,}/gi, "$1***");
  text = text.replace(/([?&](?:api_?key|key|token)=)[^&#\s]+/gi, "$1***");
  text = text.replace(/(authorization\s*[:=]\s*(?:bearer\s+)?)[^\s,}]+/gi, "$1***");
  return text;
}

export function disclosureFor(operation) {
  const map = {
    llm: "发送问题、选中的证据片段和会话上下文到所配置的模型服务。",
    embedding: "发送待向量化的文本到所配置的嵌入服务。",
    mineru: "发送完整 PDF 到所配置的 MinerU 服务。",
    web: "发送搜索词或目标 URL 到所配置的搜索/代理服务。",
  };
  return map[operation] || "不向外部服务发送数据。";
}
