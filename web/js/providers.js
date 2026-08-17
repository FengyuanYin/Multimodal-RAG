import { LIMITS } from "./constants.js";
import { redact, validateHttpUrl } from "./security.js";

export function setSecret(name, value, persistent = false) {
  const target = persistent ? localStorage : sessionStorage;
  target.setItem(`mmrag.secret.${name}`, value);
  (persistent ? sessionStorage : localStorage).removeItem(`mmrag.secret.${name}`);
}
export function getSecret(name) { return sessionStorage.getItem(`mmrag.secret.${name}`) || localStorage.getItem(`mmrag.secret.${name}`) || ""; }

export class ProviderError extends Error {
  constructor(stage, code, message, details = {}) { super(message); this.name = "ProviderError"; this.stage = stage; this.code = code; this.details = details; }
}

export async function policyFetch(url, options = {}) {
  const parsed = validateHttpUrl(url);
  const method = String(options.method || "GET").toUpperCase();
  const canRetry = ["GET", "HEAD", "OPTIONS"].includes(method);
  const attempts = canRetry ? Math.min(options.retries ?? LIMITS.maxRetries, LIMITS.maxRetries) + 1 : 1;
  let last;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(new DOMException("请求超时", "TimeoutError")), options.timeoutMs || LIMITS.requestTimeoutMs);
    const abort = () => controller.abort(options.signal.reason);
    options.signal?.addEventListener("abort", abort, { once: true });
    try {
      const response = await fetch(parsed, { ...options, signal: controller.signal });
      if (!response.ok) throw new ProviderError(options.stage || "request", `HTTP_${response.status}`, `服务返回 ${response.status}`);
      return response;
    } catch (error) {
      last = error;
      if (attempt + 1 >= attempts || options.signal?.aborted) break;
      await new Promise((resolve) => setTimeout(resolve, 250 * 2 ** attempt));
    } finally { clearTimeout(timer); options.signal?.removeEventListener("abort", abort); }
  }
  throw last instanceof ProviderError ? last : new ProviderError(options.stage || "request", last?.name === "AbortError" ? "ABORTED" : "NETWORK_ERROR", redact(last?.message || "网络请求失败"));
}

export async function openAIJson(baseUrl, path, apiKey, body, options = {}) {
  const response = await policyFetch(`${String(baseUrl).replace(/\/$/, "")}/${path.replace(/^\//, "")}`, { ...options, method: "POST", headers: { "content-type": "application/json", authorization: `Bearer ${apiKey}`, ...options.headers }, body: JSON.stringify(body), stage: options.stage || "model" });
  return response.json();
}
