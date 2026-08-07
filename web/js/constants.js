export const SCHEMA_VERSION = 2;
export const DB_NAME = "multimodal-rag-workspace";
export const STORES = Object.freeze({
  documents: "documents",
  chunks: "chunks",
  media: "media",
  references: "references",
  categories: "categories",
  conversations: "conversations",
  evalsets: "evalsets",
  metadata: "metadata",
});
export const LIMITS = Object.freeze({
  fileBytes: 50 * 1024 * 1024,
  documentChars: 8_000_000,
  mediaBytes: 12 * 1024 * 1024,
  requestTimeoutMs: 45_000,
  maxRetries: 2,
  topK: 20,
  rrfK: 60,
});
export const TASK_STATES = Object.freeze(["idle", "running", "degraded", "succeeded", "failed", "cancelled"]);
export const TERMINAL_STATES = new Set(["succeeded", "failed", "cancelled"]);
