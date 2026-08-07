const REF_PATTERN = /(?<image>(?:图\s*|Figure\s+|Fig\.?\s*)(?<imageNo>\d{1,3}))|(?<table>(?:表格?\s*|Table\s+)(?<tableNo>\d{1,3}))/giu;

export async function sha256(value) {
  const bytes = value instanceof ArrayBuffer ? new Uint8Array(value) : new TextEncoder().encode(String(value));
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}
export async function stableId(prefix, ...parts) { return `${prefix}_${(await sha256(parts.join("\u241f"))).slice(0, 20)}`; }

export function detectMediaReferences(text, documentId, page = 1) {
  const refs = [];
  for (const match of String(text || "").matchAll(REF_PATTERN)) {
    const type = match.groups.imageNo ? "image" : "table";
    const number = match.groups.imageNo || match.groups.tableNo;
    refs.push({ documentId, mediaType: type, label: `${type === "image" ? "图" : "表"}${number}`, page: Math.max(1, Number(page) || 1), offset: match.index, rawLabel: match[0] });
  }
  return refs;
}

export function resolveMediaReference(ref, media = []) {
  const docAssets = media.filter((item) => item.documentId === ref.documentId || item.docId === ref.documentId);
  const typed = docAssets.filter((item) => item.type === ref.mediaType);
  const sameLabel = typed.filter((item) => item.label === ref.label);
  const exact = sameLabel.filter((item) => Number(item.page || 1) === ref.page);
  if (exact.length === 1) return { mediaId: exact[0].id, confidence: 1, resolution: "exact", reason: "同文档、同页、同类型、同标签唯一匹配" };
  if (sameLabel.length === 1) return { mediaId: sameLabel[0].id, confidence: .85, resolution: "unique_label", reason: "同文档内标签唯一匹配" };
  const layout = typed.filter((item) => item.referenceLabel === ref.label && Number(item.page || 1) === ref.page);
  if (layout.length === 1) return { mediaId: layout[0].id, confidence: .9, resolution: "layout", reason: "解析器提供明确版面关系" };
  const snapshots = docAssets.filter((item) => item.type === "page_snapshot" && Number(item.page || 1) === ref.page);
  if (snapshots.length === 1) return { mediaId: snapshots[0].id, confidence: .35, resolution: "page_match", reason: "仅定位到页面快照" };
  return { mediaId: null, confidence: 0, resolution: "unresolved", reason: sameLabel.length > 1 ? "存在多个候选" : "未找到对应媒体资产" };
}
