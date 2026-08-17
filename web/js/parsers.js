import { validateCapacity, validateHttpUrl } from "./security.js";
import { detectMediaReferences, resolveMediaReference, sha256, stableId } from "./media-association.js";

export async function parsePdfLocally(file, { pdfjs = globalThis.pdfjsLib, signal, onProgress = () => {} } = {}) {
  validateCapacity({ fileBytes: file.size });
  if (!pdfjs) throw new Error("pdf.js 未加载");
  signal?.throwIfAborted();
  const bytes = await file.arrayBuffer();
  const task = pdfjs.getDocument({ data: bytes });
  signal?.addEventListener("abort", () => task.destroy(), { once: true });
  const pdf = await task.promise;
  const fingerprint = await sha256(bytes);
  const documentId = await stableId("doc", file.name, fingerprint);
  const chunks = [], media = [], references = [];
  for (let pageNo = 1; pageNo <= pdf.numPages; pageNo += 1) {
    signal?.throwIfAborted();
    const page = await pdf.getPage(pageNo);
    const content = await page.getTextContent();
    const text = content.items.map((item) => item.str).join(" ").replace(/\s+/g, " ").trim();
    const chunkId = await stableId("chunk", documentId, pageNo, text);
    chunks.push({ id: chunkId, documentId, page: pageNo, text, content: text, modality: "text" });
    const viewport = page.getViewport({ scale: 1 });
    const canvas = document.createElement("canvas"); canvas.width = Math.ceil(viewport.width); canvas.height = Math.ceil(viewport.height);
    await page.render({ canvasContext: canvas.getContext("2d"), viewport }).promise;
    const snapshotId = await stableId("media", documentId, pageNo, "page_snapshot");
    media.push({ id: snapshotId, documentId, type: "page_snapshot", page: pageNo, label: `第${pageNo}页`, caption: "页面快照（降级证据）", searchText: text, data: canvas.toDataURL("image/jpeg", .72), mimeType: "image/jpeg", quality: "fallback", extractionMethod: "pdfjs_page_render" });
    for (const ref of detectMediaReferences(text, documentId, pageNo)) {
      const decision = resolveMediaReference(ref, media);
      references.push({ id: await stableId("ref", chunkId, ref.label, ref.offset), chunkId, documentId, mediaId: decision.mediaId, mediaType: ref.mediaType, label: ref.label, page: pageNo, offset: ref.offset, ...decision });
    }
    onProgress({ page: pageNo, pages: pdf.numPages, percent: Math.round(pageNo / pdf.numPages * 100) });
  }
  return { document: { id: documentId, fingerprint, name: file.name, sourceType: "pdf", parser: "pdfjs", pageCount: pdf.numPages, status: "degraded", warnings: ["本地模式提供页面快照，未声明为精确图表抽取"] }, chunks, media, references };
}

export async function normalizeWebDocument({ url, title, text }) {
  const normalizedUrl = validateHttpUrl(url).href.replace(/#.*$/, "");
  const normalizedText = String(text || "").replace(/\s+/g, " ").trim();
  if (!normalizedText) throw new Error("网页正文为空");
  const fingerprint = await sha256(`${normalizedUrl}\u241f${normalizedText}`);
  const id = await stableId("doc", fingerprint);
  return { document: { id, fingerprint, name: title || normalizedUrl, source: normalizedUrl, sourceType: "web", status: "ready" }, chunks: [{ id: await stableId("chunk", id, normalizedText), documentId: id, page: 1, text: normalizedText, modality: "text" }], media: [], references: [] };
}

export async function normalizeMinerU(result, metadata = {}) {
  const pages = result.pages || result.data?.pages || [];
  const source = metadata.source || result.source || "mineru";
  const fingerprint = await sha256(JSON.stringify(pages));
  const documentId = await stableId("doc", source, fingerprint);
  const chunks = [], media = [], pendingRefs = [];
  for (const [index, page] of pages.entries()) {
    const pageNo = Number(page.page_no ?? page.page ?? index + 1);
    const text = page.markdown || page.text || "";
    const chunkId = await stableId("chunk", documentId, pageNo, text);
    chunks.push({ id: chunkId, documentId, page: pageNo, text, modality: "mixed" });
    for (const [mediaIndex, item] of (page.images || page.figures || []).entries()) media.push({ id: await stableId("media", documentId, pageNo, "image", mediaIndex, item.url || item.caption), documentId, type: "image", page: pageNo, label: item.label || `图${mediaIndex + 1}`, caption: item.caption || "", searchText: item.caption || "", externalUrl: item.url || "", quality: "exact", extractionMethod: "mineru" });
    for (const [tableIndex, item] of (page.tables || []).entries()) media.push({ id: await stableId("media", documentId, pageNo, "table", tableIndex, item.text || item.markdown), documentId, type: "table", page: pageNo, label: item.label || `表${tableIndex + 1}`, caption: item.caption || "", searchText: item.text || item.markdown || item.caption || "", dataText: item.text || item.markdown || "", quality: "exact", extractionMethod: "mineru" });
    pendingRefs.push(...detectMediaReferences(text, documentId, pageNo).map((ref) => ({ ref, chunkId })));
  }
  const references = [];
  for (const { ref, chunkId } of pendingRefs) references.push({ id: await stableId("ref", chunkId, ref.label, ref.offset), chunkId, documentId, mediaType: ref.mediaType, label: ref.label, page: ref.page, offset: ref.offset, ...resolveMediaReference(ref, media) });
  return { document: { id: documentId, fingerprint, name: metadata.title || result.title || "MinerU 文档", sourceType: "pdf", parser: "mineru", pageCount: pages.length, status: "ready" }, chunks, media, references };
}
