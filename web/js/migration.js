import { STORES } from "./constants.js";
import { sha256, stableId } from "./media-association.js";

const readJson = (key, fallback) => { try { return JSON.parse(localStorage.getItem(key) || JSON.stringify(fallback)); } catch { return fallback; } };

export function legacySecrets() {
  const settings = readJson("pdfchat.settings.v1", {});
  const names = ["apiKey", "vlmApiKey", "mineruApiKey", "webTavilyKey"];
  return names.filter((name) => settings[name]).map((name) => ({ name, masked: `${String(settings[name]).slice(0, 4)}***`, value: settings[name] }));
}

export async function migrateLegacy(repository, { persistSecrets = false } = {}) {
  await repository.open();
  const done = (await repository.all(STORES.metadata)).find((item) => item.key === "legacyMigrationV2");
  if (done) return { migrated: false, secretsPending: legacySecrets().length };
  const docs = readJson("pdfchat.docs.v1", []);
  let migrated = 0;
  for (const old of docs) {
    const content = old.text || (old.chunks || []).map((chunk) => chunk.text || chunk.content || "").join("\n");
    if (!content) continue;
    const fingerprint = await sha256(`${old.name || old.title || "legacy"}\u241f${content}`);
    const documentId = old.id || await stableId("doc", fingerprint);
    const chunks = (old.chunks?.length ? old.chunks : [{ text: content }]).map((chunk, index) => ({ id: chunk.id || `${documentId}_chunk_${String(index).padStart(4, "0")}`, text: chunk.text || chunk.content || "", page: chunk.page || 1, modality: chunk.modality || "text" }));
    await repository.upsertDocument({ document: { id: documentId, fingerprint, name: old.name || old.title || "旧版文档", categoryId: old.categoryId || old.catId || "", status: "ready", migratedFrom: "localStorage" }, chunks, media: old.media || [], references: old.references || [] });
    migrated += 1;
  }
  const settings = readJson("pdfchat.settings.v1", {});
  for (const item of legacySecrets()) sessionStorage.setItem(`mmrag.secret.${item.name}`, item.value);
  if (persistSecrets) for (const item of legacySecrets()) localStorage.setItem(`mmrag.secret.${item.name}`, item.value);
  const tx = repository.db.transaction(STORES.metadata, "readwrite");
  tx.objectStore(STORES.metadata).put({ key: "legacyMigrationV2", migratedAt: Date.now(), documentCount: migrated });
  return { migrated: true, documentCount: migrated, settings, secretsPending: persistSecrets ? 0 : legacySecrets().length };
}
