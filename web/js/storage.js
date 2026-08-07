import { DB_NAME, SCHEMA_VERSION, STORES } from "./constants.js";

const requestPromise = (request) => new Promise((resolve, reject) => {
  request.onsuccess = () => resolve(request.result);
  request.onerror = () => reject(request.error);
});
const transactionPromise = (tx) => new Promise((resolve, reject) => {
  tx.oncomplete = () => resolve();
  tx.onerror = () => reject(tx.error);
  tx.onabort = () => reject(tx.error || new Error("IndexedDB 事务已回滚"));
});

export class BrowserKnowledgeRepository {
  constructor(name = DB_NAME) { this.name = name; this.db = null; }
  async open() {
    if (this.db) return this;
    const request = indexedDB.open(this.name, SCHEMA_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      const ensure = (name, options, indexes = []) => {
        const store = db.objectStoreNames.contains(name) ? request.transaction.objectStore(name) : db.createObjectStore(name, options);
        for (const [indexName, keyPath, opts] of indexes) if (!store.indexNames.contains(indexName)) store.createIndex(indexName, keyPath, opts);
      };
      ensure(STORES.documents, { keyPath: "id" }, [["fingerprint", "fingerprint", { unique: true }], ["categoryId", "categoryId", {}]]);
      ensure(STORES.chunks, { keyPath: "id" }, [["documentId", "documentId", {}]]);
      ensure(STORES.media, { keyPath: "id" }, [["documentId", "documentId", {}], ["documentPage", ["documentId", "page"], {}]]);
      ensure(STORES.references, { keyPath: "id" }, [["documentId", "documentId", {}], ["chunkId", "chunkId", {}], ["mediaId", "mediaId", {}]]);
      ensure(STORES.categories, { keyPath: "id" });
      ensure(STORES.conversations, { keyPath: "id" });
      ensure(STORES.evalsets, { keyPath: "id" });
      ensure(STORES.metadata, { keyPath: "key" });
    };
    this.db = await requestPromise(request);
    this.db.onversionchange = () => { this.db.close(); this.db = null; };
    return this;
  }
  async upsertDocument(bundle, beforeCommit) {
    await this.open();
    const names = [STORES.documents, STORES.chunks, STORES.media, STORES.references];
    const tx = this.db.transaction(names, "readwrite");
    try {
      const existing = bundle.document.fingerprint
        ? await requestPromise(tx.objectStore(STORES.documents).index("fingerprint").get(bundle.document.fingerprint)) : null;
      const documentId = existing?.id || bundle.document.id;
      const document = { ...bundle.document, id: documentId, updatedAt: Date.now(), createdAt: existing?.createdAt || Date.now() };
      tx.objectStore(STORES.documents).put(document);
      await this.#deleteChildren(tx, documentId);
      for (const chunk of bundle.chunks || []) tx.objectStore(STORES.chunks).put({ ...chunk, documentId });
      for (const item of bundle.media || []) tx.objectStore(STORES.media).put({ ...item, documentId });
      for (const ref of bundle.references || []) tx.objectStore(STORES.references).put({ ...ref, documentId });
      if (beforeCommit) await beforeCommit({ tx, documentId });
      await transactionPromise(tx);
      return documentId;
    } catch (error) {
      try { tx.abort(); } catch { /* already complete */ }
      throw error;
    }
  }
  async #deleteByIndex(tx, storeName, indexName, value) {
    const store = tx.objectStore(storeName);
    const keys = await requestPromise(store.index(indexName).getAllKeys(value));
    keys.forEach((key) => store.delete(key));
  }
  async #deleteChildren(tx, documentId) {
    await this.#deleteByIndex(tx, STORES.references, "documentId", documentId);
    await this.#deleteByIndex(tx, STORES.chunks, "documentId", documentId);
    await this.#deleteByIndex(tx, STORES.media, "documentId", documentId);
  }
  async deleteDocument(documentId) {
    await this.open();
    const names = [STORES.documents, STORES.chunks, STORES.media, STORES.references];
    const tx = this.db.transaction(names, "readwrite");
    await this.#deleteChildren(tx, documentId);
    tx.objectStore(STORES.documents).delete(documentId);
    await transactionPromise(tx);
  }
  async all(storeName) { await this.open(); return requestPromise(this.db.transaction(storeName).objectStore(storeName).getAll()); }
  async retrievalCorpus() {
    const [documents, chunks, media, references] = await Promise.all([this.all(STORES.documents), this.all(STORES.chunks), this.all(STORES.media), this.all(STORES.references)]);
    const active = new Set(documents.filter((doc) => !["deleting", "failed"].includes(doc.status)).map((doc) => doc.id));
    const refMap = new Map();
    for (const ref of references) { const items = refMap.get(ref.chunkId) || []; items.push(ref); refMap.set(ref.chunkId, items); }
    return [
      ...chunks.filter((item) => active.has(item.documentId)).map((item) => ({ ...item, content: item.text ?? item.content ?? "", mediaRefs: refMap.get(item.id) || [] })),
      ...media.filter((item) => active.has(item.documentId) && (item.searchText || item.caption || item.dataText)).map((item) => ({ id: item.id, documentId: item.documentId, page: item.page, modality: item.type, content: item.searchText || item.caption || item.dataText, mediaRefs: [], provenance: { mediaId: item.id, label: item.label, quality: item.quality, source: "media" } })),
    ];
  }
  async exportSafe() {
    const output = { schemaVersion: SCHEMA_VERSION, exportedAt: new Date().toISOString() };
    for (const name of Object.values(STORES)) output[name] = await this.all(name);
    return JSON.parse(JSON.stringify(output, (key, value) => /api.?key|token|secret/i.test(key) ? undefined : value));
  }
  close() { this.db?.close(); this.db = null; }
}
