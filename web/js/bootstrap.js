import { BrowserKnowledgeRepository } from "./storage.js";
import { migrateLegacy } from "./migration.js";

export async function initializeIndustrialWorkspace() {
  const repository = new BrowserKnowledgeRepository();
  try {
    await repository.open();
    const migration = await migrateLegacy(repository);
    document.documentElement.dataset.storage = "indexeddb";
    window.MultimodalRAG = Object.freeze({ repository, migration, mode: "github-pages" });
    window.dispatchEvent(new CustomEvent("mmrag:ready", { detail: { migration } }));
  } catch (error) {
    document.documentElement.dataset.storage = "legacy";
    console.warn("IndexedDB 初始化失败，保留旧版存储。", error);
  }
}
