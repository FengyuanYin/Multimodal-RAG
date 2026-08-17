import test from "node:test";
import assert from "node:assert/strict";
import "fake-indexeddb/auto";
import { BrowserKnowledgeRepository } from "../../web/js/storage.js";

const bundle = {
  document: { id: "doc", fingerprint: "fp", name: "report", status: "ready" },
  chunks: [{ id: "chunk", text: "图1", page: 1 }],
  media: [{ id: "media", type: "image", page: 1, label: "图1" }],
  references: [{ id: "ref", chunkId: "chunk", mediaId: "media", page: 1 }],
};

test("IndexedDB bundle upsert is idempotent and cascade deletion removes children", async () => {
  const repo = new BrowserKnowledgeRepository(`test-${crypto.randomUUID()}`);
  assert.equal(await repo.upsertDocument(bundle), "doc");
  assert.equal(await repo.upsertDocument({ ...bundle, document: { ...bundle.document, id: "generated-again" } }), "doc");
  assert.equal((await repo.all("documents")).length, 1);
  await repo.deleteDocument("doc");
  assert.equal((await repo.all("chunks")).length, 0);
  assert.equal((await repo.all("media")).length, 0);
  assert.equal((await repo.all("references")).length, 0);
  repo.close();
});

test("injected ingestion failure aborts the complete IndexedDB transaction", async () => {
  const repo = new BrowserKnowledgeRepository(`test-${crypto.randomUUID()}`);
  await assert.rejects(repo.upsertDocument(bundle, () => { throw new Error("injected"); }), /injected/);
  assert.equal((await repo.all("documents")).length, 0);
  assert.equal((await repo.all("chunks")).length, 0);
  repo.close();
});
