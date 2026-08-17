import test from "node:test";
import assert from "node:assert/strict";
import { detectMediaReferences, resolveMediaReference, stableId } from "../../web/js/media-association.js";

test("stable IDs are deterministic and page-sensitive", async () => {
  assert.equal(await stableId("chunk", "doc", 1, "text"), await stableId("chunk", "doc", 1, "text"));
  assert.notEqual(await stableId("chunk", "doc", 1, "text"), await stableId("chunk", "doc", 2, "text"));
});
test("detects Chinese and English figure/table labels", () => {
  const refs = detectMediaReferences("图 1，Figure 2，Fig. 3，表格4，Table 5", "doc", 2);
  assert.deepEqual(refs.map((item) => [item.mediaType, item.label]), [["image", "图1"], ["image", "图2"], ["image", "图3"], ["table", "表4"], ["table", "表5"]]);
});
test("ambiguous assets remain unresolved", () => {
  const ref = detectMediaReferences("图1", "doc", 1)[0];
  const base = { documentId: "doc", type: "image", page: 1, label: "图1" };
  assert.equal(resolveMediaReference(ref, [{ ...base, id: "a" }, { ...base, id: "b" }]).resolution, "unresolved");
});
