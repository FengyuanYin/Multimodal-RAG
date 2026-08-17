import test from "node:test";
import assert from "node:assert/strict";
import { TaskState } from "../../web/js/state.js";
import { redact, validateHttpUrl } from "../../web/js/security.js";

test("task state rejects illegal transitions", () => { const task = new TaskState("parse"); assert.throws(() => task.transition("succeeded")); task.transition("running"); task.transition("degraded", { warning: "vector unavailable" }); task.transition("succeeded"); assert.equal(task.state, "succeeded"); });
test("URL validation rejects credentials and unsafe schemes", () => { assert.throws(() => validateHttpUrl("javascript:alert(1)")); assert.throws(() => validateHttpUrl("https://user:pass@example.com")); });
test("redaction removes common secrets", () => { const output = redact("https://x.test?a=1&api_key=sk-secret123456 Authorization: Bearer sk-another123456"); assert.ok(!output.includes("secret123456")); assert.ok(!output.includes("another123456")); });
