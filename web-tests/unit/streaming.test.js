import test from "node:test";
import assert from "node:assert/strict";
import { readOpenAIStream } from "../../web/js/streaming.js";

function responseFrom(chunks, contentType = "text/event-stream") {
  const encoder = new TextEncoder();
  return new Response(new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  }), { headers: { "content-type": contentType } });
}

test("parses SSE split across network chunks", async () => {
  const deltas = [];
  const response = responseFrom([
    'data: {"choices":[{"delta":{"content":"你"}}]}\r',
    '\n\r\ndata: {"choices":[{"delta":{"content":"好"}}]}\n\n',
    'data: [DONE]\n\n',
  ]);
  assert.equal(await readOpenAIStream(response, (value) => deltas.push(value)), "你好");
  assert.deepEqual(deltas, ["你", "好"]);
});

test("falls back to a regular OpenAI JSON response", async () => {
  const response = new Response(JSON.stringify({ choices: [{ message: { content: "done" } }] }), {
    headers: { "content-type": "application/json" },
  });
  assert.equal(await readOpenAIStream(response), "done");
});
