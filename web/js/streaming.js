function extractText(delta) {
  if (typeof delta === "string") return delta;
  if (!Array.isArray(delta)) return "";
  return delta.map((part) => (part && typeof part.text === "string" ? part.text : "")).join("");
}

export async function readOpenAIStream(response, onDelta = () => {}) {
  const contentType = response.headers.get("content-type") || "";
  if (!contentType.toLowerCase().includes("text/event-stream")) {
    const data = await response.json();
    const text = extractText(data?.choices?.[0]?.message?.content);
    if (text) onDelta(text);
    return text;
  }
  if (!response.body) throw new Error("模型响应不支持流式读取");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let answer = "";
  let finished = false;

  const consumeEvent = (eventBlock) => {
    const data = eventBlock.split("\n")
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trimStart()).join("\n").trim();
    if (!data) return;
    if (data === "[DONE]") { finished = true; return; }
    let payload;
    try { payload = JSON.parse(data); }
    catch { throw new Error("模型返回了无法解析的流式事件"); }
    if (payload.error) throw new Error(payload.error.message || payload.error.code || "模型流式请求失败");
    const text = extractText(payload?.choices?.[0]?.delta?.content);
    if (text) { answer += text; onDelta(text); }
  };

  try {
    while (!finished) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      buffer = buffer.replace(/\r\n/g, "\n");
      let boundary = buffer.indexOf("\n\n");
      while (boundary !== -1) {
        consumeEvent(buffer.slice(0, boundary));
        buffer = buffer.slice(boundary + 2);
        if (finished) break;
        boundary = buffer.indexOf("\n\n");
      }
      if (done) break;
    }
    if (!finished && buffer.trim()) consumeEvent(buffer);
    return answer;
  } finally {
    reader.releaseLock();
  }
}
