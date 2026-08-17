import { chromium } from "playwright";

const browser = await chromium.launch({ headless: true, executablePath: process.env.MMRAG_BROWSER_PATH });
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
const errors = [];
let sawStream = false;
page.on("pageerror", (error) => errors.push(error.message));
await page.addInitScript(() => {
  localStorage.setItem("pdfchat.settings.v1", JSON.stringify({
    callMode: "direct", baseUrl: "https://api.test/v1", model: "test-model",
    retrievalMode: "keyword", topK: 5, chunkSize: 800,
  }));
  sessionStorage.setItem("mmrag.secret.apiKey", "test-key");
});
await page.route("https://api.test/v1/chat/completions", async (route) => {
  const request = route.request();
  const payload = request.postDataJSON();
  if (payload.stream !== true) {
    return route.fulfill({ status: 200, contentType: "application/json", body: '{"choices":[{"message":{"content":"[]"}}]}' });
  }
  sawStream = true;
  await route.fulfill({
    status: 200,
    headers: { "content-type": "text/event-stream" },
    body: 'data: {"choices":[{"delta":{"content":"流式"}}]}\n\ndata: {"choices":[{"delta":{"content":"完成"}}]}\n\ndata: [DONE]\n\n',
  });
});
await page.goto("http://127.0.0.1:4173/", { waitUntil: "networkidle" });
await page.locator("#question").fill("你好");
await page.locator("#chatForm").evaluate((form) => form.requestSubmit());
await page.locator(".msg.assistant").filter({ hasText: "流式完成" }).waitFor();
const status = await page.locator("#streamStatus").textContent();
const stopHidden = await page.locator("#stopBtn").evaluate((button) => button.classList.contains("hidden"));
console.log(JSON.stringify({ status: status.trim(), stopHidden, sawStream, errors }));
await browser.close();
if (errors.length || !stopHidden || !sawStream || !status.includes("已就绪")) process.exit(1);
