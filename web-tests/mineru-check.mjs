import { chromium } from "playwright";

const browser = await chromium.launch({ headless: true, executablePath: process.env.MMRAG_BROWSER_PATH });
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
const errors = [];
let requestVerified = false;
page.on("pageerror", (error) => errors.push(error.message));
await page.route("https://proxy.example/proxy/mineru/parse", async (route) => {
  const request = route.request();
  const headers = request.headers();
  requestVerified = request.method() === "POST"
    && headers["x-api-key"] === "test-key"
    && decodeURIComponent(headers["x-file-name"]) === "fixture.pdf"
    && (request.postDataBuffer()?.length || 0) > 8;
  await route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify({ pages: [{ page: 1, text: "MinerU 解析成功" }], media: [] }),
  });
});
await page.goto("http://127.0.0.1:4174/", { waitUntil: "networkidle" });
await page.locator('[data-view="settings"]').click();
await page.locator("#parser").selectOption("mineru");
await page.locator("#mineruApiKey").fill("test-key");
await page.locator("#mineruProxyUrl").fill("https://proxy.example");
await page.locator("#saveSettings").click();
await page.locator('[data-view="docs"]').click();
await page.locator("#fileInput").setInputFiles({
  name: "fixture.pdf",
  mimeType: "application/pdf",
  buffer: Buffer.from("%PDF-1.4\n%%EOF"),
});
await page.getByText(/已添加 fixture\.pdf/).waitFor({ state: "attached" });
console.log(JSON.stringify({ requestVerified, errors }));
await browser.close();
if (!requestVerified || errors.length) process.exit(1);
