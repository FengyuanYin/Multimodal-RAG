import { chromium } from "playwright";

const browser = await chromium.launch({
  headless: true,
  executablePath: process.env.MMRAG_BROWSER_PATH,
});
const errors = [];
for (const [width, height] of [[375, 812], [768, 1024], [844, 390], [1024, 768], [1440, 900], [1920, 1080], [2560, 1080]]) {
  const page = await browser.newPage({ viewport: { width, height }, reducedMotion: "reduce" });
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("http://127.0.0.1:4174/", { waitUntil: "networkidle" });
  await page.screenshot({ path: `reports/screenshots/workspace-${width}.png`, fullPage: true });
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth);
  const storage = await page.getAttribute("html", "data-storage");
  await page.keyboard.press("Tab");
  const skipLinkFocused = await page.evaluate(() => document.activeElement?.classList.contains("skip-link"));
  const unnamedIconButtons = await page.evaluate(() => [...document.querySelectorAll("button")].filter((button) => button.querySelector("svg") && !(button.getAttribute("aria-label") || button.getAttribute("title") || button.textContent.trim())).length);
  const layout = await page.evaluate(() => {
    const composer = document.querySelector('.chat-form')?.getBoundingClientRect();
    const nav = document.querySelector('.sidebar')?.getBoundingClientRect();
    const content = document.querySelector('.content')?.getBoundingClientRect();
    return { composerBottom: Math.round(composer?.bottom || 0), navTop: Math.round(nav?.top || 0), contentRight: Math.round(content?.right || 0), rightGap: Math.round(innerWidth - (content?.right || 0)), viewport: innerHeight };
  });
  console.log(JSON.stringify({ width, overflow, storage, skipLinkFocused, unnamedIconButtons, layout, title: await page.title() }));
  await page.close();
}
await browser.close();
console.log(JSON.stringify({ errors }));
process.exit(errors.length ? 1 : 0);
