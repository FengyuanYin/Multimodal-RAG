import { test, expect } from "@playwright/test";

test("skip link and navigation are keyboard reachable", async ({ page }) => {
  test.setTimeout(10_000);
  await page.goto("/", { waitUntil: "domcontentloaded", timeout: 10_000 });
  await page.keyboard.press("Tab");
  expect(await page.evaluate(() => document.activeElement?.classList.contains("skip-link"))).toBe(true);
  expect(await page.getAttribute(".skip-link", "href")).toBe("#workspaceMain");
  const unnamed = await page.evaluate(() => [...document.querySelectorAll("button")].filter((button) => button.querySelector("svg") && !(button.getAttribute("aria-label") || button.getAttribute("title") || button.textContent.trim())).length);
  expect(unnamed).toBe(0);
});
