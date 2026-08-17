import { test, expect } from "@playwright/test";

for (const viewport of [{ width: 375, height: 812 }, { width: 768, height: 1024 }, { width: 1440, height: 900 }]) {
  test(`workspace renders at ${viewport.width}px`, async ({ page }) => {
    await page.setViewportSize(viewport);
    await page.goto("/");
    await expect(page).toHaveTitle(/Multimodal RAG/);
    await expect(page.getByRole("heading", { name: "对话" })).toBeVisible();
    await expect(page.locator("html")).toHaveAttribute("data-storage", /indexeddb|legacy/);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth)).toBe(true);
  });
}
