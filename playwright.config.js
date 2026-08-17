import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./web-tests",
  timeout: 30_000,
  use: { baseURL: "http://127.0.0.1:4173", trace: "retain-on-failure" },
  webServer: { command: "python -m http.server 4173 -d web", url: "http://127.0.0.1:4173", reuseExistingServer: true },
});
