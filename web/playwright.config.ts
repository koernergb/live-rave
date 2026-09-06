import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "e2e",
  timeout: 180_000,
  fullyParallel: false,
  workers: 1,
  use: {
    browserName: "chromium",
  },
  webServer: {
    command: "npx vite preview --port 4175 --strictPort",
    url: "http://localhost:4175",
    reuseExistingServer: true,
    timeout: 30_000,
  },
});