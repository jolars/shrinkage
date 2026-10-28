import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "tests",
  testMatch: "*.spec.mjs",
  outputDir: "artifacts/results",
  fullyParallel: true,
  workers: 2,
  use: {
    baseURL: "http://127.0.0.1:8792",
    browserName: "chromium",
    channel: "chromium",
    reducedMotion: "reduce",
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
  projects: [
    { name: "desktop", use: { viewport: { width: 1280, height: 1000 } } },
    {
      name: "mobile",
      use: {
        viewport: { width: 390, height: 844 },
        isMobile: true,
        hasTouch: true,
      },
    },
  ],
  webServer: {
    command: "node tests/serve.mjs",
    url: "http://127.0.0.1:8792",
    reuseExistingServer: false,
    timeout: 10000,
  },
});
