import { defineConfig } from "@playwright/test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
const database = path.join(
  mkdtempSync(path.join(tmpdir(), "stockassist-e2e-")),
  "test.sqlite3",
);
export default defineConfig({
  testDir: "./tests",
  fullyParallel: false,
  workers: 1,
  use: { baseURL: "http://127.0.0.1:3100", trace: "retain-on-failure" },
  webServer: [
    {
      command:
        "../backend/build/native/stockassist-server --port 8100",
      url: "http://127.0.0.1:8100/health",
      env: { STOCKASSIST_DATA_MODE: "demo", STOCKASSIST_DB: database },
      reuseExistingServer: false,
    },
    {
      command: "npm run start -- --hostname 127.0.0.1 --port 3100",
      url: "http://127.0.0.1:3100",
      env: { STOCKASSIST_API_URL: "http://127.0.0.1:8100" },
      reuseExistingServer: false,
    },
  ],
});
