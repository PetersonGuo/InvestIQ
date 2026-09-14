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
        "cmake --build ../backend/build/native --target stockassist-test-server -j 4 && ../backend/build/native/stockassist-test-server --port 8100",
      url: "http://127.0.0.1:8100/health",
      timeout: 180000,
      env: { STOCKASSIST_DATA_MODE: "ibkr", STOCKASSIST_TEST_DATA: "1", IBKR_PORT: "1", STOCKASSIST_DB: database },
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
