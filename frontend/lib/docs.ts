import { readFile } from "node:fs/promises";
import path from "node:path";

export const guides = {
  start: "Getting started",
  strategies: "Write a strategy",
  api: "HTTP API reference",
  troubleshooting: "Troubleshooting",
};
export type Guide = keyof typeof guides;
export async function readGuide(guide: Guide) {
  return readFile(path.join(process.cwd(), "public", "docs", `${guide}.md`), "utf8");
}
