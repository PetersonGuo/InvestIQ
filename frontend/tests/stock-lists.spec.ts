import { test, expect } from "@playwright/test";

test("create a list, evaluate strategies, review clear actions and fundamentals, backtest", async ({
  page,
}) => {
  test.setTimeout(65000);
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/pairs");
  await expect(
    page.getByRole("button", { name: "Advanced strategy setup" }),
  ).toHaveAttribute("aria-expanded", "false");
  await expect(
    page.getByRole("region", { name: "Exposure-adjusted relative value" }),
  ).toHaveCount(0);
  const main = page.getByRole("region", {
    name: "Automated stock list strategies",
  });
  await main.getByLabel("List name", { exact: true }).fill("My stock research");
  await main
    .getByLabel("Stocks to compare")
    .fill("AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA");
  await main
    .getByText("Optional scan and risk settings", { exact: true })
    .click();
  await main.getByLabel("List correlation filter").fill("0");
  await main.getByLabel("Entry score magnitude").fill("0.1");
  await main.getByLabel("Maximum generated monitors").fill("5");
  await main
    .getByRole("button", { name: "Save list & start automatic checks" })
    .click();
  await expect(main.getByText(/models evaluated/)).toBeVisible({
    timeout: 25000,
  });
  await expect(main.getByRole("article").first()).toBeVisible();
  const entry = main
    .getByRole("article")
    .filter({ hasText: "Entry candidate — review all legs" })
    .first();
  await expect(entry).toBeVisible();
  await expect(
    entry.getByRole("heading", { name: "BUY / LONG", exact: true }),
  ).toBeVisible();
  await expect(
    entry.getByRole("heading", { name: "SELL SHORT", exact: true }),
  ).toBeVisible();
  await expect(entry.getByText(/factor hedge/).first()).toBeVisible();
  await entry
    .getByRole("button", { name: "Why this signal? Fundamentals & backtest" })
    .click();
  await expect(
    entry.getByRole("heading", { name: "Why this setup?" }),
  ).toBeVisible();
  await expect(
    entry.getByText("Revenue", { exact: true }).first(),
  ).toBeVisible();
  await entry
    .getByRole("button", { name: "Backtest this setup", exact: true })
    .click();
  await expect(entry.getByRole("status")).toHaveText("Backtest: completed", {
    timeout: 25000,
  });
  await expect(entry.getByText(/Backtest cost filter rejected/)).toBeVisible();
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(390);
  const listId = await main
    .getByLabel("Stock list", { exact: true })
    .inputValue();
  await page.reload();
  await main.getByLabel("Stock list", { exact: true }).selectOption(listId);
  await expect(main.getByText(/models evaluated/)).toBeVisible();
  await main.getByRole("button", { name: "Delete list", exact: true }).click();
  await expect(main.getByRole("status")).toContainText("Its scans stopped");
  const listRows = await (
    await page.request.get("/api/pairs/stock-lists")
  ).json();
  expect(listRows).toEqual([]);
  const rules = await (
    await page.request.get("/api/pairs/opportunities")
  ).json();
  for (const row of rules)
    if (row.discovery?.list_name === "My stock research") {
      expect(row.action.state).toBe("paused");
      expect(row.action.legs).toEqual([]);
      await page.request.delete(`/api/pairs/alerts/${row.id}`);
    }
  expect(errors).toEqual([]);
});

test("waiting signals never display a buy or short action", async ({
  page,
  request,
}) => {
  const model = {
    category: "competitor",
    target: "AAPL",
    references: [{ ticker: "MSFT", weight: 1 }],
    market: "SPY",
    sector: "XLK",
    factors: ["MTUM", "VLUE"],
    lookback: 120,
    entry_z: 20,
  };
  const response = await request.post("/api/pairs/alerts", {
    data: { relative_value: model },
  });
  expect(response.ok()).toBeTruthy();
  const saved = await response.json();
  await page.goto("/pairs");
  const card = page.getByRole("article", { name: "AAPL Residual pair result" });
  await expect(
    card.getByText("Wait — no entry signal", { exact: true }),
  ).toBeVisible();
  await expect(
    card.getByRole("heading", { name: "BUY / LONG", exact: true }),
  ).toHaveCount(0);
  await expect(
    card.getByRole("heading", { name: "SELL SHORT", exact: true }),
  ).toHaveCount(0);
  await card.getByRole("button", { name: "Pause this monitor" }).click();
  await expect(
    card.getByText("Paused — no action", { exact: true }),
  ).toBeVisible();
  await request.delete(`/api/pairs/alerts/${saved.id}`);
});
