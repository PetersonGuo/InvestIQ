import { test, expect } from "@playwright/test";
test("scheduled peer valuation explains drivers and filings and suppresses unconfirmed trades", async ({
  page,
  request,
}) => {
  const list = await (
    await request.post("/api/pairs/stock-lists", {
      data: { name: "Peer valuation test", tickers: ["AAPL", "MSFT"] },
    })
  ).json();
  await page.goto("/pairs");
  const main = page.getByRole("region", {
    name: "Automated stock list strategies",
  });
  await main.getByLabel("Stock list", { exact: true }).selectOption(list.id);
  await main
    .getByRole("button", { name: "Choose strategies & enable checks" })
    .click();
  for (const name of [
    "Residual pairs",
    "Stock versus basket",
    "Known share classes",
    "Known ETF tracking pairs",
  ])
    await main.getByRole("checkbox", { name, exact: true }).uncheck();
  await main
    .getByRole("checkbox", {
      name: "Peer fundamental valuation (11+ eligible companies)",
      exact: true,
    })
    .check();
  await main
    .getByLabel("Additional valuation reference peers (optional)", {
      exact: false,
    })
    .fill("AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA, XLK, MTUM, VLUE, SPY");
  await main.getByRole("button", { name: /Save list/ }).click();
  const panel = page.getByRole("region", {
    name: "Peer relative valuation",
    exact: true,
  });
  const card = panel.getByRole("article", {
    name: "AAPL valuation",
    exact: true,
  });
  await expect(card.getByText(/Actual P\/E/)).toBeVisible({ timeout: 20000 });
  await expect(card.getByText(/sector ETF/)).toBeVisible();
  await expect(card.getByRole("heading", { name: "BUY / LONG" })).toHaveCount(
    0,
  );
  await card
    .getByText("Why this valuation? Peers, drivers & filings", { exact: true })
    .click();
  await expect(
    card.getByRole("cell", { name: "Revenue growth", exact: true }),
  ).toBeVisible();
  await expect(
    card.getByRole("link", { name: "Open filing" }).first(),
  ).toHaveAttribute("href", /^https:\/\/www.sec.gov\/Archives\//);
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(390);
  await page.reload();
  await main.getByLabel("Stock list", { exact: true }).selectOption(list.id);
  await expect(card.getByText(/Actual P\/E/)).toBeVisible();
  await main
    .getByRole("button", { name: "Pause list discovery", exact: true })
    .click();
  await expect(
    card.getByText("No action — previous estimate only", { exact: true }),
  ).toBeVisible();
  await request.delete("/api/pairs/stock-lists/" + list.id);
});
