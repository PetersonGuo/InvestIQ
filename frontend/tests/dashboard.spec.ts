import { test, expect } from "@playwright/test";

test("search, paper fills, alerts, and persistence through the UI", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Market overview" }),
  ).toBeVisible();
  await expect(
    page.getByText("SYNTHETIC DEMO DATA", { exact: false }),
  ).toBeVisible();
  await page.getByLabel("Search stocks").fill("Microsoft");
  await page
    .getByRole("button", { name: "MSFT Microsoft Corporation" })
    .click();
  await expect(
    page.getByRole("img", { name: /MSFT daily candlestick/ }),
  ).toBeVisible();
  await page.getByLabel("Shares of MSFT").fill("2");
  await page.getByRole("button", { name: "Buy MSFT · Paper" }).click();
  await expect(page.getByRole("status")).toContainText("Paper buy filled");
  await expect(
    page.getByText("2 shares held.", { exact: false }),
  ).toBeVisible();
  await page.reload();
  await page.getByRole("button", { name: "MSFT", exact: true }).first().click();
  await expect(
    page.getByText("2 shares held.", { exact: false }),
  ).toBeVisible();
  await page.getByRole("button", { name: "sell", exact: true }).click();
  await page.getByLabel("Shares of MSFT").fill("3");
  await page.getByRole("button", { name: "Sell MSFT · Paper" }).click();
  await expect(
    page
      .getByRole("alert")
      .filter({ hasText: /Insufficient shares|Start the Python backend/ }),
  ).toContainText("Insufficient shares");
  await page.getByLabel("Shares of MSFT").fill("2");
  await page.getByRole("button", { name: "Sell MSFT · Paper" }).click();
  await expect(page.getByRole("status")).toContainText("Paper sell filled");
  await page.getByLabel("Target price · USD").fill("10000");
  await page.getByRole("button", { name: "Create alert", exact: true }).click();
  await expect(page.getByText("Watching", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Edit", exact: true }).click();
  await page.getByLabel("Target price · USD").fill("9999");
  await page.getByRole("button", { name: "Save alert", exact: true }).click();
  await expect(page.getByText("At or above $9,999.00")).toBeVisible();
  await page.reload();
  await expect(page.getByText("At or above $9,999.00")).toBeVisible();
  await page.getByRole("button", { name: "Delete MSFT alert" }).click();
  await expect(
    page.getByText("Your levels, on watch.", { exact: false }),
  ).toBeVisible();
  expect(errors).toEqual([]);
});

test("mobile layout stays within the viewport", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/stock");
  await expect(
    page.getByRole("img", { name: /AAPL daily candlestick/ }),
  ).toBeVisible();
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(390);
  await page.screenshot({
    path: "test-results/dashboard-mobile.png",
    fullPage: true,
  });
});

test("backend outage has an actionable message", async ({ page }) => {
  await page.route("**/api/**", (route) =>
    route.fulfill({
      status: 503,
      contentType: "application/json",
      body: JSON.stringify({
        detail:
          "StockAssist API is unavailable. Start the Python backend and retry.",
      }),
    }),
  );
  await page.goto("/");
  await expect(
    page
      .getByRole("alert")
      .filter({ hasText: /Insufficient shares|Start the Python backend/ }),
  ).toContainText("Start the Python backend");
  await expect(
    page.getByRole("button", { name: "Buy AAPL · Paper" }),
  ).toBeDisabled();
});

test('background worker triggers a saved alert', async ({ page }) => {
  test.setTimeout(60000);
  await page.goto('/');
  await expect(page.getByRole('img', { name: /AAPL daily candlestick/ })).toBeVisible();
  await page.getByLabel('Target price · USD').fill('1');
  await page.getByRole('button', { name: 'Create alert', exact: true }).click();
  await expect(page.getByText('Triggered', { exact: true })).toBeVisible({ timeout: 45000 });
  await page.screenshot({ path: 'test-results/dashboard-desktop.png', fullPage: true });
  await page.getByRole('button', { name: 'Delete AAPL alert' }).click();
});

test('proxy rejects writes from other origins', async ({ request }) => {
  const response = await request.post('/api/order', {
    headers: { Origin: 'https://unrelated.example' },
    data: { ticker: 'AAPL', side: 'buy', quantity: 1 },
  });
  expect(response.status()).toBe(403);
});
