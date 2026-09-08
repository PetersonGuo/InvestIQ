import { test, expect } from "@playwright/test";
test("portfolio starter runs with SPY risk comparison and restores its symbols", async ({
	page,
}) => {
	test.setTimeout(60000);
	await page.goto("/research");
	await page.getByLabel("Starter strategy").selectOption("equal_weight");
	await page
		.getByRole("button", { name: "Load preset", exact: true })
		.click();
	await expect(
		page.getByLabel("Additional stocks (comma separated, up to 9)"),
	).toHaveValue("MSFT, NVDA");
	await page.getByRole("button", { name: "C++", exact: true }).click();
	await expect(page.getByLabel("Strategy code")).toContainText(
		"on_portfolio",
	);
	await page
		.getByRole("button", { name: "Run backtest", exact: true })
		.click();
	await expect(
		page.getByRole("heading", { name: "Risk and benchmark comparison" }),
	).toBeVisible({ timeout: 30000 });
	await expect(
		page.getByRole("columnheader", { name: "S&P 500 · SPY buy & hold" }),
	).toBeVisible();
	await expect(
		page.getByRole("cell", {
			name: "Average gross exposure %",
			exact: true,
		}),
	).toBeVisible();
	await page
		.getByRole("button", { name: "Load inputs", exact: true })
		.click();
	await expect(
		page.getByLabel("Additional stocks (comma separated, up to 9)"),
	).toHaveValue("MSFT, NVDA");
	await page.setViewportSize({ width: 390, height: 844 });
	expect(
		await page.evaluate(() => document.documentElement.scrollWidth),
	).toBeLessThanOrEqual(390);
});
test("cointegration results and inverse relationship controls are available", async ({
	page,
}) => {
	await page.goto("/pairs");
	await page
		.getByLabel("Relationship", { exact: false })
		.selectOption("negative");
	await page
		.getByRole("button", { name: "Preview pair", exact: true })
		.click();
	await expect(
		page.getByText("Cointegration:", { exact: false }).last(),
	).toBeVisible();
	await page
		.getByRole("button", { name: "Reverse pair order", exact: true })
		.click();
	await expect(page.getByLabel("Stock A", { exact: true })).toHaveValue(
		"MSFT",
	);
	await expect(page.getByLabel("Stock B", { exact: true })).toHaveValue(
		"AAPL",
	);
});
