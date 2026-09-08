import { test, expect } from "@playwright/test";

test("live quote updates chart without replacing it and supports IOC paper limits", async ({
	page,
}) => {
	await page.goto("/");
	await expect(page.getByText(/Demo quotes ·/)).toBeVisible();
	await expect(
		page.getByRole("button", { name: "Follow live", exact: true }),
	).toBeVisible();
	const chart = page.getByRole("img", { name: /AAPL daily candlestick/ });
	const canvas = chart.locator("canvas").first();
	await canvas.evaluate((el) => el.setAttribute("data-preserved", "yes"));
	await page.waitForTimeout(2200);
	await expect(canvas).toHaveAttribute("data-preserved", "yes");
	await page.getByLabel("Order type").selectOption("limit");
	await page.getByLabel("Limit price", { exact: true }).fill("0.01");
	await page
		.getByRole("button", { name: "Buy AAPL · Paper", exact: true })
		.click();
	await expect(
		page.getByRole("alert").filter({ hasText: "Limit does not cross" }),
	).toContainText("cancelled without a fill");
	await page.getByLabel("Order type").selectOption("market");
	await page.getByLabel("Shares of AAPL").fill("200");
	await page
		.getByRole("button", { name: "Buy AAPL · Paper", exact: true })
		.click();
	await expect(page.getByRole("status")).toContainText(
		"100 unfilled shares cancelled",
	);
	await expect(
		page.getByRole("columnheader", { name: "Unrealized P&L" }),
	).toBeVisible();
});
