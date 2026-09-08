import { test, expect } from "@playwright/test";
test("company fundamentals and news follow the selected ticker", async ({
	page,
}) => {
	await page.goto("/");
	const section = page.getByRole("region", { name: "AAPL company research" });
	await expect(
		section.getByRole("heading", { name: "Company fundamentals" }),
	).toBeVisible();
	await expect(section.getByText("Revenue", { exact: true })).toBeVisible();
	await expect(section.getByText("Filed 2026-02-15")).toBeVisible();
	await expect(
		section.getByText("AAPL: example company announcement (synthetic)"),
	).toBeVisible();
	await page
		.getByRole("button", { name: "MSFT", exact: true })
		.first()
		.click();
	const next = page.getByRole("region", { name: "MSFT company research" });
	await expect(
		next.getByText("MSFT: example company announcement (synthetic)"),
	).toBeVisible();
	await expect(
		next.getByText("AAPL: example company announcement (synthetic)"),
	).toHaveCount(0);
	await page.setViewportSize({ width: 390, height: 844 });
	expect(
		await page.evaluate(() => document.documentElement.scrollWidth),
	).toBeLessThanOrEqual(390);
});
test("IBKR headlines open article text safely", async ({ page }) => {
	await page.route("**/api/stocks/AAPL/news", (route) =>
		route.fulfill({
			json: {
				ticker: "AAPL",
				source: "ibkr",
				fetched_at: new Date().toISOString(),
				notes: [],
				articles: [
					{
						id: "test",
						title: "Company results",
						publisher: "Test provider",
						provider: "TEST",
						article_id: "one",
						published_at: "2026-09-08T12:00:00Z",
						url: null,
					},
				],
			},
		}),
	);
	await page.route("**/api/news/article?*", (route) =>
		route.fulfill({
			json: {
				text: "<script>window.injected=true</script> Earnings announcement.",
			},
		}),
	);
	await page.goto("/");
	await page
		.getByRole("button", { name: "Read article", exact: true })
		.click();
	await expect(
		page.getByText(
			"<script>window.injected=true</script> Earnings announcement.",
			{ exact: true },
		),
	).toBeVisible();
	expect(await page.evaluate(() => "injected" in window)).toBe(false);
	await page.getByRole("button", { name: "Close article" }).click();
	await expect(
		page.getByText("Earnings announcement.", { exact: true }),
	).toHaveCount(0);
});
