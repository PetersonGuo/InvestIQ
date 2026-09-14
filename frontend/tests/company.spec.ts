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

 test("IBKR headline links can be opened directly and reloaded", async ({page}) => {
   await page.route("**/api/stocks/AAPL/news", route => route.fulfill({json: {ticker: "AAPL", source: "ibkr", fetched_at: new Date().toISOString(), notes: [], articles: [{id: "T:1", title: "Linked results", publisher: "Provider", provider: "T", article_id: "one&two", published_at: "2026-09-09", url: null}]}}));
   await page.route("**/api/news/article?*", route => {
     expect(new URL(route.request().url()).searchParams.get("article")).toBe("one&two");
     return route.fulfill({json: {text: "<script>window.injected=true</script> Full article."}});
   });
   await page.goto("/");
   const link = page.getByRole("link", {name: "Linked results"});
   await expect(link).toHaveAttribute("href", /\/news\/article\?/);
   await link.click();
   await expect(page.getByRole("heading", {name: "Linked results"})).toBeVisible();
   await expect(page.getByText("<script>window.injected=true</script> Full article.", {exact: true})).toBeVisible();
   expect(await page.evaluate(() => "injected" in window)).toBe(false);
   await page.reload();
   await expect(page.getByText("<script>window.injected=true</script> Full article.", {exact: true})).toBeVisible();
 });

test("formatted articles preserve structure safely in both readers", async ({ page }) => {
  const html = `<html><head><style>body{display:none}</style></head><body>
    <h2>Quarterly results &amp; outlook</h2><p>First paragraph with <strong>bold earnings</strong>.</p>
    <p>Second paragraph&#8212;growth.<br>Next line.</p><ul><li>Revenue rose</li><li>Margins improved</li></ul>
    <table><tr><th>Period</th><th>Revenue</th></tr><tr><td>Q2</td><td>$100</td></tr></table>
    <a href="https://example.com/report?x=1&amp;y=2" onclick="window.injected=true">Source filing</a>
    <a href="javascript:window.injected=true">Unsafe link</a><script>window.injected=true</script>
    <img src="https://example.com/tracker" onerror="window.injected=true"><iframe src="https://example.com"></iframe>
    </body></html>`;
  await page.route("**/api/stocks/AAPL/news", route => route.fulfill({json: {
    ticker: "AAPL", source: "ibkr", fetched_at: new Date().toISOString(), notes: [],
    articles: [{id: "T:formatted", title: "Formatted results", publisher: "Provider", provider: "T", article_id: "formatted", published_at: "2026-09-09", url: null}]
  }}));
  await page.route("**/api/news/article?*", route => route.fulfill({json: {text: "Fallback text", html}}));
  await page.goto("/");
  await page.getByRole("button", {name: "Read article", exact: true}).click();
  const body = page.locator("[data-article-body]");
  async function verify() {
    await expect(body.getByRole("heading", {name: "Quarterly results & outlook"})).toBeVisible();
    await expect(body.locator("p")).toHaveCount(2);
    await expect(body.locator("strong")).toHaveText("bold earnings");
    await expect(body.getByRole("listitem")).toHaveCount(2);
    await expect(body.getByRole("cell", {name: "$100", exact: true})).toBeVisible();
    await expect(body.getByRole("link", {name: "Source filing"})).toHaveAttribute("href", "https://example.com/report?x=1&y=2");
    await expect(body.getByRole("link", {name: "Unsafe link"})).toHaveCount(0);
    await expect(body.locator("script,style,img,iframe,[onclick]")).toHaveCount(0);
    expect(await page.evaluate(() => "injected" in window)).toBe(false);
  }
  await verify();
  await page.getByRole("link", {name: "Formatted results"}).click();
  await verify();
  await page.setViewportSize({width: 390, height: 844});
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
});
