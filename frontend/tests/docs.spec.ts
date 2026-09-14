import { test, expect } from "@playwright/test";

test("developer guides are reachable, navigable and readable on mobile", async ({page, context, request}) => {
  await page.goto("/");
  await page.getByRole("link", {name: "API docs", exact: true}).click();
  await expect(page.getByRole("heading", {name: "Build with StockAssist", exact: true})).toBeVisible();
  await expect(page.getByRole("table").first()).toBeVisible();
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await page.getByRole("button", {name: "Copy sh example"}).first().click();
  expect(await page.evaluate(()=>navigator.clipboard.readText())).toContain("npm run setup");
  await page.getByRole("navigation", {name: "Developer guides"}).getByRole("link", {name:"Write a strategy",exact:true}).click();
  await expect(page.getByRole("heading", {name: "Single-stock Python",exact:true})).toBeVisible();
  await expect(page.getByRole("heading", {name: "Multi-stock C++",exact:true})).toBeVisible();
  await expect(page.getByText("First callback occurs after warmup", {exact:false})).toBeVisible();
  const raw=await request.get("/docs/strategies.md");
  expect(raw.status()).toBe(200);
  expect(await raw.text()).toContain('extern "C" int on_portfolio');
  await page.getByRole("navigation", {name: "Developer guides"}).getByRole("link", {name:"HTTP API reference",exact:true}).click();
  await page.getByText("On this page", {exact:true}).click();
  await page.getByRole("navigation", {name:"On this page"}).getByRole("link", {name:"Submit and poll a backtest"}).click();
  await expect(page).toHaveURL(/#submit-and-poll-a-backtest$/);
  await page.setViewportSize({width:390,height:844});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await page.screenshot({path:"test-results/developer-docs-mobile.png",fullPage:false});
  expect((await request.get('/docs/not-a-guide')).status()).toBe(404);
});

test("strategy editor links directly to the callback guide", async ({page}) => {
  await page.goto('/research');
  await page.getByRole('link', {name:'Developer docs: write your own Python or C++ strategy →'}).click();
  await expect(page.getByRole('heading',{name:'Write a strategy',exact:true})).toBeVisible();
});
