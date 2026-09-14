import { test, expect } from "@playwright/test";

test.beforeEach(async ({page}) => {
  await page.addInitScript(() => localStorage.setItem('pairs_advanced', 'true'));
});

test("factor-adjusted analysis, persistent monitor, schedule, and C++ export", async ({
  page,
}) => {
  test.setTimeout(65000);
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/pairs");
  const section = page.getByRole("region", {
    name: "Exposure-adjusted relative value",
  });
  await expect(section.getByLabel("Relationship family")).toHaveValue(
    "competitor",
  );
  await section.getByLabel("Target security").fill("AAPL");
  await section.getByLabel("Reference basket (ticker:weight)").fill("MSFT:1");
  await section.getByLabel("Sector / broad exposure hedge ETF").fill("NVDA");
  await section.getByLabel("Additional factor ETF proxies").fill("AMZN");
  await section.getByLabel("Residual entry threshold").fill("0.1");
  await section
    .getByRole("button", { name: "Analyze residual", exact: true })
    .click();
  await expect(section.getByRole("table")).toBeVisible();
  await expect(section.getByText(/cumulative residual z-score:/)).toBeVisible();
  await section
    .getByRole("button", { name: "Save C++ backtest strategy" })
    .click();
  await expect(section.getByRole("status")).toContainText("C++ strategy saved");
  const strategies = await (await page.request.get("/api/strategies")).json();
  expect(
    strategies.some(
      (s: { code: string; language: string }) =>
        s.language === "cpp" && s.code.includes("relative_value_math.h"),
    ),
  ).toBeTruthy();
  await section
    .getByRole("button", { name: "Watch residual", exact: true })
    .click();
  await expect(section.locator("summary")).toContainText("Monitoring");
  await section
    .getByRole("button", { name: "Add to scheduled search" })
    .click();
  await section
    .getByRole("button", { name: "Enable relationship schedule" })
    .click();
  await expect(section.getByText(/1 valid models/)).toBeVisible({
    timeout: 20000,
  });
  await page.reload();
  await expect(section.locator("summary")).toHaveCount(1);
  await section.locator("summary").click();
  await section.getByRole("button", { name: "Pause residual monitor" }).click();
  await expect(section.locator("summary")).toContainText("Paused");
  await section
    .getByRole("button", { name: "Pause relationship schedule" })
    .click();
  await expect(
    section.getByRole("button", { name: "Scan relationships now" }),
  ).toBeDisabled();
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(390);
  await section
    .getByRole("button", { name: "Delete relationship schedule" })
    .click();
  await section
    .getByRole("button", { name: "Delete residual monitor" })
    .click();
  await expect(section.getByText("No residual monitors saved.")).toBeVisible();
  expect(errors).toEqual([]);
});

test('C++ relative-value starters include the reference and factor hedges', async ({page}) => {
  await page.goto('/research');
  await page.getByRole('button',{name:'C++',exact:true}).click();
  await page.getByLabel('Starter strategy').selectOption('rv_basket');
  await page.getByRole('button',{name:'Load preset',exact:true}).click();
  await expect(page.getByLabel('Strategy name')).toHaveValue('Stock versus related basket');
  await expect(page.getByLabel('Strategy code')).toContainText('relative_value_math.h');
  await expect(page.getByLabel('Warmup bars')).toHaveValue('122');
  await expect(page.getByLabel('Strategy parameters')).toContainText('"reference_count": 5');
  await page.getByRole('button',{name:'Python',exact:true}).click();
  await expect(page.getByLabel('Starter strategy')).toHaveValue('moving_average');
});
