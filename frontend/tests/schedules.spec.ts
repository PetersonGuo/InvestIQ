import { test, expect } from "@playwright/test";

test.beforeEach(async ({page}) => {
  await page.addInitScript(() => localStorage.setItem('pairs_advanced', 'true'));
});

test("saved discovery runs automatically and survives reload with controls", async ({ page }) => {
  await page.goto('/pairs');
  const section = page.getByRole('region', { name: 'Scheduled pair discovery' });
  await section.getByLabel('Scheduled stock source').selectOption('symbols');
  await section.getByLabel('Scheduled stock symbols').fill('NVDA, META, AMZN');
  await section.getByLabel('Scheduled minimum correlation').fill('0');
  await section.getByLabel('Scheduled relationship').selectOption('either');
  await section.getByLabel('Scheduled cointegration evidence at 5%').uncheck();
  await section.getByLabel('Maximum managed alerts').fill('1');
  await section.getByRole('button', { name: 'Enable automatic discovery' }).click();
  await expect(section.getByText(/1 alerts added last run/)).toBeVisible({ timeout: 20000 });
  await page.reload();
  await expect(section.getByText(/1\/1 managed at last run/)).toBeVisible();
  await section.getByRole('button', { name: 'Run now' }).click();
  await expect(section.getByText(/0 alerts added last run/)).toBeVisible({ timeout: 20000 });
  await section.getByRole('button', { name: 'Pause schedule' }).click();
  await expect(section.getByRole('button', { name: 'Run now' })).toBeDisabled();
  await section.getByRole('button', { name: 'Edit schedule' }).click();
  await section.getByLabel('Repeat every (hours)').fill('6');
  await section.getByRole('button', { name: 'Save schedule', exact: true }).click();
  await expect(section.getByText(/every 6h/)).toBeVisible();
  await expect(section.getByText('Paused', { exact: true })).toBeVisible();
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await section.getByRole('button', { name: 'Delete schedule' }).click();
  await expect(section.getByText('No saved schedules yet.')).toBeVisible();
  // Clean up generated rules only; schedule deletion deliberately retained them.
  const rules = await (await page.request.get('/api/pairs/alerts')).json();
  for (const rule of rules) if (['NVDA','META','AMZN'].includes(rule.ticker_a) && ['NVDA','META','AMZN'].includes(rule.ticker_b)) await page.request.delete(`/api/pairs/alerts/${rule.id}`);
});
