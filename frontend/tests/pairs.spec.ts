import { test, expect } from '@playwright/test';

test('preview, create, trigger, edit, pause, rearm, and delete a pair alert', async ({ page }) => {
  test.setTimeout(65000);
  const errors: string[] = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.goto('/pairs');
  await expect(page.getByRole('heading', { name: 'Pair trading alerts' })).toBeVisible();
  await page.getByRole('combobox', { name: 'Metric', exact: true }).selectOption('ratio');
  await page.getByLabel('Alert level').fill('0.001');
  await page.getByRole('button', { name: 'Preview pair', exact: true }).click();
  await expect(page.getByText('Condition currently met')).toBeVisible();
  await expect(page.getByRole('img', { name: 'AAPL and MSFT price ratio history' })).toBeVisible();
  await page.getByRole('button', { name: 'Create pair alert', exact: true }).click();
  await expect(page.getByText('Watching', { exact: true })).toBeVisible();
  await expect(page.getByText('Triggered', { exact: true })).toBeVisible({ timeout: 45000 });
  const notifications = page.getByRole('heading', { name: 'Recent pair notifications' }).locator('..');
  await expect(notifications.getByText('Observed', { exact: false })).toBeVisible();
  await page.reload();
  await expect(page.getByText('Triggered', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Edit', exact: true }).click();
  await page.getByLabel('Alert level').fill('100');
  await page.getByRole('button', { name: 'Preview pair', exact: true }).click();
  await expect(page.getByText('Condition not met')).toBeVisible();
  await page.getByRole('button', { name: 'Save & rearm pair alert' }).click();
  await expect(page.getByText('Watching', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Pause AAPL MSFT pair alert' }).click();
  await expect(page.getByText('Paused', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Rearm AAPL MSFT pair alert' }).click();
  await expect(page.getByText('Watching', { exact: true })).toBeVisible();
  await page.screenshot({ path: 'test-results/pair-alerts-desktop.png', fullPage: true });
  await page.getByRole('button', { name: 'Delete AAPL MSFT pair alert' }).click();
  await expect(page.getByText('No pair alerts yet.', { exact: false })).toBeVisible();
  await expect(notifications.getByText('Observed', { exact: false })).toBeVisible();
  expect(errors).toEqual([]);
});

test('z-score preview, validation, and mobile layout', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/pairs');
  await page.getByRole('button', { name: 'Preview pair', exact: true }).click();
  await expect(page.getByRole('img', { name: 'AAPL and MSFT spread z-score history' })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await page.screenshot({ path: 'test-results/pair-alerts-mobile.png', fullPage: true });
  await page.getByLabel('Stock B', { exact: true }).fill('AAPL');
  await expect(page.getByRole('button', { name: 'Create pair alert', exact: true })).toBeDisabled();
  await page.getByRole('button', { name: 'Preview pair', exact: true }).click();
  await expect(page.getByRole('alert', { name: 'Pair alert error' })).toBeVisible();
});
