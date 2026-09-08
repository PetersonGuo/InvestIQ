import { test, expect } from '@playwright/test';

test('discover a stock, run Python and C++, save, and reopen results', async ({ page }) => {
  test.setTimeout(60000);
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/research');
  await expect(page.getByLabel('Strategy code')).toContainText('class Strategy');
  await page.getByRole('button', { name: 'Scan US stocks' }).click();
  await expect(page.getByText('Synthetic demo scan', { exact: false })).toBeVisible();
  await page.getByLabel('Ticker', { exact: true }).fill('AAPL');
  await page.getByLabel('Strategy name').fill('Browser Python strategy');
  await page.getByRole('button', { name: 'Save version' }).click();
  await expect(page.getByRole('status')).toContainText('Strategy version saved');
  await page.getByRole('button', { name: 'Run backtest', exact: true }).click();
  await expect(page.getByRole('img', { name: 'Strategy equity compared with buy and hold' })).toBeVisible({ timeout: 20000 });
  await expect(page.getByRole('heading', { name: 'AAPL · Browser Python strategy' })).toBeVisible();
  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export JSON' }).click();
  expect((await download).suggestedFilename()).toMatch(/stockassist-.*\.json/);
  await page.getByRole('button', { name: 'C++', exact: true }).click();
  await expect(page.getByLabel('Strategy code')).toContainText('extern "C"');
  await page.getByLabel('Strategy name').fill('Browser C++ strategy');
  await page.getByRole('button', { name: 'Run backtest', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'AAPL · Browser C++ strategy' })).toBeVisible({ timeout: 30000 });
  await page.reload();
  await page.getByRole('button', { name: /AAPL · Browser C\+\+ strategy/ }).click();
  await expect(page.getByRole('img', { name: 'Strategy equity compared with buy and hold' })).toBeVisible();
  await page.getByRole('button', { name: 'Load inputs' }).click();
  await expect(page.getByLabel('Strategy name')).toHaveValue('Browser C++ strategy');
  await expect(page.getByRole('button', { name: 'C++', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await page.screenshot({ path: 'test-results/research-desktop.png', fullPage: true });
  expect(errors).toEqual([]);
});

test('strategy errors are visible and mobile layout fits', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/research');
  await expect(page.getByLabel('Strategy code')).toContainText('class Strategy');
  await page.getByLabel('Strategy code').fill('class Strategy:\n    def on_bar(self, ctx):\n        raise ValueError("Example strategy error")');
  await page.getByRole('button', { name: 'Run backtest', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Backtest failed' })).toBeVisible({ timeout: 20000 });
  await expect(page.getByRole('alert').filter({ hasText: 'Example strategy error' })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
});

test('presets load matching language, parameters, and warmup', async ({ page }) => {
  await page.goto('/research');
  await expect(page.getByLabel('Strategy code')).toContainText('class Strategy');
  await page.getByLabel('Starter strategy').selectOption('mean_reversion');
  await page.getByRole('button', { name: 'Load preset', exact: true }).click();
  await expect(page.getByLabel('Strategy name')).toHaveValue('Z-score mean reversion');
  await expect(page.getByLabel('Warmup bars')).toHaveValue('20');
  await expect(page.getByLabel('Strategy parameters')).toContainText('entry_z');
  await page.getByRole('button', { name: 'C++', exact: true }).click();
  await expect(page.getByLabel('Strategy code')).toContainText('std::sqrt');
  await page.getByRole('button', { name: 'Python', exact: true }).click();
  await expect(page.getByLabel('Strategy code')).toContainText('from math import sqrt');
  await page.getByLabel('Starter strategy').selectOption('buy_and_hold');
  await page.getByRole('button', { name: 'Load preset', exact: true }).click();
  await expect(page.getByLabel('Warmup bars')).toHaveValue('0');
  await expect(page.getByLabel('Strategy parameters')).toHaveValue('{}');
  await page.getByRole('button', { name: 'Run backtest', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'AAPL · Buy and hold' })).toBeVisible({ timeout: 20000 });
  await expect(page.getByRole('img', { name: 'Strategy equity compared with buy and hold' })).toBeVisible();
});

test('identical reruns return cached results and force rerun bypasses them', async ({ page }) => {
  await page.goto('/research');
  await expect(page.getByLabel('Strategy code')).toContainText('class Strategy');
  await page.getByLabel('Strategy code').fill('class Strategy:\n    def on_bar(self, ctx):\n        return 0.314');
  await page.getByRole('button', { name: 'Run backtest', exact: true }).click();
  await expect(page.getByRole('img', { name: 'Strategy equity compared with buy and hold' })).toBeVisible({ timeout: 20000 });
  const cached = page.waitForResponse(r => r.url().endsWith('/api/backtests') && r.request().method() === 'POST');
  await page.getByRole('button', { name: 'Run backtest', exact: true }).click();
  expect((await (await cached).json()).cache_hit).toBe(true);
  await expect(page.getByRole('status')).toContainText('Cached result');
  await expect(page.getByRole('button', { name: 'Run backtest', exact: true })).toBeEnabled();
  await page.getByLabel('Force rerun').check();
  const fresh = page.waitForResponse(r => r.url().endsWith('/api/backtests') && r.request().method() === 'POST');
  await page.getByRole('button', { name: 'Run backtest', exact: true }).click();
  expect((await (await fresh).json()).cache_hit).toBe(false);
  await expect(page.getByRole('img', { name: 'Strategy equity compared with buy and hold' })).toBeVisible({ timeout: 20000 });
});

test('zooming out fetches older chart history', async ({ page }) => {
  await page.goto('/');
  const chart = page.getByRole('img', { name: /daily candlestick price chart/ });
  await expect(chart).toBeVisible();
  // Initial viewport may prefetch a page; zoom far enough to cross that page too.
  await expect(page.getByText(/daily bars loaded/)).toBeVisible();
  const history = page.waitForResponse(r => r.url().includes('before=') && r.url().includes('/api/stocks/'));
  await chart.hover();
  for (let i = 0; i < 35; i++) {
    await page.mouse.wheel(0, 100);
    await page.evaluate(() => new Promise(requestAnimationFrame));
  }
  const result = await (await history).json();
  expect(result.bars.length).toBeGreaterThan(0);
  await expect(page.getByText(/daily bars loaded/)).toBeVisible();
});

test('one-second chart history and C++ intraday backtest', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/');
  await page.getByLabel('Candle interval', { exact: true }).selectOption('1s');
  await expect(page.getByRole('img', { name: /AAPL 1s candlestick/ })).toBeVisible();
  const older = page.waitForResponse(r => r.url().includes('before=') && r.url().includes('interval=1s'));
  await page.getByRole('button', { name: 'Load older history' }).click();
  const history = await (await older).json();
  expect(history.bars[0].time).toContain('T');
  await page.goto('/research');
  await expect(page.getByLabel('Strategy code')).toContainText('class Strategy');
  await page.getByRole('button', { name: 'C++', exact: true }).click();
  await page.getByLabel('Backtest interval').selectOption('1s');
  await expect(page.getByRole('button', { name: 'Run backtest', exact: true })).toBeEnabled();
  await expect(page.getByLabel('Start time · UTC')).toHaveAttribute('type', 'datetime-local');
  await page.getByRole('button', { name: 'Run backtest', exact: true }).click();
  await expect(page.getByRole('img', { name: 'Strategy equity compared with buy and hold' })).toBeVisible({ timeout: 20000 });
  await expect(page.getByText(/500 1s bars/)).toBeVisible();
  await expect(page.getByText('Sharpe · daily runs only')).toBeVisible();
  expect(errors).toEqual([]);
});

test('individual trade view preserves trades within one second', async ({ page }) => {
  await page.route('**/api/stocks/AAPL/ticks', route => route.fulfill({ json: { ticks: [
    {time:'2026-09-04T19:59:59Z',price:319.835,size:40,exchange:'IEX',conditions:'',past_limit:false,unreported:false},
    {time:'2026-09-04T19:59:59Z',price:319.835,size:60,exchange:'NASDAQ',conditions:'',past_limit:false,unreported:false},
  ]} }));
  await page.goto('/');
  await page.getByText('Individual trades · IBKR Time & Sales', { exact: true }).click();
  await page.getByRole('button', { name: 'Load latest trades' }).click();
  await expect(page.getByText(/2 trades ·/)).toBeVisible();
  await expect(page.getByRole('cell', { name: '2026-09-04T19:59:59Z' })).toHaveCount(2);
  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export trades' }).click();
  expect((await download).suggestedFilename()).toBe('AAPL-trades.json');
});
