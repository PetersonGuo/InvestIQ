import type { Time, UTCTimestamp } from 'lightweight-charts';
export const intervals = ['1s', '5s', '10s', '15s', '30s', '1m', '2m', '3m', '5m', '15m', '30m', '1h', '4h', '1d'] as const;
export const intervalSeconds: Record<string, number> = { '1s':1,'5s':5,'10s':10,'15s':15,'30s':30,'1m':60,'2m':120,'3m':180,'5m':300,'15m':900,'30m':1800,'1h':3600,'4h':14400,'1d':86400 };
export const chartTime = (time: string): Time => time.includes('T') ? Math.floor(Date.parse(time)/1000) as UTCTimestamp : time;
