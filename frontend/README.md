# StockAssist web dashboard

See the [workspace README](../README.md) for setup, features, and operating limits.

Run `npm ci`, then `npm run dev -- --hostname 127.0.0.1` with the backend on port 8000. Set `STOCKASSIST_API_URL` in `.env.local` if the backend uses another address. The dashboard uses the local C++ backend and requires no cloud account.

Checks: `npm run lint`, `npm run typecheck`, and `npm run build`.
