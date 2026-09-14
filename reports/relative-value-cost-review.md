# Relative-value cost review

Compared unchanged saved IBKR bars and original C++ strategy source against the cost-aware version. No data was downloaded or substituted. Each run starts with $100,000, $1 per fill, 5 bps adverse slippage, 3% annual short borrow and 1× maximum gross exposure.

Fixed rule settings: cost cushion 2; capture fraction 0.5; assumed holding 10 calendar days; rebalance L1 band 0.05; forced hedge correction at L1 drift 0.15; maximum holding 20 bars. No parameter search was performed.

This is a retrospective comparison. The original full-period results had already been inspected, so the later-period check is not genuinely unseen out-of-sample evidence. It starts flat on July 1, 2026, with prior bars available for model estimation. Independent forward validation is still required.

| Pair target | Period | Rules | Net return | Fills | Slippage | Fees | Borrow | Max drawdown |
|---|---|---|---:|---:|---:|---:|---:|---:|
| SPY | 2026-03-09–2026-09-11 | original | -0.584% | 66 | $500.92 | $66.00 | $52.51 | 0.584% |
| SPY | 2026-07-01–2026-09-11 | original | 0.000% | 0 | $0.00 | $0.00 | $0.00 | 0.000% |
| SPY | 2026-03-09–2026-09-11 | cost_aware | 0.000% | 0 | $0.00 | $0.00 | $0.00 | 0.000% |
| SPY | 2026-07-01–2026-09-11 | cost_aware | 0.000% | 0 | $0.00 | $0.00 | $0.00 | 0.000% |
| GLD | 2026-03-09–2026-09-11 | original | -0.747% | 70 | $599.52 | $70.00 | $80.74 | 0.747% |
| GLD | 2026-07-01–2026-09-11 | original | -0.129% | 8 | $98.51 | $8.00 | $4.02 | 0.129% |
| GLD | 2026-03-09–2026-09-11 | cost_aware | 0.000% | 0 | $0.00 | $0.00 | $0.00 | 0.000% |
| GLD | 2026-07-01–2026-09-11 | cost_aware | 0.000% | 0 | $0.00 | $0.00 | $0.00 | 0.000% |
| GOOG | 2026-03-09–2026-09-11 | original | -0.409% | 100 | $702.14 | $100.00 | $116.97 | 0.548% |
| GOOG | 2026-07-01–2026-09-11 | original | -0.005% | 12 | $98.45 | $12.00 | $4.07 | 0.161% |
| GOOG | 2026-03-09–2026-09-11 | cost_aware | 0.000% | 0 | $0.00 | $0.00 | $0.00 | 0.000% |
| GOOG | 2026-07-01–2026-09-11 | cost_aware | 0.000% | 0 | $0.00 | $0.00 | $0.00 | 0.000% |
| JPM | 2026-03-09–2026-09-11 | original | 3.219% | 87 | $321.50 | $87.00 | $79.73 | 0.292% |
| JPM | 2026-07-01–2026-09-11 | original | 1.845% | 50 | $109.84 | $50.00 | $53.33 | 0.264% |
| JPM | 2026-03-09–2026-09-11 | cost_aware | 3.224% | 47 | $313.81 | $47.00 | $79.85 | 0.314% |
| JPM | 2026-07-01–2026-09-11 | cost_aware | 1.867% | 17 | $104.45 | $17.00 | $53.39 | 0.309% |

The strategy diagnostics and source-run/data hashes are in the accompanying JSON file. Zero trades means the policy stayed in cash; it does not establish an investable edge. Existing limitations remain: daily next-open fills, no dividends, no live borrow availability, no liquidity impact model and a selected current universe. This does not test the fundamental valuation filter.

The four updated full-period runs are saved in the application with names beginning **Cost-aware replay**. Their IDs and verified source-data hash matches are recorded in [relative-value-cost-saved-runs.json](relative-value-cost-saved-runs.json). Provider-backed attempts made while IBKR was offline remain failed records; the successful replay runs use the original stored IBKR dataset explicitly.
