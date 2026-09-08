#pragma once
#include "common.hpp"
#include "strategy_api.h"
namespace sa {
using Callback = double (*)(const SA_Bar *, int, double, int64_t, const char *);
using PortfolioCallback = int (*)(const SA_Asset *, int, int, double, double, const char *,
                                  double *);
J simulate_portfolio(const J &, PortfolioCallback, const J &, const J &);
void add_comparison(J &, const J &, const J &);
J simulate(const J &bars, Callback, const J &params, const J &settings);
} // namespace sa
