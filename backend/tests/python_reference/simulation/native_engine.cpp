// Native event loop shared by Python and C++ strategies. No Python calls for
// C++ strategies.
#include "strategy_api.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>

struct Point {
  double equity, benchmark, drawdown, cash;
  int64_t shares;
};
struct Fill {
  int index, signal, side;
  int64_t quantity;
  double price, pnl;
};
using Callback = double (*)(const SA_Bar *, int, double, int64_t, const char *);
extern "C" int sa_simulate(const SA_Bar *bars, int count, Callback callback,
                           const char *params, double initial,
                           double commission, double slip, int warmup,
                           Point *points, Fill *fills, int *fill_count,
                           double *final_signal, char *error) {
  try {
    double cash = initial, cost = 0, pending = -1, peak = initial,
           benchmark_cash = initial;
    int64_t shares = 0, benchmark_shares = 0;
    int signal = 0, n = 0;
    for (int i = 0; i < count; ++i) {
      const auto &bar = bars[i];
      if (i == warmup + 1) {
        double price = bar.open * (1 + slip / 10000);
        if ((initial - commission) / price > 9e18) {
          std::snprintf(error, 512, "Position exceeds native integer range.");
          return 1;
        }
        benchmark_shares =
            std::max(0.0, std::floor((initial - commission) / price));
        benchmark_cash = initial - benchmark_shares * price -
                         (benchmark_shares ? commission : 0);
      }
      if (pending >= 0 && i > warmup) {
        double target =
            std::floor((cash + shares * bar.open) * pending / bar.open);
        if (!std::isfinite(target) || target > 9e18) {
          std::snprintf(error, 512, "Position exceeds native integer range.");
          return 1;
        }
        int64_t delta = static_cast<int64_t>(target) - shares;
        if (delta > 0) {
          double price = bar.open * (1 + slip / 10000);
          int64_t quantity = std::min(
              delta, static_cast<int64_t>(std::max(
                         0.0, std::floor((cash - commission) / price))));
          if (quantity) {
            double total = quantity * price + commission;
            cash -= total;
            cost += total;
            shares += quantity;
            fills[n++] = {i, signal, 1, quantity, price, 0};
          }
        } else if (delta < 0) {
          double price = bar.open * (1 - slip / 10000);
          int64_t quantity = std::min(-delta, shares);
          double proceeds = quantity * price - commission;
          if (proceeds >= 0) {
            double basis = cost * quantity / shares;
            fills[n++] = {i, signal, -1, quantity, price, proceeds - basis};
            cash += proceeds;
            cost -= basis;
            shares -= quantity;
          }
        }
        pending = -1;
      }
      double equity = cash + shares * bar.close;
      if (!std::isfinite(equity)) {
        std::snprintf(error, 512, "Portfolio equity overflow.");
        return 1;
      }
      if (i >= warmup)
        peak = std::max(peak, equity);
      points[i] = {equity, benchmark_cash + benchmark_shares * bar.close,
                   (equity / peak - 1) * 100, cash, shares};
      double weight = callback(bars, i + 1, cash, shares, params);
      if (!std::isfinite(weight) ||
          (weight != -1 && (weight < 0 || weight > 1))) {
        std::snprintf(error, 512,
                      "Invalid strategy target weight at bar %d; expected hold "
                      "or 0 to 1.",
                      i);
        return 1;
      }
      if (i >= warmup && weight != -1) {
        pending = weight;
        signal = i;
      }
    }
    *fill_count = n;
    *final_signal = pending;
    return 0;
  } catch (const std::exception &e) {
    std::snprintf(error, 512, "C++ strategy exception: %s", e.what());
    return 1;
  } catch (...) {
    std::snprintf(error, 512, "Unknown C++ strategy exception.");
    return 1;
  }
}
