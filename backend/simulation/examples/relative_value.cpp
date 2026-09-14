#include "relative_value_math.h"
#include "strategy_api.h"
// Asset order: target, N references, market ETF, sector ETF, factor ETFs.
// Prior-only signals. Cost-aware entry and no-trade bands for hedge turnover.
namespace {
int entry_bar = -1, entry_checks = 0, entry_rejected = 0, rebalance_skipped = 0,
    risk_rebalances = 0, time_exits = 0;
}
extern "C" const char *sa_strategy_diagnostics() {
  static std::string report;
  report = "{\"entry_checks\":" + std::to_string(entry_checks) +
           ",\"entries_rejected_by_cost\":" + std::to_string(entry_rejected) +
           ",\"rebalances_skipped\":" + std::to_string(rebalance_skipped) +
           ",\"risk_rebalances\":" + std::to_string(risk_rebalances) +
           ",\"time_exits\":" + std::to_string(time_exits) + "}";
  return report.c_str();
}
extern "C" int on_portfolio(const SA_Asset *assets, int n, int count, double,
                            double equity, const char *params,
                            double *weights) {
  const int refs = sa_parameter(params, "reference_count", 1),
            window = sa_parameter(params, "lookback", 120),
            mode = sa_parameter(params, "mode", 1);
  if (refs < 1 || refs > 20 || window < 40 || window > 250 || mode < 0 ||
      mode > 2 || n < refs + 4 || n > 29)
    return -1;
  bool held = false;
  for (int a = 0; a < n; ++a) {
    weights[a] = 0;
    held = held || assets[a].shares != 0;
  }
  if (!held)
    entry_bar = -1;
  else if (entry_bar < 0)
    entry_bar = count;
  if (count < window + 3)
    return held ? 1 : 0;
  // These templates model DAILY relationships only.
  if (assets[0].bars[count - 1].timestamp -
          assets[0].bars[count - 2].timestamp <
      20 * 3600000LL)
    return -1;
  std::vector<std::vector<double>> prices(n);
  for (int a = 0; a < n; ++a)
    for (int i = count - window - 3; i < count; ++i)
      prices[a].push_back(assets[a].bars[i].close);
  std::vector<double> fixed;
  for (int a = 0; a < refs; ++a)
    fixed.push_back(sa_parameter(
        params, ("reference_weight_" + std::to_string(a)).c_str(), 1));
  double gap = 0;
  bool event = sa_parameter(params, "event", 0) != 0;
  if (event) {
    if (refs != 1)
      return -1;
    auto from = sa_parameter(params, "terms_from", 0),
         until = sa_parameter(params, "terms_until", 0);
    if (assets[0].bars[count - 1].timestamp < from ||
        assets[0].bars[count - 1].timestamp >= until)
      return held ? 1 : 0;
    double cash = sa_parameter(params, "cash_per_share", 0),
           shares = sa_parameter(params, "shares_per_share", 1);
    double fair = cash + shares * prices[1].back();
    if (fair <= 0)
      return -1;
    fixed[0] = shares * prices[1].back() / prices[0].back();
    gap = prices[0].back() / fair - 1;
  }
  try {
    auto model = sa_rv(prices, refs, window, mode, fixed);
    double score = model.score, entry = sa_parameter(params, "entry_z", 2),
           exit = sa_parameter(params, "exit_z", .5),
           stop = sa_parameter(params, "stop_z", 4);
    if (entry <= 0 || exit < 0 || exit >= entry || stop <= entry)
      return -1;
    if ((event && score * gap <= 0) || std::abs(score) <= exit ||
        std::abs(score) >= stop)
      return held ? 1 : 0;
    if (!held && std::abs(score) < entry)
      return 0;
    double direction = mode == 2 ? (score > 0 ? 1 : -1) : (score > 0 ? -1 : 1);
    // Close on signal reversal before considering a new entry.
    if (held && assets[0].shares * direction < 0)
      return 1;
    bool aware = sa_parameter(params, "cost_aware", 1) != 0;
    double scale =
        std::min(1., sa_parameter(params, "_sa_max_gross_exposure", 1));
    if (aware) {
      double fee = sa_parameter(params, "_sa_commission", -1),
             slip = sa_parameter(params, "_sa_slippage_bps", -1),
             borrow = sa_parameter(params, "_sa_borrow_rate_percent", -1),
             cushion = sa_parameter(params, "cost_cushion", 2),
             capture = sa_parameter(params, "capture_fraction", .5),
             days = sa_parameter(params, "cost_holding_days", 10),
             band = sa_parameter(params, "rebalance_band", .05),
             risk_band = sa_parameter(params, "max_weight_drift", .15),
             max_hold = sa_parameter(params, "max_holding_bars", 20);
      if (fee < 0 || slip < 0 || borrow < 0 || equity <= 0 || cushion < 1 ||
          capture <= 0 || capture > 1 || days < 1 || days > 365 || band < 0 ||
          risk_band <= band || risk_band > 1 || max_hold < 1 || max_hold > 250)
        return -1;
      if (held && count - entry_bar >= max_hold) {
        ++time_exits;
        return 1;
      }
      double turnover = 0, short_weight = 0, gross = 0;
      int orders = 0, legs = 0;
      for (int a = 0; a < n; ++a) {
        double target = direction * model.weights[a] * scale;
        double current =
            assets[a].shares * assets[a].bars[count - 1].close / equity;
        turnover += std::abs(target - current);
        gross += std::abs(target);
        short_weight += std::max(0., -target);
        if (std::abs(target) > 1e-8)
          ++legs;
        if (std::abs(target - current) * equity >=
            assets[a].bars[count - 1].close)
          ++orders;
      }
      double potential = sa_rv_capture(model, mode, exit, capture) * scale;
      if (!held) {
        ++entry_checks;
        double costs = sa_rv_cost(2 * gross, 2 * legs, short_weight, days,
                                  equity, fee, slip, borrow);
        if (potential <= cushion * costs) {
          ++entry_rejected;
          return 0;
        }
        entry_bar = count;
      } else if (turnover >= risk_band) {
        // Risk control is allowed to trade even when the remaining edge is
        // small.
        ++risk_rebalances;
      } else {
        double costs =
            sa_rv_cost(turnover, orders, 0, 0, equity, fee, slip, borrow);
        if (turnover < band ||
            potential * std::min(turnover, 1.) <= cushion * costs) {
          ++rebalance_skipped;
          return 0;
        }
      }
    }
    for (int a = 0; a < n; ++a)
      weights[a] = direction * model.weights[a] * scale;
    return 1;
  } catch (const std::exception &) {
    return held ? 1 : 0;
  }
}
