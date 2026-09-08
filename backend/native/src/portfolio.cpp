#include "simulation.hpp"
#include <numeric>
namespace sa {
J simulate_portfolio(const J &assets, PortfolioCallback callback, const J &params,
                     const J &settings) {
  const int n = assets.size(), warmup = settings.value("warmup", 0);
  if (n < 2 || n > 10)
    throw Error(422, "Portfolio strategies require 2–10 stocks.");
  const int count = assets[0].at("bars").size();
  if (count < warmup + 2)
    throw Error(422, "Not enough aligned bars after warmup.");
  const double initial = settings.at("initial_cash"), fee = settings.at("commission"),
               slip = settings.at("slippage_bps").get<double>() / 10000,
               max_gross = settings.value("max_gross_exposure", 1.0),
               borrow = settings.value("borrow_rate_percent", 3.0) / 100;
  if (initial <= 0 || fee < 0 || slip < 0 || slip > .1 || max_gross <= 0 || max_gross > 2 ||
      borrow < 0 || borrow > 1)
    throw Error(422, "Invalid portfolio execution settings.");
  std::vector<std::vector<SA_Bar>> bars(n);
  std::vector<std::string> symbols(n), dates;
  std::vector<SA_Asset> views(n);
  std::vector<int64_t> positions(n, 0);
  std::vector<double> averages(n, 0), targets(n, 0);
  for (int a = 0; a < n; ++a) {
    symbols[a] = assets[a].at("ticker");
    if (assets[a].at("bars").size() != size_t(count))
      throw Error(422, "Portfolio histories must be aligned.");
    double previous = -1e20;
    for (int i = 0; i < count; ++i) {
      auto &row = assets[a]["bars"][i];
      std::string time = row.at("time");
      double t = parse_time(time);
      if (t <= previous || (a && time != dates[i]))
        throw Error(422, "Portfolio timestamps must match and increase.");
      previous = t;
      if (!a)
        dates.push_back(time);
      double o = row.at("open"), h = row.at("high"), l = row.at("low"), c = row.at("close"),
             v = row.at("volume");
      if (!std::isfinite(o + h + l + c + v) || std::min({o, h, l, c}) <= 0 || v < 0 ||
          l > std::min(o, c) || h < std::max(o, c))
        throw Error(422, "Invalid portfolio OHLCV data.");
      bars[a].push_back({int64_t(t * 1000), o, h, l, c, v});
    }
    views[a] = {symbols[a].c_str(), bars[a].data(), 0};
  }
  double cash = initial, peak = initial, total_fees = 0, total_borrow = 0;
  bool pending = false;
  int closes = 0, wins = 0;
  J curve = J::array(), fills = J::array();
  std::string parameters = params.dump();
  for (int i = warmup; i < count; ++i) {
    if (i > warmup) {
      double short_value = 0;
      for (int a = 0; a < n; ++a)
        short_value += std::max(int64_t(0), -positions[a]) * bars[a][i - 1].close;
      double charge = short_value * borrow * (bars[0][i].timestamp - bars[0][i - 1].timestamp) /
                      (365.0 * 86400000);
      cash -= charge;
      total_borrow += charge;
    }
    if (pending) {
      double equity = cash;
      for (int a = 0; a < n; ++a)
        equity += positions[a] * bars[a][i].open;
      if (equity <= 0)
        throw Error(422, "Portfolio equity exhausted at " + dates[i]);
      double planned_gross = 0;
      for (double weight : targets)
        planned_gross += std::abs(weight);
      double budget = std::max(0.0, equity - n * fee) / (1 + slip * planned_gross);
      for (int a = 0; a < n; ++a) {
        double target_shares = std::trunc(targets[a] * budget / bars[a][i].open);
        if (!std::isfinite(target_shares) || std::abs(target_shares) > 1e12)
          throw Error(422, "Position size exceeds the model range.");
        int64_t desired = target_shares, delta = desired - positions[a];
        if (!delta)
          continue;
        double price = bars[a][i].open * (1 + (delta > 0 ? slip : -slip));
        auto old = positions[a];
        int64_t closed =
            old && ((old > 0) != (delta > 0)) ? std::min(std::abs(old), std::abs(delta)) : 0;
        double pnl = closed ? closed * (price - averages[a]) * (old > 0 ? 1 : -1) -
                                  fee * closed / std::abs(delta)
                            : 0;
        int64_t opened = std::abs(delta) - closed;
        if (opened) {
          double entry = price + (desired > 0 ? 1 : -1) * fee * opened / std::abs(delta) / opened;
          int64_t retained = closed ? 0 : std::abs(old);
          averages[a] = (averages[a] * retained + entry * opened) / (retained + opened);
        } else if (!desired)
          averages[a] = 0;
        cash -= delta * price + fee;
        total_fees += fee;
        positions[a] = desired;
        if (closed) {
          ++closes;
          if (pnl > 0)
            ++wins;
        }
        fills.push_back({{"ticker", symbols[a]},
                         {"time", dates[i]},
                         {"signal_date", dates[i - 1]},
                         {"side", delta > 0 ? "buy" : "sell"},
                         {"quantity", std::abs(delta)},
                         {"price", rounded(price)},
                         {"commission", fee},
                         {"realized_pnl", closed ? J(rounded(pnl)) : J(nullptr)}});
      }
    }
    double equity = cash, gross = 0, net = 0;
    J held = J::object();
    for (int a = 0; a < n; ++a) {
      double value = positions[a] * bars[a][i].close;
      equity += value;
      gross += std::abs(value);
      net += value;
      held[symbols[a]] = positions[a];
      views[a].shares = positions[a];
    }
    if (!std::isfinite(equity) || equity <= 0)
      throw Error(422, "Portfolio equity exhausted at " + dates[i]);
    peak = std::max(peak, equity);
    curve.push_back({{"time", dates[i]},
                     {"equity", rounded(equity)},
                     {"benchmark", initial},
                     {"cash", rounded(cash)},
                     {"shares", 0},
                     {"positions", held},
                     {"gross_exposure_percent", 100 * gross / equity},
                     {"net_exposure_percent", 100 * net / equity},
                     {"drawdown_percent", 100 * (equity / peak - 1)}});
    std::fill(targets.begin(), targets.end(), 0);
    int action = callback(views.data(), n, i + 1, cash, equity, parameters.c_str(), targets.data());
    if (action != 0 && action != 1)
      throw Error(422, "Portfolio callback failed or returned an invalid action.");
    pending = action == 1;
    if (pending) {
      double sum = 0;
      for (double weight : targets) {
        if (!std::isfinite(weight))
          throw Error(422, "Portfolio targets must be finite.");
        sum += std::abs(weight);
      }
      if (sum > max_gross + 1e-9)
        throw Error(422, "Portfolio target gross exposure exceeds the configured limit.");
    }
  }
  double final = curve.back()["equity"];
  return {{"equity_curve", curve},
          {"fills", fills},
          {"unfilled_final_signal", pending ? J(targets) : J(nullptr)},
          {"metrics",
           {{"initial_cash", initial},
            {"final_equity", final},
            {"total_return_percent", 100 * (final / initial - 1)},
            {"fill_count", fills.size()},
            {"sell_count", closes},
            {"win_rate_percent", closes ? J(100.0 * wins / closes) : J(nullptr)},
            {"total_commission", total_fees},
            {"total_borrow_cost", total_borrow},
            {"open_shares", 0}}},
          {"open_positions", curve.back()["positions"]}};
}
namespace {
double buy_hold(const SA_Bar *, int, double, int64_t shares, const char *) {
  return shares ? -1 : 1;
}
int equal_hold(const SA_Asset *assets, int n, int, double, double, const char *, double *weights) {
  for (int a = 0; a < n; ++a)
    if (assets[a].shares)
      return 0;
  for (int a = 0; a < n; ++a)
    weights[a] = 1.0 / n;
  return 1;
}
J risk(const J &curve, const std::string &field, bool daily, double initial) {
  std::vector<double> returns, reference;
  double peak = initial, dd = 0, gross = 0, net = 0, maxgross = 0;
  int invested = 0;
  for (size_t i = 0; i < curve.size(); ++i) {
    double value = curve[i].at(field);
    peak = std::max(peak, value);
    dd = std::max(dd, 1 - value / peak);
    double g = curve[i].value(field == "equity" ? "gross_exposure_percent" : field + "_gross", 0.0);
    gross += g;
    maxgross = std::max(maxgross, g);
    net += curve[i].value(field == "equity" ? "net_exposure_percent" : field + "_net", 0.0);
    if (g > 1e-8)
      ++invested;
    if (i) {
      returns.push_back(value / curve[i - 1].at(field).get<double>() - 1);
      reference.push_back(curve[i].at("spy").get<double>() / curve[i - 1].at("spy").get<double>() -
                          1);
    }
  }
  double mean = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
  double rm = std::accumulate(reference.begin(), reference.end(), 0.0) / reference.size();
  double variance = 0, down = 0, rv = 0, cov = 0, tracking = 0;
  for (size_t i = 0; i < returns.size(); ++i) {
    variance += std::pow(returns[i] - mean, 2);
    down += std::pow(std::min(0.0, returns[i]), 2);
    rv += std::pow(reference[i] - rm, 2);
    cov += (returns[i] - mean) * (reference[i] - rm);
    tracking += std::pow(returns[i] - reference[i] - (mean - rm), 2);
  }
  double vol = std::sqrt(variance / returns.size()), downside = std::sqrt(down / returns.size()),
         scale = daily ? std::sqrt(252) : 1;
  std::sort(returns.begin(), returns.end());
  size_t tail = std::max(size_t(1), size_t(std::ceil(returns.size() * .05)));
  double cvar = -std::accumulate(returns.begin(), returns.begin() + tail, 0.0) / tail;
  double final = curve.back().at(field),
         exponent = std::log(final / initial) * 252 / (curve.size() - 1);
  return {{"total_return_percent", 100 * (final / initial - 1)},
          {"max_drawdown_percent", 100 * dd},
          {"volatility_percent", 100 * vol * scale},
          {"downside_volatility_percent", 100 * downside * scale},
          {"annualized_return_percent",
           daily && exponent < 700 ? J(100 * std::expm1(exponent)) : J(nullptr)},
          {"sharpe_ratio", vol > 1e-12 ? J(mean / vol * scale) : J(nullptr)},
          {"sortino_ratio", downside > 1e-12 ? J(mean / downside * scale) : J(nullptr)},
          {"var_95_percent", 100 * std::max(0.0, -returns[tail - 1])},
          {"cvar_95_percent", 100 * std::max(0.0, cvar)},
          {"beta_to_spy", rv > 1e-20 ? J(cov / rv) : J(nullptr)},
          {"tracking_error_percent", 100 * std::sqrt(tracking / returns.size()) * scale},
          {"average_gross_exposure_percent", gross / curve.size()},
          {"max_gross_exposure_percent", maxgross},
          {"average_net_exposure_percent", net / curve.size()},
          {"time_in_market_percent", 100.0 * invested / curve.size()}};
}
} // namespace
void add_comparison(J &result, const J &input, const J &settings) {
  if (!input.contains("benchmark_bars"))
    return;
  auto spy = simulate(input["benchmark_bars"], buy_hold, J::object(), settings);
  J basket;
  bool multi = input.contains("assets");
  if (multi) {
    J baseline_settings = settings;
    baseline_settings["max_gross_exposure"] = 1;
    basket = simulate_portfolio(input["assets"], equal_hold, J::object(), baseline_settings);
  }
  auto &curve = result["equity_curve"];
  if (curve.size() != spy["equity_curve"].size())
    throw Error(422, "Benchmark alignment failed.");
  int warmup = settings.at("warmup");
  for (size_t i = 0; i < curve.size(); ++i) {
    auto &p = curve[i];
    auto &b = spy["equity_curve"][i];
    if (p["time"] != b["time"])
      throw Error(422, "Benchmark timestamps differ.");
    p["spy"] = b["equity"];
    double close = input["benchmark_bars"][i + warmup]["close"];
    p["spy_gross"] = p["spy_net"] =
        100 * b["shares"].get<double>() * close / b["equity"].get<double>();
    if (multi) {
      p["benchmark"] = basket["equity_curve"][i]["equity"];
      p["benchmark_gross"] = basket["equity_curve"][i]["gross_exposure_percent"];
      p["benchmark_net"] = basket["equity_curve"][i]["net_exposure_percent"];
    } else {
      double c = input["bars"][i + warmup]["close"];
      p["gross_exposure_percent"] = p["net_exposure_percent"] =
          100 * p["shares"].get<double>() * c / p["equity"].get<double>();
      // Reuse the core's buy-and-hold valuation; infer its fixed share count from adjacent mark
      // changes below.
      p["benchmark_gross"] = p["benchmark_net"] = 0;
    }
  }
  if (!multi) {
    auto stock = simulate(input["bars"], buy_hold, J::object(), settings);
    for (size_t i = 0; i < curve.size(); ++i) {
      double c = input["bars"][i + warmup]["close"];
      curve[i]["benchmark_gross"] = curve[i]["benchmark_net"] =
          100 * stock["equity_curve"][i]["shares"].get<double>() * c /
          stock["equity_curve"][i]["equity"].get<double>();
    }
  }
  bool daily = settings.value("interval", std::string("1d")) == "1d";
  double initial = settings.at("initial_cash");
  result["risk_comparison"] = {{"strategy", risk(curve, "equity", daily, initial)},
                               {"spy", risk(curve, "spy", daily, initial)},
                               {"buy_and_hold", risk(curve, "benchmark", daily, initial)}};
  result["risk_basis"] =
      daily ? "Daily returns; volatility and risk-adjusted ratios annualized using 252 sessions. "
              "VaR/CVaR are historical one-day 95% losses. Zero risk-free rate."
            : "Per-bar returns; no annualization. VaR/CVaR are historical one-bar 95% losses. Zero "
              "risk-free rate.";
  result["risk_basis"] =
      result["risk_basis"].get<std::string>() +
      " Exposures and drawdowns use closing marks; intrabar extremes are not captured.";
  result["benchmark_label"] = multi ? "Equal-weight basket buy & hold" : "Stock buy & hold";
  result["metrics"]["spy_return_percent"] =
      result["risk_comparison"]["spy"]["total_return_percent"];
  result["metrics"]["benchmark_return_percent"] =
      result["risk_comparison"]["buy_and_hold"]["total_return_percent"];
  if (multi)
    for (auto key : {"max_drawdown_percent", "annualized_return_percent", "sharpe_ratio"})
      result["metrics"][key] = result["risk_comparison"]["strategy"][key];
}
} // namespace sa
