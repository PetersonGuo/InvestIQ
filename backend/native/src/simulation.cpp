#include "simulation.hpp"
#include <numeric>
struct Point {
  double equity, benchmark, drawdown, cash;
  int64_t shares;
};
struct Fill {
  int index, signal, side;
  int64_t quantity;
  double price, pnl;
};
extern "C" int sa_simulate(const SA_Bar *, int, sa::Callback, const char *, double, double, double,
                           int, Point *, Fill *, int *, double *, char *);
namespace sa {
J simulate(const J &bars, Callback callback, const J &params, const J &settings) {
  size_t count = bars.size();
  int warmup = settings.value("warmup", 0);
  double initial = settings.value("initial_cash", 100000.0),
         commission = settings.value("commission", 1.0), slip = settings.value("slippage_bps", 5.0);
  if (count < 2 || warmup < 0 || warmup > int(count) - 2)
    throw Error(422, "Choose at least two trading bars after warmup.");
  if (!std::isfinite(initial) || initial <= 0 || !std::isfinite(commission) || commission < 0 ||
      commission > initial || !std::isfinite(slip) || slip < 0 || slip > 1000)
    throw Error(422, "Invalid capital, commission, or slippage.");
  std::vector<SA_Bar> input;
  std::vector<std::string> dates;
  double previous = -1e20;
  for (auto &bar : bars) {
    auto time = bar.at("time").get<std::string>();
    double t = parse_time(time);
    if (t <= previous)
      throw Error(422, "Bars must have unique, ascending dates.");
    previous = t;
    double o = bar.at("open"), h = bar.at("high"), l = bar.at("low"), c = bar.at("close"),
           v = bar.at("volume");
    for (double x : {o, h, l, c, v})
      if (!std::isfinite(x))
        throw Error(422, "Invalid OHLCV data.");
    if (std::min({o, h, l, c}) <= 0 || v < 0 || l > std::min(o, c) || h < std::max(o, c))
      throw Error(422, "Invalid OHLCV range.");
    input.push_back({int64_t(t * 1000), o, h, l, c, v});
    dates.push_back(time);
  }
  std::vector<Point> points(count);
  std::vector<Fill> fills(count);
  int fill_count = 0;
  double pending = -1;
  char error[512] = {0};
  std::string parameters = params.dump();
  int rc = sa_simulate(input.data(), count, callback, parameters.c_str(), initial, commission, slip,
                       warmup, points.data(), fills.data(), &fill_count, &pending, error);
  if (rc)
    throw Error(422, error);
  J curve = J::array(), trades = J::array();
  double drawdown = 0;
  int sells = 0, wins = 0;
  for (int i = warmup; i < int(count); i++) {
    auto &p = points[i];
    drawdown = std::min(drawdown, p.drawdown);
    curve.push_back({{"time", dates[i]},
                     {"equity", rounded(p.equity)},
                     {"benchmark", rounded(p.benchmark)},
                     {"drawdown_percent", rounded(p.drawdown)},
                     {"cash", rounded(p.cash)},
                     {"shares", p.shares}});
  }
  for (int i = 0; i < fill_count; i++) {
    auto &f = fills[i];
    if (f.side == -1) {
      ++sells;
      if (f.pnl > 0)
        ++wins;
    }
    trades.push_back({{"signal_date", dates[f.signal]},
                      {"time", dates[f.index]},
                      {"side", f.side == 1 ? "buy" : "sell"},
                      {"quantity", f.quantity},
                      {"price", rounded(f.price)},
                      {"commission", commission},
                      {"realized_pnl", f.side == 1 ? J(nullptr) : J(rounded(f.pnl))}});
  }
  std::vector<double> returns;
  for (size_t i = 1; i < curve.size(); i++)
    returns.push_back(curve[i]["equity"].get<double>() / curve[i - 1]["equity"].get<double>() - 1);
  double mean = returns.empty()
                    ? 0
                    : std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size(),
         variance = 0;
  for (double r : returns)
    variance += (r - mean) * (r - mean);
  double vol = returns.size() > 1 ? std::sqrt(variance / returns.size()) : 0;
  double final = curve.back()["equity"], years = (curve.size() - 1) / 252.0;
  J annual = nullptr, sharpe = nullptr;
  if (settings.value("interval", std::string("1d")) == "1d") {
    if (years > 0 && final > 0) {
      double exponent = std::log(final / initial) / years;
      if (exponent < 700)
        annual = std::expm1(exponent) * 100;
    }
    if (vol)
      sharpe = mean / vol * std::sqrt(252);
  }
  J metrics = {
      {"initial_cash", initial},
      {"final_equity", final},
      {"total_return_percent", (final / initial - 1) * 100},
      {"benchmark_return_percent", (curve.back()["benchmark"].get<double>() / initial - 1) * 100},
      {"annualized_return_percent", annual},
      {"max_drawdown_percent", std::abs(drawdown)},
      {"sharpe_ratio", sharpe},
      {"fill_count", fill_count},
      {"sell_count", sells},
      {"win_rate_percent", sells ? J(100.0 * wins / sells) : J(nullptr)},
      {"total_commission", commission * fill_count},
      {"open_shares", points.back().shares}};
  return {{"metrics", metrics},
          {"equity_curve", curve},
          {"fills", trades},
          {"unfilled_final_signal", pending == -1 ? J(nullptr) : J(pending)}};
}
} // namespace sa
