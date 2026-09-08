#include "pairs.hpp"
#include "simulation.hpp"
#include <iostream>
using namespace sa;
void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
J bars(std::initializer_list<double> prices) {
  J result = J::array();
  int i = 0;
  for (double p : prices) {
    result.push_back(
        {{"time", stamp(parse_time("2025-01-01") + i++ * 86400, true)},
         {"open", p},
         {"high", p + 1},
         {"low", std::max(.5, p - 1)},
         {"close", p},
         {"volume", 1000}});
  }
  return result;
}
double enter_once(const SA_Bar *, int count, double, int64_t, const char *) {
  return count == 1 ? 1 : -1;
}
double round_trip(const SA_Bar *, int count, double, int64_t, const char *) {
  return count == 1 ? 1 : 0;
}
double buy(const SA_Bar *, int, double, int64_t, const char *) { return 1; }
double invalid(const SA_Bar *, int, double, int64_t, const char *) {
  return NAN;
}
int main() {
  try {
    load_config();
    TempDir temp;
    config.db = temp.path / "unit.sqlite3";
    config.mode = "demo";
    initialize();
    J settings = {{"initial_cash", 100},
                  {"commission", 0},
                  {"slippage_bps", 0},
                  {"warmup", 0}};
    auto result =
        simulate(bars({10, 20, 30}), enter_once, J::object(), settings);
    require(result["fills"][0]["time"] == "2025-01-02", "next-open date");
    require(result["fills"][0]["quantity"] == 5, "whole-share sizing");
    require(result["metrics"]["final_equity"] == 150, "mark to close");
    settings.update(
        {{"initial_cash", 1000}, {"commission", 2}, {"slippage_bps", 100}});
    result = simulate(bars({100, 100, 100}), round_trip, J::object(), settings);
    require(result["metrics"]["final_equity"] == 978,
            "fees and slippage accounting");
    require(result["fills"][1]["realized_pnl"] == -22, "realized P&L");
    settings.update({{"initial_cash", 100},
                     {"commission", 0},
                     {"slippage_bps", 0},
                     {"warmup", 1}});
    result = simulate(bars({10, 10, 10}), buy, J::object(), settings);
    require(result["fills"][0]["time"] == "2025-01-03", "warmup suppression");
    require(result["unfilled_final_signal"] == 1, "final signal not filled");
    try {
      simulate(bars({10, 10, 10}), invalid, J::object(), settings);
      throw std::runtime_error("invalid target accepted");
    } catch (const Error &) {
    }
    auto data = bars({10, 10, 10});
    data[1]["time"] = data[0]["time"];
    try {
      simulate(data, buy, J::object(), settings);
      throw std::runtime_error("duplicate date accepted");
    } catch (const Error &) {
    }
    require(stamp(parse_time("1960-01-01"), true) == "1960-01-01",
            "dates before Unix epoch");
    require(parse_time("2026-09-04T11:00:00-04:00") ==
                parse_time("2026-09-04T15:00:00Z"),
            "UTC offsets");
    J c = validate_pair(
          {{"ticker_a", "AAPL"}, {"ticker_b", "MSFT"}, {"lookback", 20}}),
      a = J::array(), b = J::array();
    double end = parse_time(ny_date());
    for (int i = 0; i < 21; i++) {
      auto time = stamp(end - (21 - i) * 86400, true);
      a.push_back(
          {{"time", time},
           {"close", 100 * std::exp(i == 20 ? .1 : (i % 2 ? .01 : -.01))}});
      b.push_back({{"time", time}, {"close", 100}});
    }
    auto snapshot = calculate_pair(c, a, b);
    require(std::abs(snapshot["z_score"].get<double>() - 10) < 1e-10,
            "z-score uses preceding bars only");
    require(snapshot["baseline_end"] == a[19]["time"],
            "baseline excludes current bar");
    J cfg = validate_pair({{"ticker_a", "AAPL"},
                           {"ticker_b", "MSFT"},
                           {"metric", "ratio"},
                           {"condition", "above"},
                           {"threshold", .01},
                           {"repeat", true}});
    auto alert = pair_save(cfg);
    std::thread first(check_pair_alerts), second(check_pair_alerts);
    first.join();
    second.join();
    require(pair_events().size() == 1, "concurrent pair checks deduplicate");
    pair_state(alert["id"], true);
    check_pair_alerts();
    require(pair_events().size() == 2, "rearm increments event version");
    std::cout << "Native accounting, time, pair statistics, and concurrent "
                 "alert tests passed\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
