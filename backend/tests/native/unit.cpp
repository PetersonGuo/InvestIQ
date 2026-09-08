#include "pair_math.h"
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
int paired_once(const SA_Asset *, int, int count, double, double, const char *,
                double *weights) {
  if (count != 1)
    return 0;
  weights[0] = .5;
  weights[1] = -.5;
  return 1;
}
int reverse_pair(const SA_Asset *, int, int count, double, double, const char *,
                 double *weights) {
  if (count > 2)
    return 0;
  weights[0] = count == 1 ? .5 : -.5;
  weights[1] = -weights[0];
  return 1;
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
    const double at = parse_time("2026-09-08T15:00:00Z");
    J quote = {{"connected", true},
               {"market_data_type", 1},
               {"bid", 100},
               {"ask", 101},
               {"ask_received_at", stamp(at)},
               {"last_time", stamp(at)}};
    validate_execution_quote(quote, "buy", at);
    for (const auto &change :
         std::vector<J>{{{"market_data_type", 3}},
                        {{"connected", false}},
                        {{"bid", 102}},
                        {{"ask_received_at", stamp(at - 16)}},
                        {{"last_time", stamp(at - 61)}},
                        {{"last_time", stamp(at + 30)}},
                        {{"error", "No subscription"}}}) {
      J invalid = quote;
      invalid.update(change);
      bool rejected = false;
      try {
        validate_execution_quote(invalid, "buy", at);
      } catch (const Error &) {
        rejected = true;
      }
      require(rejected, "Invalid live quotes never fill paper orders");
    }
    J annual = {{"start", "2024-01-01"},
                {"end", "2024-12-31"},
                {"filed", "2025-02-01"},
                {"form", "10-K"},
                {"val", 100},
                {"accn", "0000000001-25-000001"}};
    J amended = annual;
    amended["filed"] = "2025-03-01";
    amended["val"] = 110;
    J quarter = annual;
    quarter["start"] = "2025-01-01";
    quarter["end"] = "2025-03-31";
    quarter["filed"] = "2025-05-01";
    quarter["form"] = "10-Q";
    quarter["val"] = 30;
    J company = {
        {"entityName", "Example"},
        {"cik", 1},
        {"facts",
         {{"us-gaap",
           {{"Revenues",
             {{"units", {{"USD", J::array({quarter, annual, amended})}}}}}}}}}};
    auto financials = parse_company_facts(company);
    require(
        financials["metrics"][0]["value"] == 110,
        "Annual fundamentals exclude quarters and prefer latest amendments");
    require(financials["metrics"][0]["period_end"] == "2024-12-31",
            "Fundamentals retain original reporting period");
    require(financials["metrics"][1]["value"].is_null(),
            "Missing fundamentals are not zero");
    {
      J first = bars({100, 110, 120}), second = bars({100, 90, 80});
      // Signals on the first close fill at the following open, not the signal
      // close.
      first[1]["open"] = 100;
      first[1]["low"] = 99;
      second[1]["open"] = 100;
      second[1]["high"] = 101;
      first[2]["open"] = 110;
      first[2]["low"] = 109;
      second[2]["open"] = 90;
      second[2]["high"] = 91;
      J assets = J::array({{{"ticker", "A"}, {"bars", first}},
                           {{"ticker", "B"}, {"bars", second}}});
      J settings = {{"initial_cash", 10000},  {"commission", 0},
                    {"slippage_bps", 0},      {"warmup", 0},
                    {"interval", "1d"},       {"borrow_rate_percent", 0},
                    {"max_gross_exposure", 1}};
      auto multi =
          simulate_portfolio(assets, paired_once, J::object(), settings);
      require(multi["metrics"]["final_equity"] == 12000,
              "Long/short portfolio shares a cash ledger");
      require(multi["fills"].size() == 2 && multi["fills"][1]["side"] == "sell",
              "Short entries are recorded per stock");
      auto reversed =
          simulate_portfolio(assets, reverse_pair, J::object(), settings);
      require(reversed["metrics"]["final_equity"] == 9890,
              "Reversals close existing legs and open the opposite positions");
      require(reversed["fills"][2]["realized_pnl"] == 500 &&
                  reversed["fills"][3]["realized_pnl"] == 500,
              "Short and long realized P&L account correctly on reversal");
      settings["borrow_rate_percent"] = 36.5;
      auto borrow_result =
          simulate_portfolio(assets, paired_once, J::object(), settings);
      require(
          std::abs(borrow_result["metrics"]["total_borrow_cost"].get<double>() -
                   4.5) < 1e-8,
          "Borrow costs use elapsed calendar time and prior short value");
      settings["initial_cash"] = 1e8;
      settings["slippage_bps"] = 50;
      settings["commission"] = 1;
      auto capped_one =
          simulate_portfolio(assets, paired_once, J::object(), settings);
      settings["max_gross_exposure"] = 2;
      auto capped_two =
          simulate_portfolio(assets, paired_once, J::object(), settings);
      require(capped_one["equity_curve"] == capped_two["equity_curve"],
              "Unused leverage capacity does not change fills");
      std::vector<double> logs_a, logs_b;
      double level = 4, residual = 0;
      for (int i = 0; i < 80; ++i) {
        level += .01 * std::sin(i * .7) + .008 * std::cos(i * 1.9);
        residual = .35 * residual + .006 * std::sin(i * 2.3);
        logs_b.push_back(level);
        logs_a.push_back(1.4 * level + .2 + residual);
      }
      auto fit = sa_pair_fit(logs_a, logs_b);
      require(fit.test_valid && std::abs(fit.beta - 1.4006318233159136) < 1e-10,
              "Pair OLS matches independent least squares");
      require(std::abs(fit.adf - (-209.20162319666125)) < 1e-6,
              "Engle–Granger statistic matches independent matrix regression");
      require(std::abs(fit.critical - (-3.4145662922608553)) < 1e-10,
              "Cointegration critical values use N=2 sample adjustment");
    }
    std::cout << "Native accounting, time, pair statistics, and concurrent "
                 "alert tests passed\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
