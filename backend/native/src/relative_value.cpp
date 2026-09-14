#include "relative_value.hpp"
#include "../../simulation/relative_value_math.h"
#include "relative_value_config.hpp"
namespace sa {
J analyze_relative_value(const J &c) {
  auto v = c.at("relative_value");
  auto symbols = relative_value_symbols(v);
  std::vector<std::map<std::string, double>> histories;
  std::string source, latest;
  for (auto &ticker : symbols) {
    auto data = history(ticker, 500);
    std::string provider = data.at("source");
    if (!source.empty() && source != provider)
      throw Error(409, "Mixed data providers in relative-value model.");
    source = provider;
    std::map<std::string, double> prices;
    for (auto &bar : data.at("bars"))
      if (bar.at("time").get<std::string>() < ny_date())
        prices[bar["time"]] = bar["close"];
    if (prices.empty())
      throw Error(422, "No completed daily bars for " + ticker);
    auto end = prices.rbegin()->first;
    if (!latest.empty() && end != latest)
      throw Error(409, "Relationship histories have different latest completed dates.");
    latest = end;
    histories.push_back(prices);
  }
  if (parse_time(ny_date()) - parse_time(latest) > 5 * 86400)
    throw Error(409, "Relative-value history is stale.");
  std::vector<std::string> dates;
  std::vector<std::vector<double>> prices(symbols.size());
  for (auto &[date, value] : histories[0]) {
    bool aligned = true;
    for (auto &h : histories)
      if (!h.count(date))
        aligned = false;
    if (!aligned)
      continue;
    dates.push_back(date);
    for (size_t a = 0; a < histories.size(); ++a)
      prices[a].push_back(histories[a].at(date));
  }
  int window = v["lookback"], refs = v["references"].size();
  if (dates.size() < size_t(window + 3))
    throw Error(422, "Not enough aligned history for relative value.");
  int mode = v["category"] == "lead_lag" ? 2 : (v["fit_reference"] == true ? 1 : 0);
  std::vector<double> fixed;
  for (auto &r : v["references"])
    fixed.push_back(r["weight"]);
  J gap = nullptr;
  if (v["category"] == "event") {
    if (v["terms_as_of"].get<std::string>() > dates.back() ||
        v["expires_on"].get<std::string>() < dates.back())
      throw Error(422, "Event terms must predate the estimation window and remain unexpired. "
                       "Update the event window/terms.");
    double fair =
        v["cash_per_share"].get<double>() + v["shares_per_share"].get<double>() * prices[1].back();
    fixed[0] = v["shares_per_share"].get<double>() * prices[1].back() / prices[0].back();
    gap = 100 * (prices[0].back() / fair - 1);
  }
  SA_RVResult fit;
  try {
    fit = sa_rv(prices, refs, window, mode, fixed);
  } catch (const std::exception &e) {
    throw Error(422, e.what());
  }
  bool matched = std::abs(fit.score) >= v["entry_z"].get<double>();
  // Event signals must agree with the contractual payoff gap; no made-up deal feed.
  if (!gap.is_null() && fit.score * gap.get<double>() <= 0)
    matched = false;
  double direction = mode == 2 ? (fit.score > 0 ? 1 : -1) : (fit.score > 0 ? -1 : 1);
  J legs = J::array(), exposures = J::array(), points = J::array();
  std::string directions;
  for (size_t i = 0; i < symbols.size(); ++i) {
    double weight = direction * fit.weights[i];
    if (std::abs(weight) < 1e-8)
      continue;
    auto side = weight > 0 ? "long" : "short";
    legs.push_back(
        {{"ticker", symbols[i]},
         {"side", side},
         {"weight", weight},
         {"role", i == 0 ? "target" : (i <= size_t(refs) ? "reference" : "factor hedge")}});
    if (!directions.empty())
      directions += " · ";
    directions += (weight > 0 ? "Long " : "Short ") + symbols[i];
  }
  for (size_t i = 0; i < fit.before.size(); ++i)
    exposures.push_back({{"ticker", symbols[refs + 1 + i]},
                         {"role", i == 0   ? "market"
                                  : i == 1 ? "sector"
                                           : "factor proxy"},
                         {"before_beta", direction * fit.before[i]},
                         {"after_beta", direction * fit.after[i]}});
  for (size_t i = 0; i < fit.path.size(); ++i)
    points.push_back({{"time", dates[dates.size() - fit.path.size() + i]}, {"value", fit.path[i]}});
  return {{"ticker_a", c["ticker_a"]},
          {"ticker_b", c["ticker_b"]},
          {"metric", "zscore"},
          {"value", fit.score},
          {"z_score", fit.score},
          {"ratio", prices[0].back() / prices[1].back()},
          {"spread", fit.innovation},
          {"mean", 0},
          {"std", fit.sd},
          {"lookback", window},
          {"as_of", dates.back()},
          {"source", source},
          {"price_a", prices[0].back()},
          {"price_b", prices[1].back()},
          {"aligned_bars", dates.size()},
          {"baseline_start", dates[dates.size() - window - 1]},
          {"baseline_end", dates[dates.size() - 2]},
          {"matched", matched},
          {"points", points},
          {"cointegration", nullptr},
          {"signal", matched ? J{{"long", nullptr},
                                 {"short", nullptr},
                                 {"direction_text", directions},
                                 {"legs", legs},
                                 {"reason", mode == 2 ? "One-day-ahead residual forecast"
                                                      : "Exposure-adjusted residual divergence"}}
                             : J()},
          {"relative_value",
           {{"category", v["category"]},
            {"score_type", mode == 2 ? "predicted residual return / forecast noise"
                                     : "cumulative residual z-score"},
            {"hedge_legs", legs},
            {"exposures", exposures},
            {"fit_r2", fit.r2},
            {"reference_coefficients", fit.reference_beta},
            {"event_gap_percent", gap},
            {"neutrality",
             "In-sample neutrality to the selected ETF return proxies, with all hedge legs "
             "included. Future exposures can drift; residuals are not proof of mispricing."}}}};
}
J relative_value_catalog() {
  J result = J::array();
  auto add = [&](std::string category, std::string name, std::string target, J refs,
                 std::string sector, std::string note, bool fitted) {
    J v = {{"category", category},    {"target", target}, {"references", refs},
           {"market", "SPY"},         {"sector", sector}, {"factors", J::array({"MTUM", "VLUE"})},
           {"fit_reference", fitted}, {"lookback", 120},  {"entry_z", 2}};
    result.push_back({{"id", category}, {"name", name}, {"description", note}, {"config", v}});
  };
  add("same_company", "Same-company share classes", "GOOG",
      J::array({{{"ticker", "GOOGL"}, {"weight", 1}}}), "XLC",
      "Listed US share-class relationship; voting rights can sustain a spread. ADR/local "
      "conversion requires FX, share ratios and foreign contract support, which are not available "
      "here.",
      false);
  add("etf", "ETF tracking relative value", "IVV", J::array({{{"ticker", "VOO"}, {"weight", 1}}}),
      "VTI",
      "User-selected funds with similar exposure; fees, distributions and tracking differences "
      "remain. This is not ETF creation/redemption or NAV arbitrage.",
      false);
  add("competitor", "Competitor residual", "V", J::array({{{"ticker", "MA"}, {"weight", 1}}}),
      "XLF",
      "Joint return regression estimates the competitor hedge and market, sector and factor hedge "
      "legs.",
      true);
  add("basket", "Stock versus related basket", "XOM",
      J::array({{{"ticker", "CVX"}, {"weight", 1}},
                {{"ticker", "COP"}, {"weight", 1}},
                {{"ticker", "EOG"}, {"weight", 1}},
                {{"ticker", "OXY"}, {"weight", 1}},
                {{"ticker", "SLB"}, {"weight", 1}}}),
      "XLE",
      "Fit a stock using up to 20 user-selected related names, with separate exposure hedges. "
      "Economic membership is supplied, not inferred from correlation.",
      true);
  add("lead_lag", "Daily residual lead-lag", "MA", J::array({{{"ticker", "V"}, {"weight", 1}}}),
      "XLF",
      "Predict the next daily residual return from lagged factor-adjusted leaders. Daily research "
      "only; no claim of subsecond execution or fast ADR arbitrage.",
      false);
  result.push_back(
      {{"id", "event"},
       {"name", "Event relative value"},
       {"description", "Manual cash-plus-stock event terms, source and validity dates required. No "
                       "automatic merger/news interpretation. Deal failure risk remains."},
       {"config",
        {{"category", "event"},
         {"target", ""},
         {"references", J::array({{{"ticker", ""}, {"weight", 1}}})},
         {"market", "SPY"},
         {"sector", "XLF"},
         {"factors", J::array({"MTUM", "VLUE"})},
         {"fit_reference", false},
         {"lookback", 120},
         {"entry_z", 2},
         {"cash_per_share", 0},
         {"shares_per_share", 1},
         {"terms_as_of", ""},
         {"expires_on", ""},
         {"terms_source", ""}}}});
  add("cross_asset", "Cross-asset ETF proxy residual", "XOM",
      J::array({{{"ticker", "USO"}, {"weight", 1}}}), "XLE",
      "Commodity ETF versus producer, or other US-listed proxies. ETF roll/carry remains; direct "
      "futures calendars, bonds and convertibles need additional contract models.",
      true);
  return result;
}
J relative_value_strategy(const J &c) {
  auto v = c.at("relative_value");
  J params = {{"cost_aware", 1},
              {"cost_cushion", 2},
              {"capture_fraction", .5},
              {"cost_holding_days", 10},
              {"rebalance_band", .05},
              {"max_weight_drift", .15},
              {"max_holding_bars", 20},
              {"lookback", v["lookback"]},
              {"entry_z", v["entry_z"]},
              {"exit_z", std::min(.5, v["entry_z"].get<double>() / 2)},
              {"stop_z", std::max(4.0, v["entry_z"].get<double>() * 2)},
              {"reference_count", v["references"].size()},
              {"mode", v["category"] == "lead_lag" ? 2 : (v["fit_reference"] == true ? 1 : 0)}};
  for (size_t i = 0; i < v["references"].size(); ++i)
    params["reference_weight_" + std::to_string(i)] = v["references"][i]["weight"];
  if (v["category"] == "event") {
    params["event"] = 1;
    params["cash_per_share"] = v["cash_per_share"];
    params["shares_per_share"] = v["shares_per_share"];
    params["terms_from"] = parse_time(v["terms_as_of"]) * 1000;
    params["terms_until"] = parse_time(v["expires_on"]) * 1000 + 86400000;
  }
  return {{"name", "Relative value: " + v["category"].get<std::string>() + " / " +
                       v["target"].get<std::string>()},
          {"language", "cpp"},
          {"code", read(config.root / "simulation/examples/relative_value.cpp")},
          {"params", params},
          {"warmup", v["lookback"].get<int>() + 2},
          {"tickers", relative_value_symbols(v)}};
}
J discover_relationships(const J &relationships) {
  J pairs = J::array(), errors = J::array();
  for (auto &c : relationships)
    try {
      auto snapshot = analyze_relative_value(c);
      pairs.push_back({{"config", c}, {"snapshot", snapshot}, {"correlation", nullptr}});
    } catch (const std::exception &e) {
      errors.push_back({{"ticker", c["ticker_a"]}, {"error", e.what()}});
    }
  return {{"pairs", pairs}, {"errors", errors}};
}
} // namespace sa
