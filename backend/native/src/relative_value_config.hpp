#pragma once
#include "common.hpp"
#include <set>
namespace sa {
inline J validate_relative_value(const J &j) {
  if (!j.is_object())
    throw Error(422, "relative_value must be an object.");
  auto category =
      choice(j, "category", "competitor",
             {"same_company", "etf", "competitor", "basket", "lead_lag", "event", "cross_asset"});
  J v = {{"category", category},
         {"target", symbol(text(j, "target"))},
         {"market", symbol(text(j, "market", "SPY"))},
         {"sector", symbol(text(j, "sector"))},
         {"fit_reference",
          boolean(j, "fit_reference", category == "competitor" || category == "basket")},
         {"lookback", int(number(j, "lookback", 120, 40, 250, true))},
         {"entry_z", number(j, "entry_z", 2, .1, 20)},
         {"interval", choice(j, "interval", "1d", {"1d"})}};
  auto instrument = choice(j, "instrument_scope", "us_stocks_etfs", {"us_stocks_etfs"});
  v["instrument_scope"] = instrument;
  if (!j.contains("references") || !j["references"].is_array() || j["references"].empty() ||
      j["references"].size() > 20)
    throw Error(422, "Supply 1–20 reference stocks/ETFs.");
  std::set<std::string> unique = {v["target"].get<std::string>()};
  v["references"] = J::array();
  for (auto &r : j["references"]) {
    auto ticker = symbol(text(r, "ticker"));
    if (!unique.insert(ticker).second)
      throw Error(422, "Target, references and factor hedge instruments must be distinct.");
    v["references"].push_back({{"ticker", ticker}, {"weight", number(r, "weight", 1, -20, 20)}});
  }
  v["factors"] = J::array();
  if (!j.contains("factors") || !j["factors"].is_array() || j["factors"].empty() ||
      j["factors"].size() > 6)
    throw Error(422, "Supply 1–6 additional factor ETF proxies.");
  for (auto &f : j["factors"]) {
    if (!f.is_string())
      throw Error(422, "Factors must be ticker strings.");
    v["factors"].push_back(symbol(f.get<std::string>()));
  }
  for (auto f : {v["market"], v["sector"]})
    if (!unique.insert(f.get<std::string>()).second)
      throw Error(422, "Market and sector hedges must be distinct from the traded relationship.");
  for (auto &f : v["factors"])
    if (!unique.insert(f.get<std::string>()).second)
      throw Error(422, "Duplicate factor or hedge ticker.");
  if (v["lookback"].get<int>() <
      5 * (int(v["factors"].size()) + 3 +
           (v["fit_reference"] == true ? int(v["references"].size()) : 0)))
    throw Error(
        422, "Increase lookback: at least five observations per fitted coefficient are required.");
  if (category == "event") {
    if (v["references"].size() != 1)
      throw Error(422, "Event terms require one acquirer/reference.");
    v["fit_reference"] = false;
    v["cash_per_share"] = number(j, "cash_per_share", 0, 0, 1e6);
    v["shares_per_share"] = number(j, "shares_per_share", 1, 0, 10000);
    if (v["cash_per_share"].get<double>() == 0 && v["shares_per_share"].get<double>() == 0)
      throw Error(422, "Event consideration must be positive.");
    auto asof = text(j, "terms_as_of"), expiry = text(j, "expires_on");
    if (asof.size() != 10 || expiry.size() != 10 || parse_time(asof) >= parse_time(expiry))
      throw Error(422, "Supply valid terms_as_of and expires_on dates (YYYY-MM-DD).");
    v["terms_as_of"] = asof;
    v["expires_on"] = expiry;
    v["terms_source"] = text(j, "terms_source", "", 2000);
    if (v["terms_source"].get<std::string>().empty())
      throw Error(422, "Record the source of the event terms.");
  }
  return {{"relative_value", v},
          {"ticker_a", v["target"]},
          {"ticker_b", v["references"][0]["ticker"]},
          {"lookback", v["lookback"]},
          {"threshold", v["entry_z"]},
          {"metric", "zscore"},
          {"condition", "outside"},
          {"hedge_ratio", 1},
          {"repeat", true}};
}
inline std::vector<std::string> relative_value_symbols(const J &v) {
  std::vector<std::string> out = {v["target"].get<std::string>()};
  for (auto &r : v["references"])
    out.push_back(r["ticker"].get<std::string>());
  out.push_back(v["market"].get<std::string>());
  out.push_back(v["sector"].get<std::string>());
  for (auto &f : v["factors"])
    out.push_back(f.get<std::string>());
  return out;
}
} // namespace sa
