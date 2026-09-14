// Test-only company responses, excluded from the production server.
#include "market.hpp"
#define valuation_company provider_valuation_company
#define fundamentals provider_fundamentals
#define company_news provider_company_news
#include "../../native/src/company.cpp"
#undef valuation_company
#undef fundamentals
#undef company_news
namespace sa {
J valuation_company(const std::string &ticker) {
  if (env("STOCKASSIST_TEST_DATA", "1") != "1")
    return provider_valuation_company(ticker);
  auto h = history(ticker, 2);
  int seed = 0;
  for (auto ch : ticker)
    seed += ch;
  double rev = 1e9, margin = .1 + (seed % 7) * .01,
         growth = .05 + (seed % 5) * .03;
  double pe = ticker == "AAPL" ? std::stod(env("STOCKASSIST_TEST_AAPL_PE", "5"))
                               : 20 + (seed % 3);
  auto year = std::stoi(ny_date().substr(0, 4)) - 1;
  J facts = J::object();
  auto add = [&](std::string tag, std::string unit, double value, bool annual,
                 int y) {
    std::string start = std::to_string(y) + "-01-01",
                end = std::to_string(y) + "-12-31";
    J f = {{"val", value},
           {"end", end},
           {"filed", std::to_string(y + 1) + "-02-15"},
           {"form", "10-K"},
           {"accn", "0000000001-26-000001"}};
    if (annual)
      f["start"] = start;
    auto &rows = facts[tag]["units"][unit];
    if (rows.is_null())
      rows = J::array();
    rows.push_back(f);
    f["val"] = value * 100;
    f["filed"] = "2099-01-01";
    rows.push_back(f);
  };
  add("Revenues", "USD", rev, true, year);
  add("Revenues", "USD", rev / (1 + growth), true, year - 1);
  add("NetIncomeLoss", "USD", rev * margin, true, year);
  add("EarningsPerShareDiluted", "USD/shares", h["price"].get<double>() / pe,
      true, year);
  add("NetCashProvidedByUsedInOperatingActivities", "USD", rev * (margin + .02),
      true, year);
  add("Assets", "USD", rev * 2, false, year);
  if (ticker == "MSFT") {
    add("LiabilitiesCurrent", "USD", rev * .2, false, year);
    add("LiabilitiesNoncurrent", "USD", rev * (.1 + (seed % 4) * .1), false,
        year);
  } else
    add("Liabilities", "USD", rev * (.3 + (seed % 4) * .1), false, year);
  return {{"cik", seed},
          {"entityName", "Test issuer " + ticker},
          {"sic", "3571"},
          {"sic_description", "Test computers"},
          {"issuer_tickers", J::array({ticker})},
          {"source", "fixture"},
          {"facts", {{"us-gaap", facts}}}};
}
J fundamentals(const std::string &ticker) {
  if (env("STOCKASSIST_TEST_DATA", "1") != "1")
    return provider_fundamentals(ticker);

  auto stock = history(ticker, 2);
  return {{"ticker", ticker},
          {"name", stock["name"]},
          {"source", "fixture"},
          {"fetched_at", stamp()},
          {"metrics", J::array({{{"label", "Revenue"},
                                 {"value", 1000000000},
                                 {"unit", "USD"},
                                 {"basis", "Synthetic annual example"},
                                 {"period_end", "2025-12-31"},
                                 {"filed", "2026-02-15"}}})},
          {"notes", J::array({"Synthetic fixture fundamentals; not actual "
                              "company financials."})}};
}
J company_news(const std::string &ticker) {
  if (env("STOCKASSIST_TEST_DATA", "1") != "1")
    return provider_company_news(ticker);

  history(ticker, 2);
  return {{"ticker", ticker},
          {"source", "fixture"},
          {"fetched_at", stamp()},
          {"notes", J::array()},
          {"articles",
           J::array({{{"id", "fixture-" + ticker},
                      {"title",
                       ticker + ": example company announcement (synthetic)"},
                      {"publisher", "Test fixture news"},
                      {"published_at", stamp()},
                      {"url", nullptr}}})}};
}
} // namespace sa
