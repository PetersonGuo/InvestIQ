#include "market.hpp"
#include <curl/curl.h>
namespace sa {
namespace {
size_t receive(char *data, size_t size, size_t count, void *output) {
  auto &body = *static_cast<std::string *>(output);
  if (body.size() + size * count > 24 * 1024 * 1024)
    return 0;
  body.append(data, size * count);
  return size * count;
}
J sec_get(const std::string &url, int ttl = 3600) {
  static std::mutex mutex;
  static std::map<std::string, std::pair<double, J>> cache;
  static std::map<std::string, double> failures;
  static double last = 0;
  std::lock_guard lock(mutex);
  if (cache.count(url) && now() - cache[url].first < ttl)
    return cache[url].second;
  if (failures.count(url) && now() - failures[url] < 300)
    throw Error(503, "SEC data is temporarily unavailable. Retry later.");
  std::this_thread::sleep_for(std::chrono::duration<double>(std::max(0.0, .5 - (now() - last))));
  CURL *handle = curl_easy_init();
  if (!handle)
    throw Error(502, "SEC HTTP client unavailable.");
  auto agent = env("SEC_USER_AGENT", "StockAssist/1.0 personal stock research");
  std::string body;
  curl_easy_setopt(handle, CURLOPT_URL, url.c_str());
  curl_easy_setopt(handle, CURLOPT_USERAGENT, agent.c_str());
  curl_easy_setopt(handle, CURLOPT_TIMEOUT, 10L);
  curl_easy_setopt(handle, CURLOPT_WRITEFUNCTION, receive);
  curl_easy_setopt(handle, CURLOPT_WRITEDATA, &body);
  auto code = curl_easy_perform(handle);
  long status = 0;
  curl_easy_getinfo(handle, CURLINFO_RESPONSE_CODE, &status);
  curl_easy_cleanup(handle);
  last = now();
  if (code != CURLE_OK || status != 200) {
    failures[url] = now();
    throw Error(503, "SEC data request failed. Retry later or configure SEC_USER_AGENT with your "
                     "application/contact details.");
  }
  auto result = J::parse(body);
  if (cache.size() >= 32)
    cache.clear();
  cache[url] = {now(), result};
  return result;
}
std::string cik_for(const std::string &ticker) {
  static const std::map<std::string, std::string> known = {
      {"AAPL", "0000320193"},  {"MSFT", "0000789019"}, {"NVDA", "0001045810"},
      {"GOOGL", "0001652044"}, {"GOOG", "0001652044"}, {"AMZN", "0001018724"},
      {"META", "0001326801"},  {"TSLA", "0001318605"}};
  if (known.count(ticker))
    return known.at(ticker);
  auto directory = sec_get("https://www.sec.gov/files/company_tickers.json", 86400);
  for (auto &[_, entry] : directory.items()) {
    std::string name = entry.value("ticker", "");
    std::replace(name.begin(), name.end(), '-', '.');
    if (name == ticker) {
      auto cik = std::to_string(entry.at("cik_str").get<int64_t>());
      return std::string(10 - cik.size(), '0') + cik;
    }
  }
  throw Error(
      404,
      "No SEC company mapping found. Funds and non-US issuers may not report these fundamentals.");
}
bool safe_url(const J &url) {
  return url.is_string() && (url.get<std::string>().starts_with("https://") ||
                             url.get<std::string>().starts_with("http://"));
}
} // namespace
J parse_company_facts(const J &company) {
  struct Metric {
    const char *name, *unit;
    bool annual;
    std::vector<std::string> tags;
  };
  const std::vector<Metric> metrics = {
      {"Revenue",
       "USD",
       true,
       {"RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"}},
      {"Net income", "USD", true, {"NetIncomeLoss", "ProfitLoss"}},
      {"Diluted EPS", "USD/shares", true, {"EarningsPerShareDiluted"}},
      {"Operating cash flow", "USD", true, {"NetCashProvidedByUsedInOperatingActivities"}},
      {"Total assets", "USD", false, {"Assets"}},
      {"Total liabilities", "USD", false, {"Liabilities"}},
      {"Shareholders’ equity",
       "USD",
       false,
       {"StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"}},
      {"Cash and equivalents", "USD", false, {"CashAndCashEquivalentsAtCarryingValue"}}};
  J output = J::array();
  auto gaap = company.value("facts", J::object()).value("us-gaap", J::object());
  for (auto &metric : metrics) {
    J best = nullptr;
    for (auto &tag : metric.tags) {
      if (!gaap.contains(tag))
        continue;
      auto units = gaap[tag].value("units", J::object());
      if (!units.contains(metric.unit))
        continue;
      for (auto &fact : units[metric.unit]) {
        auto form = fact.value("form", "");
        if (form != "10-K" && form != "10-K/A" && form != "10-Q" && form != "10-Q/A")
          continue;
        if (!fact.contains("val") || !fact["val"].is_number() || !fact.contains("end") ||
            !fact.contains("filed"))
          continue;
        if (metric.annual) {
          if (!fact.contains("start"))
            continue;
          double days = (parse_time(fact["end"]) - parse_time(fact["start"])) / 86400;
          if (days < 330 || days > 400 || (form != "10-K" && form != "10-K/A"))
            continue;
        } else if (fact.contains("start"))
          continue;
        if (best.is_null() || fact["end"] > best["end"] ||
            (fact["end"] == best["end"] && fact["filed"] > best["filed"]))
          best = fact;
      }
    }
    J row = {{"label", metric.name},
             {"unit", metric.unit},
             {"basis", metric.annual ? "Annual" : "Point in time"},
             {"value", nullptr}};
    if (!best.is_null()) {
      row["value"] = best["val"];
      row["period_end"] = best["end"];
      row["filed"] = best["filed"];
      row["period_start"] = best.value("start", "");
      row["form"] = best["form"];
      std::string accession = best.value("accn", "");
      std::string compact = accession;
      compact.erase(std::remove(compact.begin(), compact.end(), '-'), compact.end());
      if (!compact.empty() && compact.find_first_not_of("0123456789") == std::string::npos)
        row["filing_url"] = "https://www.sec.gov/Archives/edgar/data/" +
                            std::to_string(company.at("cik").get<int64_t>()) + "/" + compact + "/" +
                            accession + "-index.html";
    }
    output.push_back(row);
  }
  return {{"name", company.value("entityName", "")}, {"metrics", output}};
}
J fundamentals(const std::string &ticker) {
  if (config.mode == "demo") {
    auto stock = history(ticker, 2);
    return {{"ticker", ticker},
            {"name", stock["name"]},
            {"source", "demo"},
            {"fetched_at", stamp()},
            {"metrics", J::array({{{"label", "Revenue"},
                                   {"value", 1000000000},
                                   {"unit", "USD"},
                                   {"basis", "Synthetic annual example"},
                                   {"period_end", "2025-12-31"},
                                   {"filed", "2026-02-15"}}})},
            {"notes", J::array({"Synthetic demo fundamentals; not actual company financials."})}};
  }
  J result = {{"ticker", ticker},
              {"source", "sec"},
              {"fetched_at", stamp()},
              {"metrics", J::array()},
              {"notes", J::array()}};
  try {
    auto cik = cik_for(ticker);
    result.update(parse_company_facts(
        sec_get("https://data.sec.gov/api/xbrl/companyfacts/CIK" + cik + ".json")));
    result["notes"].push_back(
        "Income and cash flow figures use the latest reported annual period. Balance-sheet figures "
        "use the latest reported date. Periods may differ; these are not live or TTM estimates.");
  } catch (const std::exception &e) {
    result["notes"].push_back(e.what());
  }
  // Optional licensed enrichment works independently of the selected price provider.
  if (!env("MASSIVE_API_KEY", env("POLYGON_API_KEY")).empty()) {
    try {
      auto overview = massive_data("/v3/reference/tickers/" + ticker).value("results", J::object());
      for (const auto &field : {"name", "description", "sic_description", "total_employees",
                                "market_cap", "homepage_url"})
        if (overview.contains(field))
          result[field] = overview[field];
      result["overview_source"] = "massive";
      auto ratios = massive_data("/stocks/financials/v1/ratios?ticker=" + ticker + "&limit=1")
                        .value("results", J::array());
      if (!ratios.empty())
        result["ratios"] = ratios[0];
    } catch (...) {
      result["notes"].push_back(
          "Optional Massive profile/valuation data is unavailable. Check its API key and plan.");
    }
  }
  return result;
}
J company_news(const std::string &ticker) {
  if (config.mode == "demo") {
    history(ticker, 2);
    return {
        {"ticker", ticker},
        {"source", "demo"},
        {"fetched_at", stamp()},
        {"notes", J::array()},
        {"articles", J::array({{{"id", "demo-" + ticker},
                                {"title", ticker + ": example company announcement (synthetic)"},
                                {"publisher", "Demo news"},
                                {"published_at", stamp()},
                                {"url", nullptr}}})}};
  }
  J notes = J::array();
  if (config.mode == "ibkr") {
    try {
      return {{"ticker", ticker},
              {"source", "ibkr"},
              {"fetched_at", stamp()},
              {"articles", ib_request("news", {{"symbol", ticker}})},
              {"notes", notes}};
    } catch (const std::exception &e) {
      notes.push_back(e.what());
    }
  }
  try {
    auto rows = massive_data("/v2/reference/news?ticker=" + ticker +
                             "&limit=20&order=desc&sort=published_utc")
                    .value("results", J::array());
    J articles = J::array();
    for (auto &row : rows)
      articles.push_back(
          {{"id", row.value("id", "")},
           {"title", row.value("title", "")},
           {"published_at", row.value("published_utc", "")},
           {"publisher", row.value("publisher", J::object()).value("name", "")},
           {"url", row.contains("article_url") && safe_url(row["article_url"]) ? row["article_url"]
                                                                               : J(nullptr)}});
    return {{"ticker", ticker},
            {"source", "massive"},
            {"fetched_at", stamp()},
            {"articles", articles},
            {"notes", notes}};
  } catch (...) {
    notes.push_back("News is unavailable. Enable an IBKR API news provider or configure a valid "
                    "Massive API key with news access.");
  }
  return {{"ticker", ticker},
          {"source", "unavailable"},
          {"fetched_at", stamp()},
          {"articles", J::array()},
          {"notes", notes}};
}
} // namespace sa
