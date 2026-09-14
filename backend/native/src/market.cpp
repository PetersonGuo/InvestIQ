#include "market.hpp"
#include <curl/curl.h>
namespace sa {
namespace {
const std::map<std::string, std::string> companies = {
    {"AAPL", "Apple Inc."},     {"MSFT", "Microsoft Corporation"}, {"NVDA", "NVIDIA Corporation"},
    {"GOOGL", "Alphabet Inc."}, {"AMZN", "Amazon.com Inc."},       {"META", "Meta Platforms Inc."},
    {"TSLA", "Tesla Inc."},     {"SPY", "SPDR S&P 500 ETF"}};
size_t append(char *p, size_t size, size_t n, void *out) {
  auto &s = *static_cast<std::string *>(out);
  if (s.size() + size * n > 64 * 1024 * 1024)
    return 0;
  s.append(p, size * n);
  return size * n;
}
J provider_get(const std::string &path) {
  static std::mutex m;
  static std::map<std::string, std::pair<double, J>> cache;
  {
    std::lock_guard lock(m);
    if (cache.count(path) && now() - cache[path].first < 60)
      return cache[path].second;
  }
  auto key = env("MASSIVE_API_KEY", env("POLYGON_API_KEY"));
  if (key.empty())
    throw Error(503, "Set MASSIVE_API_KEY to use provider data.");
  CURL *c = curl_easy_init();
  if (!c)
    throw Error(502, "Market data HTTP client unavailable.");
  std::string response, url = "https://api.massive.com" + path,
                        auth = "Authorization: Bearer " + key;
  curl_slist *headers = curl_slist_append(nullptr, auth.c_str());
  curl_easy_setopt(c, CURLOPT_URL, url.c_str());
  curl_easy_setopt(c, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(c, CURLOPT_TIMEOUT, 10L);
  curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, append);
  curl_easy_setopt(c, CURLOPT_WRITEDATA, &response);
  auto code = curl_easy_perform(c);
  long status;
  curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &status);
  curl_slist_free_all(headers);
  curl_easy_cleanup(c);
  if (code != CURLE_OK || status >= 400)
    throw Error(502, "Market data provider unavailable. Check your API key, "
                     "plan, and rate limit.");
  auto data = J::parse(response);
  {
    std::lock_guard lock(m);
    if (cache.size() > 256)
      cache.clear();
    cache[path] = {now(), data};
  }
  return data;
}
std::string escape(const std::string &s) {
  CURL *c = curl_easy_init();
  char *e = curl_easy_escape(c, s.c_str(), s.size());
  std::string out = e ? e : "";
  curl_free(e);
  curl_easy_cleanup(c);
  return out;
}
} // namespace
J massive_data(const std::string &path) { return provider_get(path); }
J company_profile(const std::string &ticker) {
  if (config.mode == "ibkr")
    return ib_request("profile", {{"symbol", ticker}});
  auto data = massive_data("/v3/reference/tickers/" + ticker).value("results", J::object());
  return {{"ticker", ticker},
          {"name", data.value("name", ticker)},
          {"industry", data.value("sic_description", "")},
          {"category", data.value("sic_description", "")},
          {"source", "massive"},
          {"fetched_at", stamp()}};
}
J history(const std::string &s, int days, const std::string &before, const std::string &interval) {
  auto r = resolution(interval);
  J bars = J::array();
  bool daily = interval == "1d";
  if (config.mode == "ibkr") {
    bars = ib_request("history", {{"symbol", s}, {"interval", interval}, {"before", before}});
    if (daily && bars.size() > size_t(days))
      bars.erase(bars.begin(), bars.end() - days);
  } else {
    if (!daily)
      throw Error(422, "Intraday data currently requires IBKR mode.");
    double end = parse_time(before.empty() ? stamp(-1, true) : before);
    auto data = provider_get("/v2/aggs/ticker/" + s + "/range/1/day/" +
                             stamp(end - (days * 2 + 10) * 86400, true) + "/" +
                             stamp(end - 86400, true) + "?adjusted=true&sort=asc&limit=50000");
    for (auto &b : data.value("results", J::array()))
      bars.push_back({{"time", stamp(b["t"].get<double>() / 1000, true)},
                      {"open", b["o"]},
                      {"high", b["h"]},
                      {"low", b["l"]},
                      {"close", b["c"]},
                      {"volume", b["v"]}});
    if (bars.size() > size_t(days))
      bars.erase(bars.begin(), bars.end() - days);
  }
  if (bars.empty()) {
    if (!before.empty())
      return {{"ticker", s},
              {"source", config.mode},
              {"bars", bars},
              {"has_more", false},
              {"interval", interval}};
    throw Error(404, "No price history available. Check the symbol and "
                     "market-data permissions.");
  }
  auto &last = bars.back();
  double previous =
      bars.size() > 1 ? bars[bars.size() - 2]["close"].get<double>() : last["open"].get<double>();
  return {{"ticker", s},
          {"name", companies.count(s) ? companies.at(s) : s},
          {"source", config.mode},
          {"as_of", last["time"]},
          {"price", last["close"]},
          {"change_percent", rounded((last["close"].get<double>() / previous - 1) * 100, 2)},
          {"bars", bars},
          {"has_more", daily ? bars.size() >= size_t(days) : true},
          {"interval", interval}};
}
J search(const std::string &q, int limit) {
  J rows = J::array();
  if (config.mode == "ibkr") {
    rows = ib_request("search", q.empty() ? "A" : q);
  } else {
    auto data = provider_get("/v3/reference/tickers?market=stocks&active=true&limit=" +
                             std::to_string(limit) + "&sort=ticker&search=" + escape(q));
    for (auto &r : data.value("results", J::array()))
      rows.push_back(
          {{"ticker", r["ticker"]}, {"name", r.value("name", r["ticker"].get<std::string>())}});
  }
  if (rows.size() > size_t(limit))
    rows.erase(rows.begin() + limit, rows.end());
  return {{"results", rows}, {"source", config.mode}};
}
J scan(J q) {
  J f = {{"scan_code", choice(q, "scan_code", "HOT_BY_VOLUME",
                              {"HOT_BY_VOLUME", "TOP_PERC_GAIN", "TOP_PERC_LOSE"})},
         {"min_price", number(q, "min_price", 5, 0, 100000)},
         {"max_price", number(q, "max_price", 1000, 1e-12, 100000)},
         {"min_volume", int64_t(number(q, "min_volume", 100000, 0, 1e9, true))},
         {"limit", int(number(q, "limit", 20, 1, 50, true))}};
  if (f["min_price"] > f["max_price"])
    throw Error(422, "Minimum price cannot exceed maximum price.");
  J rows = J::array();
  if (config.mode == "ibkr")
    rows = ib_request("scanner", f);
  else
    throw Error(422, "Market scanning requires IBKR mode.");
  return {{"results", rows},
          {"source", config.mode},
          {"scan_code", f["scan_code"]},
          {"as_of", stamp()}};
}
J ticks(const std::string &s, const std::string &before) {
  if (config.mode != "ibkr")
    throw Error(422, "Individual trade records require IBKR mode.");
  auto rows = ib_request("ticks", {{"symbol", s}, {"interval", "1s"}, {"before", before}});
  return {{"ticker", s},          {"source", "ibkr"},
          {"ticks", rows},        {"timestamp_resolution", "1 second"},
          {"session", "regular"}, {"as_of", rows.empty() ? J(nullptr) : rows.back()["time"]}};
}
J quote(const std::string &ticker) {
  if (config.mode == "ibkr")
    return ib_request("quote", {{"symbol", ticker}});
  throw Error(422, "Live quotes currently require IBKR mode.");
}
} // namespace sa
