// Test-only market fixture. Never linked into stockassist-server or
// stockassist-worker.
#include "market.hpp"
namespace sa {
J provider_history(const std::string &, int = 90, const std::string & = "",
                   const std::string & = "1d");
}
#define company_profile provider_company_profile
#define history provider_history
#define search provider_search
#define scan provider_scan
#define quote provider_quote
#include "../../native/src/market.cpp"
#undef company_profile
#undef history
#undef search
#undef scan
#undef quote
#include <numeric>
namespace sa {
namespace {
const std::map<std::string, std::pair<std::string, double>> fixture_companies =
    {{"AAPL", {"Apple Inc.", 218}},
     {"MSFT", {"Microsoft Corporation", 425}},
     {"NVDA", {"NVIDIA Corporation", 132}},
     {"GOOGL", {"Alphabet Inc.", 176}},
     {"AMZN", {"Amazon.com Inc.", 198}},
     {"META", {"Meta Platforms Inc.", 540}},
     {"TSLA", {"Tesla Inc.", 245}},
     {"XLK", {"Test technology hedge", 220}},
     {"MTUM", {"Test momentum hedge", 190}},
     {"VLUE", {"Test value hedge", 110}},
     {"SPY", {"SPDR S&P 500 ETF", 562}}};
}
J history(const std::string &s, int days, const std::string &before,
          const std::string &interval) {
  if (env("STOCKASSIST_TEST_DATA", "1") != "1")
    return provider_history(s, days, before, interval);
  auto r = resolution(interval);
  J bars = J::array();
  bool daily = interval == "1d";

  if (!fixture_companies.count(s))
    throw Error(404, "Ticker not available in the fixture dataset.");
  double base = fixture_companies.at(s).second;
  int seed = std::accumulate(s.begin(), s.end(), 0);
  if (daily) {
    double end = parse_time(before.empty() ? stamp(-1, true) : before);
    for (int offset = 365; offset > 0; --offset) {
      double t = end - offset * 86400;
      time_t seconds = t;
      std::tm tm{};
      gmtime_r(&seconds, &tm);
      if (tm.tm_wday == 0 || tm.tm_wday == 6)
        continue;
      double day = std::floor(t / 86400) + 719163;
      double close = rounded(base * (1 + .06 * std::sin(day / 19 + seed) +
                                     .018 * std::sin(day / 3 + seed)),
                             2),
             open = rounded(close * (1 + .008 * std::sin(day + seed)), 2);
      bars.push_back(
          {{"time", stamp(t, true)},
           {"open", open},
           {"high", rounded(std::max(open, close) * 1.012, 2)},
           {"low", rounded(std::min(open, close) * .988, 2)},
           {"close", close},
           {"volume",
            int64_t(15000000 + 9000000 * (1 + std::sin(day + seed)))}});
    }
    if (bars.size() > size_t(days))
      bars.erase(bars.begin(), bars.end() - days);
  } else {
    double end = before.empty() ? now() : parse_time(before),
           t = std::floor(end / r.seconds) * r.seconds - r.seconds;
    while (bars.size() < 500) {
      auto local = ny_time(t);
      int minute = local.tm_hour * 60 + local.tm_min;
      if (local.tm_wday != 0 && local.tm_wday != 6 && minute >= 570 &&
          minute < 960) {
        double open = rounded(base * (1 + .004 * std::sin(t / 600)), 4),
               close = rounded(open * (1 + .0002 * std::sin(t)), 4);
        bars.push_back({{"time", stamp(t)},
                        {"open", open},
                        {"high", std::max(open, close) + .01},
                        {"low", std::min(open, close) - .01},
                        {"close", close},
                        {"volume", 1000}});
      }
      t -= r.seconds;
    }
    std::reverse(bars.begin(), bars.end());
  }
  if (bars.empty()) {
    if (!before.empty())
      return {{"ticker", s},
              {"source", "fixture"},
              {"bars", bars},
              {"has_more", false},
              {"interval", interval}};
    throw Error(404, "No price history available. Check the symbol and "
                     "market-data permissions.");
  }
  auto &last = bars.back();
  double previous = bars.size() > 1
                        ? bars[bars.size() - 2]["close"].get<double>()
                        : last["open"].get<double>();
  return {
      {"ticker", s},
      {"name", fixture_companies.count(s) ? fixture_companies.at(s).first : s},
      {"source", "fixture"},
      {"as_of", last["time"]},
      {"price", last["close"]},
      {"change_percent",
       rounded((last["close"].get<double>() / previous - 1) * 100, 2)},
      {"bars", bars},
      {"has_more", daily ? bars.size() >= size_t(days) : true},
      {"interval", interval}};
}
J company_profile(const std::string &ticker) {
  if (env("STOCKASSIST_TEST_DATA", "1") != "1")
    return provider_company_profile(ticker);
  if (!fixture_companies.count(ticker))
    throw Error(404, "No test profile available.");
  auto delay = env("STOCKASSIST_TEST_PROFILE_DELAY_MS", "0");
  if (delay != "0")
    std::this_thread::sleep_for(std::chrono::milliseconds(std::stoi(delay)));
  return {
      {"ticker", ticker},         {"name", fixture_companies.at(ticker).first},
      {"industry", "Technology"}, {"category", "Test industry"},
      {"source", "fixture"},      {"fetched_at", stamp()}};
}
J search(const std::string &q, int limit) {
  if (env("STOCKASSIST_TEST_DATA", "1") != "1")
    return provider_search(q, limit);
  J rows = J::array();

  auto lower = [](std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return s;
  };
  for (auto &[s, c] : fixture_companies)
    if (lower(s).find(lower(q)) != std::string::npos ||
        lower(c.first).find(lower(q)) != std::string::npos)
      rows.push_back({{"ticker", s}, {"name", c.first}});
  if (rows.size() > size_t(limit))
    rows.erase(rows.begin() + limit, rows.end());
  return {{"results", rows}, {"source", "fixture"}};
}
J scan(J q) {
  if (env("STOCKASSIST_TEST_DATA", "1") != "1")
    return provider_scan(q);
  J f = {{"scan_code",
          choice(q, "scan_code", "HOT_BY_VOLUME",
                 {"HOT_BY_VOLUME", "TOP_PERC_GAIN", "TOP_PERC_LOSE"})},
         {"min_price", number(q, "min_price", 5, 0, 100000)},
         {"max_price", number(q, "max_price", 1000, 1e-12, 100000)},
         {"min_volume", int64_t(number(q, "min_volume", 100000, 0, 1e9, true))},
         {"limit", int(number(q, "limit", 20, 1, 50, true))}};
  if (f["min_price"] > f["max_price"])
    throw Error(422, "Minimum price cannot exceed maximum price.");
  J rows = J::array();

  std::vector<J> items;
  for (auto &[s, _] : fixture_companies) {
    auto item = history(s, 2);
    if (item["price"] >= f["min_price"] && item["price"] <= f["max_price"] &&
        item["bars"].back()["volume"] >= f["min_volume"])
      items.push_back(item);
  }
  std::sort(items.begin(), items.end(), [&](const J &a, const J &b) {
    double av = f["scan_code"] == "HOT_BY_VOLUME"
                    ? a["bars"].back()["volume"].get<double>()
                    : a["change_percent"].get<double>(),
           bv = f["scan_code"] == "HOT_BY_VOLUME"
                    ? b["bars"].back()["volume"].get<double>()
                    : b["change_percent"].get<double>();
    return f["scan_code"] == "TOP_PERC_LOSE" ? av < bv : av > bv;
  });
  for (size_t i = 0; i < items.size() && i < size_t(f["limit"].get<int>()); i++)
    rows.push_back({{"rank", i + 1},
                    {"ticker", items[i]["ticker"]},
                    {"name", items[i]["name"]},
                    {"exchange", "FIXTURE"}});
  return {{"results", rows},
          {"source", "fixture"},
          {"scan_code", f["scan_code"]},
          {"as_of", stamp()}};
}
J quote(const std::string &ticker) {
  if (env("STOCKASSIST_TEST_DATA", "1") != "1")
    return provider_quote(ticker);
  double price = history(ticker, 2)["price"];
  return {{"ticker", ticker},
          {"source", "fixture"},
          {"market_data_type", 1},
          {"bid", price - .01},
          {"ask", price + .01},
          {"last", price},
          {"bid_size", 100},
          {"ask_size", 100},
          {"last_time", stamp()},
          {"bid_received_at", stamp()},
          {"ask_received_at", stamp()},
          {"received_at", stamp()},
          {"connected", true}};
}
} // namespace sa
