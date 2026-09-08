#include "pairs.hpp"
#include "../../simulation/pair_math.h"
#include <numeric>
#include <set>
namespace sa {
namespace {
std::pair<double, double> stats(const std::vector<double> &v, size_t start, size_t end) {
  long double sum = 0;
  for (size_t i = start; i < end; i++)
    sum += v[i];
  double mean = sum / (end - start);
  long double variance = 0;
  for (size_t i = start; i < end; i++)
    variance += (v[i] - mean) * (v[i] - mean);
  return {mean, std::sqrt(variance / (end - start))};
}
J decode(J row) {
  auto c = J::parse(row["config_json"].get<std::string>());
  row["latest"] =
      row["latest_json"].is_null() ? J(nullptr) : J::parse(row["latest_json"].get<std::string>());
  row.erase("latest_json");
  row.erase("config_json");
  row.update(c);
  return row;
}
} // namespace
bool matches(double value, const J &c) {
  double t = c["threshold"];
  std::string condition = c["condition"];
  if (condition == "above")
    return value >= t;
  if (condition == "below")
    return value <= t;
  if (condition == "outside")
    return std::abs(value) >= t;
  return std::abs(value) <= t;
}
J calculate_pair(const J &c, const J &a_rows, const J &b_rows, const std::string &today) {
  std::map<std::string, double> a, b;
  for (auto &bar : a_rows)
    a[bar["time"]] = bar["close"];
  for (auto &bar : b_rows)
    b[bar["time"]] = bar["close"];
  if (a.empty() || b.empty())
    throw Error(422, "Both stocks need completed daily price history.");
  if (a.rbegin()->first != b.rbegin()->first)
    throw Error(409, "Latest bar dates differ. Waiting for aligned data.");
  std::vector<std::string> dates;
  std::vector<double> ratios, spreads, log_a, log_b;
  double hedge = c["hedge_ratio"];
  for (auto &[day, price] : a)
    if (b.count(day)) {
      if (!std::isfinite(price) || !std::isfinite(b[day]) || price <= 0 || b[day] <= 0)
        throw Error(502, "Pair history contains invalid prices.");
      dates.push_back(day);
      log_a.push_back(std::log(price));
      log_b.push_back(std::log(b[day]));
      ratios.push_back(price / b[day]);
      spreads.push_back(std::log(price) - hedge * std::log(b[day]));
    }
  auto current = today.empty() ? ny_date() : today;
  double age = (parse_time(current) - parse_time(dates.back())) / 86400;
  if (age <= 0 || age > 7)
    throw Error(409, "Pair data must contain completed closes no more than "
                     "seven calendar days old. Waiting for fresh history.");
  int lookback = c["lookback"];
  bool z = c["metric"] == "zscore";
  size_t required = z ? lookback + 1 : 2;
  if (dates.size() < required)
    throw Error(422, "Need " + std::to_string(required) + " aligned trading dates; only " +
                         std::to_string(dates.size()) + " are available.");
  J mean = nullptr, stddev = nullptr, zscore = nullptr, points = J::array();
  double value;
  if (z) {
    auto [m, s] = stats(spreads, spreads.size() - lookback - 1, spreads.size() - 1);
    if (s <= 1e-12)
      throw Error(422, "The spread has no measurable variation in this window; its "
                       "z-score is undefined. Try a different pair or lookback.");
    mean = m;
    stddev = s;
    value = (spreads.back() - m) / s;
    zscore = value;
    for (size_t i = std::max(lookback, int(dates.size()) - 90); i < dates.size(); i++) {
      auto [base, sd] = stats(spreads, i - lookback, i);
      if (sd > 1e-12)
        points.push_back({{"time", dates[i]}, {"value", (spreads[i] - base) / sd}});
    }
  } else {
    value = ratios.back();
    for (size_t i = dates.size() > 90 ? dates.size() - 90 : 0; i < dates.size(); i++)
      points.push_back({{"time", dates[i]}, {"value", ratios[i]}});
  }
  if (!std::isfinite(value))
    throw Error(502, "Pair metric is not finite.");
  J cointegration = nullptr;
  if (dates.size() >= size_t(lookback + 1)) {
    auto fit = sa_pair_fit(std::vector<double>(log_a.end() - lookback - 1, log_a.end() - 1),
                           std::vector<double>(log_b.end() - lookback - 1, log_b.end() - 1));
    cointegration = {{"method", "Engle–Granger, intercept, fixed ADF lag 1"},
                     {"observations", lookback},
                     {"test_statistic", fit.test_valid ? J(fit.adf) : J(nullptr)},
                     {"critical_5_percent", fit.test_valid ? J(fit.critical) : J(nullptr)},
                     {"reject_no_cointegration_5_percent",
                      fit.test_valid ? J(fit.adf < fit.critical) : J(nullptr)},
                     {"fitted_hedge_ratio", fit.valid ? J(fit.beta) : J(nullptr)},
                     {"half_life_bars", fit.half_life > 0 ? J(fit.half_life) : J(nullptr)},
                     {"baseline_end", dates[dates.size() - 2]},
                     {"assumptions",
                      "Assumes I(1) log-price series. No multiple-testing correction. A "
                      "rejection is evidence, not a guarantee of reversion."}};
  }
  J signal = nullptr;
  if (z && matches(value, c) && c["condition"] != "inside" && std::abs(value) > 1e-12)
    signal = {{"long", value > 0 ? c["ticker_b"] : c["ticker_a"]},
              {"short", value > 0 ? c["ticker_a"] : c["ticker_b"]},
              {"reason", "Mean-reversion entry: short the relatively expensive leg and long the "
                         "relatively cheap leg."}};
  if (!signal.is_null()) {
    std::string action_a = value > 0 ? "short" : "long",
                action_b = (value * hedge) > 0 ? "long" : "short";
    signal["legs"] = J::array({{{"ticker", c["ticker_a"]}, {"side", action_a}},
                               {{"ticker", c["ticker_b"]}, {"side", action_b}}});
    signal["direction_text"] =
        (action_a == "long" ? "Long " : "Short ") + c["ticker_a"].get<std::string>() + " / " +
        (action_b == "long" ? "Long " : "Short ") + c["ticker_b"].get<std::string>();
    if (hedge < 0) {
      signal["long"] =
          value < 0 ? J(c["ticker_a"].get<std::string>() + " / " + c["ticker_b"].get<std::string>())
                    : J(nullptr);
      signal["short"] =
          value > 0 ? J(c["ticker_a"].get<std::string>() + " / " + c["ticker_b"].get<std::string>())
                    : J(nullptr);
      signal["reason"] = "Inverse relationship: reversion requires both legs in the same "
                         "direction; this is not market-neutral.";
    }
  }
  return {{"cointegration", cointegration},
          {"deviation",
           z ? (std::abs(value) >= std::abs(c["threshold"].get<double>()) ? "outside_entry_band"
                                                                          : "inside_entry_band")
             : "ratio"},
          {"signal", signal},
          {"ticker_a", c["ticker_a"]},
          {"ticker_b", c["ticker_b"]},
          {"metric", c["metric"]},
          {"value", value},
          {"ratio", ratios.back()},
          {"spread", spreads.back()},
          {"z_score", zscore},
          {"mean", mean},
          {"std", stddev},
          {"hedge_ratio", hedge},
          {"lookback", lookback},
          {"as_of", dates.back()},
          {"price_a", a[dates.back()]},
          {"price_b", b[dates.back()]},
          {"aligned_bars", dates.size()},
          {"baseline_start", z ? J(dates[dates.size() - lookback - 1]) : J(nullptr)},
          {"baseline_end", z ? J(dates[dates.size() - 2]) : J(nullptr)},
          {"matched", matches(value, c)},
          {"points", points}};
}
J analyze_pair(const J &c) {
  auto a = history(c["ticker_a"], 500), b = history(c["ticker_b"], 500);
  if (a["source"] != b["source"])
    throw Error(409, "Both legs must use the same market-data source.");
  auto result = calculate_pair(c, a["bars"], b["bars"]);
  result["source"] = a["source"];
  return result;
}
J discover_pairs(const J &request) {
  if (!request.contains("tickers") || !request["tickers"].is_array() ||
      request["tickers"].size() < 2 || request["tickers"].size() > 20)
    throw Error(422, "Supply between 2 and 20 stock symbols.");
  int lookback = number(request, "lookback", 60, 20, 250, true);
  double minimum = number(request, "min_correlation", .8, 0, 1);
  auto relationship =
      choice(request, "relationship", "positive", {"positive", "negative", "either"});
  bool require_cointegration = boolean(request, "require_cointegration", false);
  double threshold = number(request, "threshold", 2, .1, 20);
  std::map<std::string, J> histories;
  J errors = J::array(), results = J::array();
  std::set<std::string> symbols;
  for (const auto &ticker : request["tickers"]) {
    if (!ticker.is_string())
      throw Error(422, "Stock symbols must be strings.");
    symbols.insert(symbol(ticker.get<std::string>()));
  }
  if (symbols.size() < 2)
    throw Error(422, "Supply at least two distinct symbols.");
  for (const auto &ticker : symbols) {
    try {
      histories[ticker] = history(ticker, 500);
    } catch (const std::exception &e) {
      errors.push_back({{"ticker", ticker}, {"error", e.what()}});
    }
  }
  for (auto a = histories.begin(); a != histories.end(); ++a)
    for (auto b = std::next(a); b != histories.end(); ++b) {
      try {
        if (a->second["source"] != b->second["source"])
          continue;
        std::map<std::string, double> prices;
        for (auto &bar : a->second["bars"])
          prices[bar["time"]] = bar["close"];
        std::vector<double> x, y;
        for (auto &bar : b->second["bars"]) {
          auto day = bar["time"].get<std::string>();
          double price = bar["close"];
          if (prices.count(day) && prices[day] > 0 && price > 0) {
            x.push_back(std::log(prices[day]));
            y.push_back(std::log(price));
          }
        }
        if (x.size() < size_t(lookback + 1))
          throw Error(422, "Insufficient aligned history.");
        size_t start = x.size() - lookback - 1;
        std::vector<double> rx, ry;
        for (size_t i = start + 1; i < x.size(); ++i) {
          rx.push_back(x[i] - x[i - 1]);
          ry.push_back(y[i] - y[i - 1]);
        }
        auto [mx, sx] = stats(rx, 0, rx.size());
        auto [my, sy] = stats(ry, 0, ry.size());
        if (sx <= 1e-12 || sy <= 1e-12)
          continue;
        double covariance = 0;
        for (size_t i = 0; i < rx.size(); ++i)
          covariance += (rx[i] - mx) * (ry[i] - my);
        double correlation = std::clamp(covariance / (lookback * sx * sy), -1.0, 1.0);
        if ((relationship == "positive" && correlation < minimum) ||
            (relationship == "negative" && correlation > -minimum) ||
            (relationship == "either" && std::abs(correlation) < minimum))
          continue;
        auto [px, dx] = stats(x, start, x.size() - 1);
        auto [py, dy] = stats(y, start, y.size() - 1);
        if (dy <= 1e-12)
          continue;
        double cov = 0;
        for (size_t i = start; i < x.size() - 1; ++i)
          cov += (x[i] - px) * (y[i] - py);
        double hedge = cov / (lookback * dy * dy);
        if (std::abs(hedge) < 1e-12 || std::abs(hedge) > 100)
          continue;
        J config = validate_pair({{"ticker_a", a->first},
                                  {"ticker_b", b->first},
                                  {"lookback", lookback},
                                  {"hedge_ratio", hedge},
                                  {"threshold", threshold},
                                  {"repeat", true}});
        J snapshot = calculate_pair(config, a->second["bars"], b->second["bars"]);
        snapshot["source"] = a->second["source"];
        if (require_cointegration &&
            snapshot["cointegration"]["reject_no_cointegration_5_percent"] != true)
          continue;
        results.push_back(
            {{"correlation", correlation}, {"config", config}, {"snapshot", snapshot}});
      } catch (const std::exception &e) {
        errors.push_back({{"ticker", a->first + "/" + b->first}, {"error", e.what()}});
      }
    }
  std::sort(results.begin(), results.end(), [](const J &a, const J &b) {
    return std::abs(a["correlation"].get<double>()) > std::abs(b["correlation"].get<double>());
  });
  return {{"pairs", results},
          {"errors", errors},
          {"lookback", lookback},
          {"correlation_method", "Pearson correlation of aligned daily log returns"}};
}
J pair_list() {
  Db d;
  auto rows = d.rows("SELECT * FROM pair_alerts ORDER BY created_at DESC");
  for (auto &row : rows)
    row = decode(row);
  return rows;
}
J pair_get(const std::string &id) {
  Db d;
  auto rows = d.rows("SELECT * FROM pair_alerts WHERE id=?", {id});
  if (rows.empty())
    throw Error(404, "Pair alert not found.");
  return decode(rows[0]);
}
J pair_save(const J &c, const std::string &existing) {
  if (!existing.empty())
    pair_get(existing);
  auto snapshot = analyze_pair(c);
  auto id = existing.empty() ? sa::id() : existing;
  Db d;
  auto time = stamp();
  if (existing.empty())
    d.exec("INSERT INTO "
           "pair_alerts(id,config_json,created_at,updated_at,latest_json) "
           "VALUES(?,?,?,?,?)",
           {id, c.dump(), time, time, snapshot.dump()});
  else {
    d.exec("UPDATE pair_alerts SET "
           "config_json=?,active=1,version=version+1,updated_at=?,last_checked_"
           "at=NULL,last_as_of=NULL,last_matched=0,latest_json=?,last_error="
           "NULL,triggered_at=NULL WHERE id=?",
           {c.dump(), time, snapshot.dump(), id});
    if (!d.changes())
      throw Error(404, "Pair alert not found.");
  }
  return pair_get(id);
}
J pair_state(const std::string &id, bool active) {
  Db d;
  d.exec("UPDATE pair_alerts SET "
         "active=?,version=version+1,updated_at=?,last_as_of=NULL,last_matched="
         "0,last_error=NULL,triggered_at=NULL WHERE id=?",
         {active, stamp(), id});
  if (!d.changes())
    throw Error(404, "Pair alert not found.");
  return pair_get(id);
}
void pair_delete(const std::string &id) {
  Db d;
  d.exec("DELETE FROM pair_alerts WHERE id=?", {id});
  if (!d.changes())
    throw Error(404, "Pair alert not found.");
}
J pair_events() {
  Db d;
  auto rows = d.rows("SELECT * FROM pair_events ORDER BY created_at DESC LIMIT 50");
  for (auto &row : rows) {
    auto event = J::parse(row["event_json"].get<std::string>());
    row.erase("event_json");
    row.erase("version");
    row.update(event);
  }
  return rows;
}
void check_pair_alerts() {
  Db d;
  auto alerts = d.rows("SELECT * FROM pair_alerts WHERE active=1");
  for (auto &row : alerts) {
    auto c = J::parse(row["config_json"].get<std::string>());
    auto time = stamp();
    J snapshot;
    try {
      snapshot = analyze_pair(c);
    } catch (const std::exception &e) {
      d.exec("UPDATE pair_alerts SET last_checked_at=?,last_error=? WHERE id=? "
             "AND version=? AND active=1",
             {time, std::string(e.what()).substr(0, 1000), row["id"], row["version"]});
      continue;
    }
    Db tx;
    tx.exec("BEGIN IMMEDIATE");
    auto rows = tx.rows("SELECT * FROM pair_alerts WHERE id=? AND version=? AND active=1",
                        {row["id"], row["version"]});
    if (rows.empty()) {
      tx.exec("COMMIT");
      continue;
    }
    auto current = rows[0];
    if (!current["last_as_of"].is_null() && snapshot["as_of"] < current["last_as_of"]) {
      tx.exec("UPDATE pair_alerts SET last_checked_at=?,last_error=? WHERE id=?",
              {time, "Pair data moved backward in time. Waiting for newer closes.", row["id"]});
      tx.exec("COMMIT");
      continue;
    }
    bool new_bar = current["last_as_of"] != snapshot["as_of"],
         fire = new_bar && snapshot["matched"].get<bool>() && !current["last_matched"].get<int>();
    if (fire) {
      J event_snapshot = snapshot;
      event_snapshot.erase("points");
      J event = {{"config", c}, {"snapshot", event_snapshot}};
      tx.exec("INSERT OR IGNORE INTO pair_events VALUES(?,?,?,?,?,?)",
              {id(), row["id"], row["version"], snapshot["as_of"], time, event.dump()});
    }
    int active = fire && !c["repeat"].get<bool>() ? 0 : 1;
    tx.exec("UPDATE pair_alerts SET "
            "active=?,last_checked_at=?,last_as_of=?,last_matched=?,latest_"
            "json=?,last_error=NULL,triggered_at=CASE WHEN ? THEN ? ELSE "
            "triggered_at END WHERE id=? AND version=?",
            {active, time, snapshot["as_of"],
             new_bar ? int(snapshot["matched"].get<bool>()) : current["last_matched"].get<int>(),
             snapshot.dump(), fire, time, row["id"], row["version"]});
    tx.exec("COMMIT");
  }
}
J portfolio() {
  Db d;
  J orders = d.rows("SELECT o.*,d.detail_json FROM orders o LEFT JOIN order_details d ON d.id=o.id "
                    "ORDER BY o.created_at DESC LIMIT 50");
  for (auto &order : orders) {
    J metadata = order["detail_json"];
    order.erase("detail_json");
    if (!metadata.is_null())
      order.update(J::parse(metadata.get<std::string>()));
  }
  return {{"cash_cents", d.rows("SELECT cash_cents FROM account WHERE id=1")[0]["cash_cents"]},
          {"positions", d.rows("SELECT * FROM positions ORDER BY ticker")},
          {"orders", orders}};
}
void validate_execution_quote(const J &quote, const std::string &side, double at) {
  auto t = ny_time(at);
  int minute = t.tm_hour * 60 + t.tm_min;
  if (config.mode != "demo" && (t.tm_wday == 0 || t.tm_wday == 6 || minute < 570 || minute >= 960))
    throw Error(
        409,
        "Paper execution is available during regular US trading hours (09:30–16:00 New York).");
  if (!quote.value("connected", false) || quote.value("market_data_type", 0) != 1 ||
      quote.contains("error"))
    throw Error(
        409,
        "A live IBKR quote is required; delayed/frozen or unavailable quotes cannot fill orders.");
  std::string key = side == "buy" ? "ask" : "bid";
  if (!quote.contains("bid") || !quote.contains("ask") || quote["bid"].get<double>() <= 0 ||
      quote["ask"].get<double>() <= 0 || quote["bid"].get<double>() > quote["ask"].get<double>() ||
      !quote.contains(key + "_received_at") || at - parse_time(quote[key + "_received_at"]) > 15 ||
      !quote.contains("last_time") || at - parse_time(quote["last_time"]) > 60 ||
      parse_time(quote["last_time"]) > at + 5 || parse_time(quote[key + "_received_at"]) > at + 5)
    throw Error(
        409,
        "Waiting for a fresh, uncrossed live quote and a recent trade. No paper fill was made.");
}
J place_order(const J &q) {
  auto ticker = symbol(text(q, "ticker"));
  auto side = choice(q, "side", "", {"buy", "sell"});
  int64_t quantity = number(q, "quantity", 0, 1, 1000000, true);
  auto type = choice(q, "order_type", "market", {"market", "limit"});
  double limit = type == "limit" ? number(q, "limit_price", 0, .01, 1e10) : 0;
  auto request_id = text(q, "request_id", "", 100);
  Db d;
  auto replay = [&]() -> J {
    if (request_id.empty())
      return nullptr;
    auto rows = d.rows("SELECT * FROM order_details WHERE request_id=?", {request_id});
    if (rows.empty())
      return nullptr;
    if (J::parse(rows[0]["request_json"].get<std::string>()) != q)
      throw Error(409, "This order request ID was already used for different details.");
    return J::parse(rows[0]["detail_json"].get<std::string>());
  };
  auto previous = replay();
  if (!previous.is_null())
    return previous;
  auto quote = sa::quote(ticker);
  validate_execution_quote(quote, side, now());
  std::string key = side == "buy" ? "ask" : "bid";
  double raw_price = quote[key];
  if (!std::isfinite(raw_price) || raw_price <= 0 || raw_price > 1e10)
    throw Error(422, "Invalid execution quote.");
  // Round against the order so a limit can never execute beyond its price.
  int64_t price =
      side == "buy" ? std::ceil(raw_price * 100 - 1e-8) : std::floor(raw_price * 100 + 1e-8);
  if (type == "limit" && (side == "buy" ? price / 100.0 > limit : price / 100.0 < limit))
    throw Error(409, "Limit does not cross the current quote. Immediate-or-cancel order cancelled "
                     "without a fill.");
  int64_t requested = quantity;
  int64_t available = std::min(1000000.0, std::floor(quote.value(key + "_size", 0.0)));
  std::string liquidity_key =
      ticker + ":" + key + ":" + quote[key + "_received_at"].get<std::string>();
  d.exec("BEGIN IMMEDIATE");
  previous = replay();
  if (!previous.is_null()) {
    d.exec("COMMIT");
    return previous;
  }
  validate_execution_quote(quote, side, now());
  auto used = d.rows("SELECT used FROM paper_liquidity WHERE key=?", {liquidity_key});
  if (!used.empty())
    available -= used[0]["used"].get<int64_t>();
  quantity = std::min(quantity, std::max(int64_t(0), available));
  if (!quantity)
    throw Error(409, "No unused displayed liquidity is available. IOC order cancelled.");
  int64_t fee = std::max(int64_t(100), int64_t(std::ceil(quantity * .5)));
  int64_t total = price * quantity;
  int64_t cash = d.rows("SELECT cash_cents FROM account WHERE id=1")[0]["cash_cents"];
  auto positions = d.rows("SELECT * FROM positions WHERE ticker=?", {ticker});
  int64_t held = positions.empty() ? 0 : positions[0]["quantity"].get<int64_t>(),
          cost = positions.empty() ? 0 : positions[0]["cost_cents"].get<int64_t>();
  if (side == "buy") {
    if (total + fee > cash)
      throw Error(409, "Insufficient paper cash.");
    cash -= total + fee;
    held += quantity;
    cost += total + fee;
  } else {
    if (requested > held)
      throw Error(409, "Insufficient shares. Short selling is not supported.");
    cash += total - fee;
    cost = std::nearbyint(double(cost) * (held - quantity) / held);
    held -= quantity;
  }
  if (cash < 0)
    throw Error(409, "Insufficient paper cash for commission.");
  d.exec("UPDATE account SET cash_cents=? WHERE id=1", {cash});
  if (held)
    d.exec("INSERT OR REPLACE INTO positions VALUES(?,?,?)", {ticker, held, cost});
  else
    d.exec("DELETE FROM positions WHERE ticker=?", {ticker});
  J order = {{"id", id()},
             {"ticker", ticker},
             {"side", side},
             {"quantity", quantity},
             {"price_cents", price},
             {"created_at", stamp()},
             {"price_as_of", quote[key + "_received_at"]},
             {"fee_cents", fee},
             {"requested_quantity", requested},
             {"cancelled_quantity", requested - quantity},
             {"status", quantity == requested ? "filled" : "partially_filled"},
             {"order_type", type},
             {"time_in_force", "IOC"},
             {"source", quote["source"]}};
  d.exec("INSERT INTO orders VALUES(?,?,?,?,?,?,?)",
         {order["id"], ticker, side, quantity, price, order["created_at"], order["price_as_of"]});
  d.exec("INSERT INTO order_details VALUES(?,?,?,?)",
         {order["id"], request_id.empty() ? J(nullptr) : J(request_id), q.dump(), order.dump()});
  d.exec("INSERT INTO paper_liquidity VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET "
         "used=used+excluded.used,updated=excluded.updated",
         {liquidity_key, quantity, now()});
  d.exec("DELETE FROM paper_liquidity WHERE updated<?", {now() - 120});
  d.exec("COMMIT");
  return order;
}
void check_price_alerts() {
  Db d;
  auto rows = d.rows("SELECT * FROM alerts WHERE active=1");
  std::map<std::string, double> prices;
  for (auto &a : rows) {
    auto s = a["ticker"].get<std::string>();
    if (!prices.count(s)) {
      try {
        prices[s] = history(s, 2)["price"];
      } catch (...) {
        prices[s] = NAN;
      }
    }
    double p = prices[s];
    if (!std::isfinite(p))
      continue;
    if (a["direction"] == "above" ? p >= a["threshold"].get<double>()
                                  : p <= a["threshold"].get<double>())
      d.exec("UPDATE alerts SET active=0,triggered_at=? WHERE id=? AND active=1",
             {stamp(), a["id"]});
  }
}
} // namespace sa
