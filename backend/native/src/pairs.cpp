#include "pairs.hpp"
#include <numeric>
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
  std::vector<double> ratios, spreads;
  double hedge = c["hedge_ratio"];
  for (auto &[day, price] : a)
    if (b.count(day)) {
      if (!std::isfinite(price) || !std::isfinite(b[day]) || price <= 0 || b[day] <= 0)
        throw Error(502, "Pair history contains invalid prices.");
      dates.push_back(day);
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
  return {{"ticker_a", c["ticker_a"]},
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
  return {{"cash_cents", d.rows("SELECT cash_cents FROM account WHERE id=1")[0]["cash_cents"]},
          {"positions", d.rows("SELECT * FROM positions ORDER BY ticker")},
          {"orders", d.rows("SELECT * FROM orders ORDER BY created_at DESC LIMIT 50")}};
}
J place_order(const J &q) {
  auto ticker = symbol(text(q, "ticker"));
  auto side = choice(q, "side", "", {"buy", "sell"});
  int64_t quantity = number(q, "quantity", 0, 1, 1000000, true);
  auto quote = history(ticker, 2);
  double raw_price = quote["price"];
  if (!std::isfinite(raw_price) || raw_price <= 0 || raw_price > 1e10)
    throw Error(422, "Price must be positive and within the paper ledger's "
                     "supported range.");
  int64_t price = std::llround(raw_price * 100), total = price * quantity;
  Db d;
  d.exec("BEGIN IMMEDIATE");
  int64_t cash = d.rows("SELECT cash_cents FROM account WHERE id=1")[0]["cash_cents"];
  auto positions = d.rows("SELECT * FROM positions WHERE ticker=?", {ticker});
  int64_t held = positions.empty() ? 0 : positions[0]["quantity"].get<int64_t>(),
          cost = positions.empty() ? 0 : positions[0]["cost_cents"].get<int64_t>();
  if (side == "buy") {
    if (total > cash)
      throw Error(409, "Insufficient paper cash.");
    cash -= total;
    held += quantity;
    cost += total;
  } else {
    if (quantity > held)
      throw Error(409, "Insufficient shares. Short selling is not supported.");
    cash += total;
    cost = std::nearbyint(double(cost) * (held - quantity) / held);
    held -= quantity;
  }
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
             {"price_as_of", quote["as_of"]}};
  d.exec("INSERT INTO orders VALUES(?,?,?,?,?,?,?)",
         {order["id"], ticker, side, quantity, price, order["created_at"], order["price_as_of"]});
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
