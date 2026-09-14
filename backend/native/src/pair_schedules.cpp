#include "pairs.hpp"
#include "relative_value.hpp"
#include "stock_lists.hpp"
#include <set>
namespace sa {
namespace {
J validate_schedule(const J &j) {
  J c = {
      {"source",
       choice(j, "source", "scanner", {"scanner", "symbols", "relationships", "stock_list"})},
      {"interval_hours", int(number(j, "interval_hours", 24, 1, 168, true))},
      {"max_alerts", int(number(j, "max_alerts", 5, 1, 50, true))},
      {"lookback", int(number(j, "lookback", 60, 20, 250, true))},
      {"min_correlation", number(j, "min_correlation", .8, 0, 1)},
      {"threshold", number(j, "threshold", 2, .1, 20)},
      {"relationship", choice(j, "relationship", "positive", {"positive", "negative", "either"})},
      {"require_cointegration", boolean(j, "require_cointegration", true)}};
  if (c["source"] == "stock_list") {
    c.update(validate_list_search(j));
    c["lookback"] = int(number(j, "lookback", 120, 40, 250, true));
  } else if (c["source"] == "relationships") {
    if (!j.contains("relationships") || !j["relationships"].is_array() ||
        j["relationships"].empty() || j["relationships"].size() > 20)
      throw Error(422, "Supply 1–20 economic relationships.");
    c["relationships"] = J::array();
    for (auto &r : j["relationships"]) {
      auto rule = validate_pair(r);
      if (!rule.contains("relative_value"))
        throw Error(422, "Each relationship needs relative_value configuration.");
      c["relationships"].push_back(rule);
    }
  } else if (c["source"] == "symbols") {
    if (!j.contains("tickers") || !j["tickers"].is_array())
      throw Error(422, "tickers must be an array.");
    std::set<std::string> unique;
    for (auto &t : j["tickers"]) {
      if (!t.is_string())
        throw Error(422, "Invalid ticker.");
      unique.insert(symbol(t.get<std::string>()));
    }
    if (unique.size() < 2 || unique.size() > 20)
      throw Error(422, "Choose 2–20 distinct stocks.");
    c["tickers"] = unique;
  } else {
    c["scan_code"] = choice(j, "scan_code", "HOT_BY_VOLUME",
                            {"HOT_BY_VOLUME", "TOP_PERC_GAIN", "TOP_PERC_LOSE"});
    c["limit"] = int(number(j, "limit", 20, 2, 20, true));
    c["min_price"] = number(j, "min_price", 5, 0, 100000);
    c["max_price"] = number(j, "max_price", 1000, .01, 100000);
    c["min_volume"] = int(number(j, "min_volume", 100000, 0, 1e9, true));
    if (c["min_price"].get<double>() >= c["max_price"].get<double>())
      throw Error(422, "Maximum price must exceed minimum price.");
  }
  return c;
}
std::string key(const J &c) {
  if (c.contains("relative_value")) {
    auto identity = c["relative_value"];
    identity.erase("lookback");
    identity.erase("entry_z");
    return "rv:" + hash(identity.dump());
  }
  auto a = c["ticker_a"].get<std::string>(), b = c["ticker_b"].get<std::string>();
  return std::min(a, b) + "/" + std::max(a, b);
}
J decode(J row) {
  row["config"] = J::parse(row["config_json"].get<std::string>());
  row.erase("config_json");
  row["active"] = row["active"] == 1;
  row["running"] = row["running"] == 1;
  row["next_run_at"] = stamp(row["next_run"].get<double>());
  row["last_result"] =
      row["last_result"].is_null() ? J() : J::parse(row["last_result"].get<std::string>());
  return row;
}
} // namespace
J schedule_list() {
  Db d;
  J out = J::array();
  for (auto r : d.rows("SELECT * FROM pair_schedules ORDER BY created_at"))
    out.push_back(decode(r));
  return out;
}
J schedule_save(const J &j, const std::string &existing) {
  auto c = validate_schedule(j);
  Db d;
  auto uid = existing.empty() ? id() : existing;
  if (existing.empty())
    d.exec("INSERT INTO pair_schedules(id,config_json,created_at,next_run) VALUES(?,?,?,?)",
           {uid, c.dump(), stamp(), now()});
  else {
    d.exec("UPDATE pair_schedules SET "
           "config_json=?,version=version+1,next_run=?,last_result=NULL,last_error=NULL WHERE id=?",
           {c.dump(), now(), uid});
    if (!d.changes())
      throw Error(404, "Schedule not found.");
  }
  return decode(d.rows("SELECT * FROM pair_schedules WHERE id=?", {uid})[0]);
}
J schedule_action(const std::string &uid, const std::string &action, bool active) {
  Db d;
  d.exec("BEGIN IMMEDIATE");
  auto rows = d.rows("SELECT * FROM pair_schedules WHERE id=?", {uid});
  if (rows.empty())
    throw Error(404, "Schedule not found.");
  if (action == "delete") {
    d.exec("DELETE FROM pair_schedule_links WHERE schedule_id=?", {uid});
    d.exec("DELETE FROM pair_schedules WHERE id=?", {uid});
    d.exec("COMMIT");
    return nullptr;
  }
  if (action == "run") {
    if (rows[0]["running"] == 1 || rows[0]["active"] != 1)
      throw Error(409, "Schedule must be active and idle.");
    d.exec("UPDATE pair_schedules SET next_run=? WHERE id=?", {now(), uid});
  } else
    d.exec("UPDATE pair_schedules SET active=?,version=version+1,next_run=? WHERE id=?",
           {active ? 1 : 0, now(), uid});
  auto result = decode(d.rows("SELECT * FROM pair_schedules WHERE id=?", {uid})[0]);
  d.exec("COMMIT");
  return result;
}
void check_pair_schedules() {
  Db d;
  for (auto row : d.rows(
           "SELECT * FROM pair_schedules WHERE active=1 AND running=0 AND next_run<=?", {now()})) {
    auto uid = row["id"].get<std::string>();
    int version = row["version"].get<int>();
    auto c = J::parse(row["config_json"].get<std::string>());
    d.exec("UPDATE pair_schedules SET running=1,last_started=? WHERE id=? AND version=? AND "
           "active=1 AND running=0 AND next_run<=?",
           {stamp(), uid, version, now()});
    if (!d.changes())
      continue;
    try {
      J request = c;
      if (c["source"] == "scanner") {
        request["tickers"] = J::array();
        auto universe = scan(c);
        for (auto &stock : universe["results"])
          request["tickers"].push_back(stock["ticker"]);
      }
      auto result = c["source"] == "stock_list" ? discover_stock_list(c)
                                                : (c["source"] == "relationships"
                                                       ? discover_relationships(c["relationships"])
                                                       : discover_pairs(request));
      if (result["pairs"].empty() && !result["errors"].empty() &&
          result.value("valuation", J()).is_null())
        throw Error(503, result["errors"].dump());
      d.exec("BEGIN IMMEDIATE");
      auto current = d.rows("SELECT id FROM pair_schedules WHERE id=? AND version=? AND active=1",
                            {uid, version});
      if (!current.empty()) {
        std::set<std::string> seen;
        for (auto &a : d.rows("SELECT config_json FROM pair_alerts"))
          seen.insert(key(J::parse(a["config_json"].get<std::string>())));
        for (auto &a :
             d.rows("SELECT pair_key FROM pair_schedule_links WHERE schedule_id=?", {uid}))
          seen.insert(a["pair_key"].get<std::string>());
        int owned = d.rows("SELECT COUNT(*) AS n FROM pair_schedule_links l JOIN pair_alerts a ON "
                           "a.id=l.alert_id WHERE l.schedule_id=?",
                           {uid})[0]["n"]
                        .get<int>();
        J created = J::array();
        for (auto &candidate : result["pairs"]) {
          if (owned >= c["max_alerts"].get<int>())
            break;
          auto rule = validate_pair(candidate["config"]);
          rule["repeat"] = true;
          auto pair = key(rule);
          if (!seen.insert(pair).second)
            continue;
          auto aid = id(), time = stamp();
          d.exec("INSERT INTO pair_alerts(id,config_json,created_at,updated_at,latest_json) "
                 "VALUES(?,?,?,?,?)",
                 {aid, rule.dump(), time, time, candidate["snapshot"].dump()});
          if (candidate["snapshot"].contains("discovery"))
            d.exec("INSERT INTO pair_alert_context VALUES(?,?)",
                   {aid, candidate["snapshot"]["discovery"].dump()});
          d.exec("INSERT INTO pair_schedule_links(schedule_id,pair_key,alert_id) VALUES(?,?,?)",
                 {uid, pair, aid});
          created.push_back(rule["ticker_a"].get<std::string>() + "/" +
                            rule["ticker_b"].get<std::string>());
          ++owned;
        }
        J summary = {{"candidates", result["pairs"].size()},
                     {"created", created},
                     {"managed_alerts", owned},
                     {"errors", result["errors"]},
                     {"tickers", request.value("tickers", J::array())}};
        for (auto key : {"list", "eligible", "evaluated", "skipped", "valuation"})
          if (result.contains(key))
            summary[key] = result[key];
        if (result.contains("valuation") && result["valuation"].is_object())
          for (auto &v : result["valuation"]["rows"])
            if (v["action"]["state"] == "entry") {
              auto eid = hash(uid + v["ticker"].get<std::string>() +
                              result["valuation"]["as_of"].get<std::string>() +
                              v["valuation_direction"].get<std::string>());
              d.exec("INSERT OR IGNORE INTO valuation_events VALUES(?,?,?,?)",
                     {eid, uid, stamp(), v.dump()});
            }
        d.exec("UPDATE pair_schedules SET last_result=?,last_error=NULL,last_finished=?,next_run=? "
               "WHERE id=? AND version=?",
               {summary.dump(), stamp(), now() + 3600 * c["interval_hours"].get<double>(), uid,
                version});
      }
      d.exec("UPDATE pair_schedules SET running=0 WHERE id=?", {uid});
      d.exec("COMMIT");
    } catch (const std::exception &e) {
      // A failed transaction must be rolled back before recording its error.
      try {
        d.exec("ROLLBACK");
      } catch (...) {
      }
      d.exec("UPDATE pair_schedules SET last_error=?,last_finished=?,next_run=? WHERE id=? AND "
             "version=?",
             {std::string(e.what()), stamp(), now() + 300, uid, version});
      d.exec("UPDATE pair_schedules SET running=0 WHERE id=?", {uid});
    }
  }
}
} // namespace sa
