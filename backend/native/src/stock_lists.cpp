#include "stock_lists.hpp"
#include "relative_value.hpp"
#include "valuation.hpp"
#include <regex>
#include <set>
namespace sa {
namespace {
J decode(J row) {
  row["tickers"] = J::parse(row["tickers_json"].get<std::string>());
  row.erase("tickers_json");
  return row;
}
bool contained(const J &rule, const std::set<std::string> &symbols) {
  if (!symbols.count(rule["ticker_a"].get<std::string>()))
    return false;
  if (rule.contains("relative_value")) {
    for (auto &r : rule["relative_value"]["references"])
      if (!symbols.count(r["ticker"].get<std::string>()))
        return false;
    return true;
  }
  return symbols.count(rule["ticker_b"].get<std::string>());
}
void invalidate(Db &d, const std::string &uid, const std::set<std::string> &symbols,
                bool deleting) {
  for (auto &row : d.rows("SELECT id,config_json FROM pair_schedules")) {
    auto c = J::parse(row["config_json"].get<std::string>());
    if (c.value("source", "") != "stock_list" || c.value("stock_list_id", "") != uid)
      continue;
    auto sid = row["id"].get<std::string>();
    for (auto &a : d.rows("SELECT a.id,a.config_json FROM pair_alerts a JOIN pair_schedule_links l "
                          "ON l.alert_id=a.id WHERE l.schedule_id=?",
                          {sid})) {
      if (deleting || !contained(J::parse(a["config_json"].get<std::string>()), symbols))
        d.exec("UPDATE pair_alerts SET active=0,version=version+1,updated_at=? WHERE id=?",
               {stamp(), a["id"]});
    }
    if (deleting) {
      d.exec("DELETE FROM pair_schedule_links WHERE schedule_id=?", {sid});
      d.exec("DELETE FROM pair_schedules WHERE id=?", {sid});
    } else
      d.exec("UPDATE pair_schedules SET "
             "version=version+1,next_run=?,last_result=NULL,last_error=NULL WHERE id=?",
             {now(), sid});
  }
}
std::string sector_proxy(const J &profile) {
  std::string s = profile.value("industry", "");
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
  s = std::regex_replace(s, std::regex("[^a-z0-9]+"), " ");
  for (auto &[keyword, ticker] :
       std::vector<std::pair<std::string, std::string>>{{"real estate", "XLRE"},
                                                        {"technology", "XLK"},
                                                        {"financial", "XLF"},
                                                        {"energy", "XLE"},
                                                        {"health", "XLV"},
                                                        {"consumer non", "XLP"},
                                                        {"consumer staple", "XLP"},
                                                        {"consumer cyclical", "XLY"},
                                                        {"consumer discretionary", "XLY"},
                                                        {"basic material", "XLB"},
                                                        {"materials", "XLB"},
                                                        {"industrial", "XLI"},
                                                        {"capital goods", "XLI"},
                                                        {"utilities", "XLU"},
                                                        {"communication", "XLC"},
                                                        {"telecom", "XLC"}})
    if (s.find(keyword) != std::string::npos)
      return ticker;
  return "";
}
} // namespace
J stock_lists() {
  Db d;
  J out = J::array();
  for (auto row : d.rows("SELECT * FROM stock_lists ORDER BY created_at"))
    out.push_back(decode(row));
  return out;
}
J stock_list(const std::string &uid) {
  Db d;
  auto r = d.rows("SELECT * FROM stock_lists WHERE id=?", {uid});
  if (r.empty())
    throw Error(404, "Stock list not found.");
  return decode(r[0]);
}
J save_stock_list(const J &j, const std::string &existing) {
  auto name = text(j, "name", "", 100);
  if (name.find_first_not_of(" \t\r\n") == std::string::npos)
    throw Error(422, "Give this stock list a name.");
  if (!j.contains("tickers") || !j["tickers"].is_array() || j["tickers"].size() > 20)
    throw Error(422, "Supply 2–20 stock symbols.");
  std::set<std::string> unique;
  J tickers = J::array();
  for (auto &t : j["tickers"]) {
    if (!t.is_string())
      throw Error(422, "Tickers must be strings.");
    auto s = symbol(t.get<std::string>());
    if (unique.insert(s).second)
      tickers.push_back(s);
  }
  if (unique.size() < 2)
    throw Error(422, "Supply 2–20 distinct stock symbols.");
  auto uid = existing.empty() ? id() : existing;
  Db d;
  d.exec("BEGIN IMMEDIATE");
  if (existing.empty())
    d.exec("INSERT INTO stock_lists(id,name,tickers_json,created_at,updated_at) VALUES(?,?,?,?,?)",
           {uid, name, tickers.dump(), stamp(), stamp()});
  else {
    d.exec("UPDATE stock_lists SET name=?,tickers_json=?,version=version+1,updated_at=? WHERE id=?",
           {name, tickers.dump(), stamp(), uid});
    if (!d.changes())
      throw Error(404, "Stock list not found.");
    invalidate(d, uid, unique, false);
  }
  auto result = decode(d.rows("SELECT * FROM stock_lists WHERE id=?", {uid})[0]);
  d.exec("COMMIT");
  return result;
}
void delete_stock_list(const std::string &uid) {
  Db d;
  d.exec("BEGIN IMMEDIATE");
  if (d.rows("SELECT id FROM stock_lists WHERE id=?", {uid}).empty())
    throw Error(404, "Stock list not found.");
  invalidate(d, uid, {}, true);
  d.exec("DELETE FROM stock_lists WHERE id=?", {uid});
  d.exec("COMMIT");
}
J validate_list_search(const J &j) {
  auto uid = text(j, "stock_list_id", "", 100);
  stock_list(uid);
  if (!j.contains("algorithms") || !j["algorithms"].is_array() || j["algorithms"].empty() ||
      j["algorithms"].size() > 6)
    throw Error(422, "Choose at least one strategy.");
  std::set<std::string> unique;
  for (auto &a : j["algorithms"]) {
    J one = {{"algorithm", a}};
    unique.insert(
        choice(one, "algorithm", "",
               {"residual_pair", "basket", "lead_lag", "same_company", "etf", "peer_valuation"}));
  }
  J factors = j.value("factors", J::array({"MTUM", "VLUE"}));
  if (!factors.is_array() || factors.empty() || factors.size() > 6)
    throw Error(422, "Choose 1–6 factor ETF proxies.");
  std::set<std::string> hedges;
  auto market = symbol(text(j, "market", "SPY"));
  hedges.insert(market);
  for (auto &f : factors) {
    if (!f.is_string())
      throw Error(422, "Factor proxies must be symbols.");
    auto s = symbol(f.get<std::string>());
    if (!hedges.insert(s).second)
      throw Error(422, "Market and factor proxies must be distinct.");
    f = s;
  }
  auto sector = text(j, "sector", "auto", 20);
  if (sector != "auto") {
    sector = symbol(sector);
    if (hedges.count(sector))
      throw Error(422, "Sector hedge must differ from market and factors.");
  }
  auto vp = j.value("valuation_peers", J::array());
  if (!vp.is_array() || vp.size() > 40)
    throw Error(422, "Supply up to 40 additional valuation peers.");
  std::set<std::string> peer_symbols;
  for (auto &p : vp) {
    if (!p.is_string())
      throw Error(422, "Peer tickers must be strings.");
    peer_symbols.insert(symbol(p.get<std::string>()));
  }
  return {{"stock_list_id", uid},
          {"valuation_min_gap", number(j, "valuation_min_gap", .2, .05, 1)},
          {"valuation_peers", peer_symbols},
          {"algorithms", unique},
          {"market", market},
          {"sector", sector},
          {"factors", factors},
          {"basket_size", int(number(j, "basket_size", 5, 2, 10, true))},
          {"model_limit", int(number(j, "model_limit", 20, 1, 30, true))}};
}
J discover_stock_list(const J &c) {
  auto list = stock_list(c["stock_list_id"]);
  J valuation = nullptr;
  if (std::find(c["algorithms"].begin(), c["algorithms"].end(), "peer_valuation") !=
      c["algorithms"].end()) {
    valuation = peer_valuation(c);
    if (c["algorithms"].size() == 1)
      return {{"pairs", J::array()},   {"errors", J::array()}, {"list", list},
              {"evaluated", 0},        {"eligible", 0},        {"skipped", J::array()},
              {"valuation", valuation}};
  }
  J errors = J::array(), pairs = J::array(), profiles = J::object(), skipped = J::array();
  std::set<std::string> members;
  for (auto &t : list["tickers"])
    members.insert(t.get<std::string>());
  for (auto &t : members)
    try {
      auto p = company_profile(t);
      p["sector_proxy"] = c["sector"] == "auto" ? J(sector_proxy(p)) : c["sector"];
      profiles[t] = p;
    } catch (const std::exception &e) {
      errors.push_back({{"ticker", t}, {"error", e.what()}});
      if (c["sector"] != "auto")
        profiles[t] = {{"industry", "User-selected sector"},
                       {"category", ""},
                       {"sector_proxy", c["sector"]},
                       {"source", "user"}};
    }
  auto scan_request = c;
  scan_request["tickers"] = list["tickers"];
  scan_request["require_cointegration"] = false;
  scan_request["relationship"] = "positive";
  auto discovered = discover_pairs(scan_request);
  for (auto &item : discovered.value("skipped", J::array()))
    skipped.push_back(item);
  if (std::find(c["algorithms"].begin(), c["algorithms"].end(), "basket") !=
          c["algorithms"].end() &&
      members.size() < 3)
    skipped.push_back({{"ticker", "Basket strategy"},
                       {"reason", "Requires at least three stocks: one target and two eligible "
                                  "references. This list has only " +
                                      std::to_string(members.size()) + "."}});
  for (auto &e : discovered["errors"])
    errors.push_back(e);
  std::map<std::string, std::vector<J>> queues;
  std::map<std::string, std::vector<std::pair<std::string, double>>> neighbors;
  auto existing_monitors = pair_list();
  auto add = [&](std::string algorithm, std::string target, J references, std::string sector,
                 std::string reason, J extra = J::object()) {
    J v = {{"category", algorithm == "residual_pair" ? "competitor" : algorithm},
           {"target", target},
           {"references", references},
           {"market", c["market"]},
           {"sector", sector},
           {"factors", c["factors"]},
           {"fit_reference", algorithm == "residual_pair" || algorithm == "basket"},
           {"lookback", c["lookback"]},
           {"entry_z", c["threshold"]}};
    // Reuse an existing fitted pair's orientation so a list does not create
    // a second, reverse regression for the same economic pair by accident.
    if (algorithm == "residual_pair")
      for (auto &old : existing_monitors) {
        if (!old.contains("relative_value"))
          continue;
        auto &prior = old["relative_value"];
        if (prior["category"] != "competitor" || prior["fit_reference"] != true ||
            prior["references"].size() != 1 || prior["market"] != v["market"] ||
            prior["sector"] != v["sector"] || prior["factors"] != v["factors"])
          continue;
        std::set<std::string> left = {target, references[0]["ticker"].get<std::string>()},
                              right = {prior["target"].get<std::string>(),
                                       prior["references"][0]["ticker"].get<std::string>()};
        if (left == right) {
          v["target"] = prior["target"];
          v["references"] = prior["references"];
          break;
        }
      }
    v.update(extra);
    try {
      auto rule = validate_pair({{"relative_value", v}});
      J evidence = {
          {"list_id", list["id"]},  {"list_name", list["name"]}, {"list_version", list["version"]},
          {"algorithm", algorithm}, {"reason", reason},          {"profiles", J::object()}};
      for (auto &s : std::vector<std::string>{target, references[0]["ticker"].get<std::string>()})
        if (profiles.contains(s))
          evidence["profiles"][s] = profiles[s];
      queues[algorithm].push_back({{"config", rule}, {"evidence", evidence}});
    } catch (const std::exception &e) {
      skipped.push_back({{"ticker", target}, {"reason", e.what()}});
    }
  };
  for (auto &p : discovered["pairs"]) {
    auto a = p["config"]["ticker_a"].get<std::string>(),
         b = p["config"]["ticker_b"].get<std::string>();
    if (!profiles.contains(a) || !profiles.contains(b))
      continue;
    auto pa = profiles[a], pb = profiles[b];
    auto sector = pa.value("sector_proxy", "");
    bool same = !sector.empty() && sector == pb.value("sector_proxy", "") &&
                (c["sector"] != "auto" || (!pa.value("category", "").empty() &&
                                           pa.value("category", "") == pb.value("category", "")));
    if (!same) {
      skipped.push_back({{"ticker", a + "/" + b},
                         {"reason", "No matching provider industry/category and sector hedge. Set "
                                    "a sector override only for a related list."}});
      continue;
    }
    auto reason =
        c["sector"] == "auto"
            ? "Provider profiles place both securities in " + pa.value("category", "") +
                  ". Prior daily return correlation passes the list filter."
            : "You selected a common sector for this list. Prior daily return correlation passes "
              "the filter; economic equivalence is not independently verified.";
    J refs = J::array({{{"ticker", b}, {"weight", 1}}});
    add("residual_pair", a, refs, sector, reason);
    add("lead_lag", a, refs, sector, reason + " Test the next-day residual forecast.");
    add("lead_lag", b, J::array({{{"ticker", a}, {"weight", 1}}}), sector,
        reason + " Reverse forecast direction.");
    neighbors[a].push_back({b, p["correlation"]});
    neighbors[b].push_back({a, p["correlation"]});
  }
  for (auto &[target, near] : neighbors) {
    if (near.size() < 2)
      continue;
    std::sort(near.begin(), near.end(), [](auto &a, auto &b) { return a.second > b.second; });
    J refs = J::array();
    for (auto &[ticker, corr] : near) {
      if (refs.size() >= size_t(c["basket_size"].get<int>()))
        break;
      refs.push_back({{"ticker", ticker}, {"weight", 1}});
    }
    add("basket", target, refs, profiles[target]["sector_proxy"],
        "Fit the target against its strongest correlated, same-group list members; retain the "
        "unexplained component after factor hedging.");
  }
  if (members.count("GOOG") && members.count("GOOGL"))
    add("same_company", "GOOG", J::array({{{"ticker", "GOOGL"}, {"weight", 1}}}), "XLC",
        "Known Alphabet share classes: common issuer, different voting rights. A zero spread is "
        "not assumed.");
  for (auto &group : std::vector<std::vector<std::string>>{{"SPY", "IVV", "VOO"}, {"GLD", "IAU"}})
    for (size_t a = 0; a < group.size(); ++a)
      for (size_t b = a + 1; b < group.size(); ++b)
        if (members.count(group[a]) && members.count(group[b])) {
          bool gold = group[0] == "GLD";
          J extra = gold ? J::object() : J{{"market", "VTI"}};
          add("etf", group[a], J::array({{{"ticker", group[b]}, {"weight", 1}}}),
              gold ? "GDX" : "IWB",
              gold
                  ? "Both configured funds track gold bullion; fee and tracking differences remain."
                  : "Both configured funds track the S&P 500; dividends, fees and execution costs "
                    "remain.",
              extra);
        }
  // Round-robin strategy families so a large pair universe cannot starve baskets.
  J selected = J::array();
  bool more = true;
  size_t position = 0, total = 0;
  for (auto &a : c["algorithms"])
    total += queues[a.get<std::string>()].size();
  while (more && selected.size() < size_t(c["model_limit"].get<int>())) {
    more = false;
    for (auto &a : c["algorithms"]) {
      auto &q = queues[a.get<std::string>()];
      if (position < q.size()) {
        selected.push_back(q[position]);
        more = true;
        if (selected.size() >= size_t(c["model_limit"].get<int>()))
          break;
      }
    }
    ++position;
  }
  for (auto &candidate : selected)
    try {
      auto snapshot = analyze_pair(candidate["config"]);
      snapshot["discovery"] = candidate["evidence"];
      pairs.push_back(
          {{"config", candidate["config"]}, {"snapshot", snapshot}, {"correlation", nullptr}});
    } catch (const std::exception &e) {
      errors.push_back({{"ticker", candidate["config"]["ticker_a"]}, {"error", e.what()}});
    }
  return {{"valuation", valuation},
          {"pairs", pairs},
          {"errors", errors},
          {"list",
           {{"id", list["id"]},
            {"name", list["name"]},
            {"version", list["version"]},
            {"tickers", list["tickers"]}}},
          {"evaluated", selected.size()},
          {"eligible", total},
          {"skipped", skipped}};
}
J pair_opportunities() {
  Db d;
  J contexts = J::object();
  for (auto &r : d.rows("SELECT * FROM pair_alert_context"))
    contexts[r["alert_id"].get<std::string>()] = J::parse(r["discovery_json"].get<std::string>());
  J out = J::array();
  for (auto row : pair_list()) {
    if (!row.contains("relative_value"))
      continue;
    auto latest = row["latest"];
    std::string state = "wait", action = "Wait — no entry signal";
    if (row["active"] != 1) {
      state = "paused";
      action = "Paused — no action";
    } else if (!row["last_error"].is_null()) {
      state = "error";
      action = "Data/model error — do not open";
    } else if (latest.is_null()) {
      state = "waiting_data";
      action = "Waiting for analysis — no action";
    } else if (parse_time(ny_date()) - parse_time(latest["as_of"].get<std::string>()) > 5 * 86400) {
      state = "stale";
      action = "Outdated signal — wait for fresh data";
    } else if (std::abs(latest["value"].get<double>()) >=
               std::max(4.0, 2 * row["threshold"].get<double>())) {
      state = "avoid";
      action = "Beyond the strategy stop band — do not open";
    } else if (latest["matched"] == true) {
      state = "entry";
      action = "Entry candidate — review all legs";
    }
    row["action"] = {
        {"state", state},
        {"label", action},
        {"legs", state == "entry" ? latest["relative_value"]["hedge_legs"] : J::array()},
        {"entry_z", row["threshold"]},
        {"exit_z", std::min(.5, row["threshold"].get<double>() / 2)},
        {"stop_z", std::max(4.0, row["threshold"].get<double>() * 2)}};
    row["discovery"] = contexts.value(row["id"].get<std::string>(), J());
    // Points remain in the full alert endpoint, not the compact action feed.
    if (!row["latest"].is_null())
      row["latest"].erase("points");
    out.push_back(row);
  }
  return out;
}
} // namespace sa
