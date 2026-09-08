#include "backtests.hpp"
#include "httplib.h"
#include "pairs.hpp"
#include <condition_variable>
#include <csignal>
#include <iostream>
namespace sa {
namespace {
J object(const httplib::Request &r) {
  if (r.body.empty())
    throw Error(422, "JSON body is required.");
  J j = J::parse(r.body);
  if (!j.is_object())
    throw Error(422, "Expected a JSON object.");
  return j;
}
std::string query(const httplib::Request &r, const std::string &key,
                  const std::string &fallback = "") {
  return r.has_param(key) ? r.get_param_value(key) : fallback;
}
int query_int(const httplib::Request &r, const std::string &key, int fallback, int min, int max) {
  if (!r.has_param(key))
    return fallback;
  auto value = query(r, key);
  if (!std::regex_match(value, std::regex("[0-9]{1,9}")))
    throw Error(422, "Invalid " + key + ".");
  int n = std::stoi(value);
  if (n < min || n > max)
    throw Error(422, key + " is outside its allowed range.");
  return n;
}
void json(httplib::Response &r, const J &j, int status = 200) {
  r.status = status;
  r.set_content(j.dump(), "application/json");
}
J alert_config(const J &q) {
  double threshold = number(q, "threshold", 0, 1e-12, 1e7);
  return {{"ticker", symbol(text(q, "ticker"))},
          {"direction", choice(q, "direction", "", {"above", "below"})},
          {"threshold", threshold}};
}
J create_alert(const J &q, const std::string &existing = "") {
  auto a = alert_config(q);
  history(a["ticker"], 2);
  Db d;
  auto key = existing.empty() ? id() : existing;
  if (existing.empty()) {
    a.update({{"id", key}, {"active", 1}, {"triggered_at", nullptr}, {"created_at", stamp()}});
    d.exec("INSERT INTO alerts VALUES(?,?,?,?,1,NULL,?)",
           {key, a["ticker"], a["direction"], a["threshold"], a["created_at"]});
  } else {
    d.exec("UPDATE alerts SET "
           "ticker=?,direction=?,threshold=?,active=1,triggered_at=NULL WHERE "
           "id=?",
           {a["ticker"], a["direction"], a["threshold"], key});
    if (!d.changes())
      throw Error(404, "Alert not found.");
    a = d.rows("SELECT * FROM alerts WHERE id=?", {key})[0];
  }
  return a;
}
std::atomic<bool> shutdown_requested{false};
void on_signal(int) { shutdown_requested = true; }
} // namespace
} // namespace sa
int main(int argc, char **argv) {
  using namespace sa;
  try {
    load_config();
    for (int i = 1; i < argc; i++)
      if (std::string(argv[i]) == "--port" && i + 1 < argc)
        config.port = std::stoi(argv[++i]);
    initialize();
    httplib::Server server;
    server.set_payload_max_length(128 * 1024);
    server.new_task_queue = []() { return new httplib::ThreadPool(8); };
    server.set_exception_handler([](const auto &, auto &r, std::exception_ptr e) {
      try {
        std::rethrow_exception(e);
      } catch (const Error &ex) {
        json(r, {{"detail", ex.what()}}, ex.status);
      } catch (const nlohmann::json::exception &) {
        json(r, {{"detail", "Invalid JSON or field type."}}, 422);
      } catch (const std::exception &ex) {
        std::cerr << "Backend error: " << ex.what() << '\n';
        json(r, {{"detail", "Backend operation failed. Check local server logs."}}, 500);
      }
    });
    server.set_pre_routing_handler([](const auto &r, auto &response) {
      std::string host = r.get_header_value("Host");
      if (!std::regex_match(host, std::regex(R"((localhost|127\.0\.0\.1|\[::1\])(:\d+)?)"))) {
        json(response, {{"detail", "StockAssist API is available on localhost only."}}, 400);
        return httplib::Server::HandlerResponse::Handled;
      }
      if (r.method != "GET" && r.method != "HEAD" && r.method != "OPTIONS") {
        auto origin = r.get_header_value("Origin");
        if (!origin.empty() && origin != "http://127.0.0.1:3000" &&
            origin != "http://localhost:3000") {
          json(response, {{"detail", "Cross-origin write rejected."}}, 403);
          return httplib::Server::HandlerResponse::Handled;
        }
        if (r.method != "DELETE" &&
            r.get_header_value("Content-Type")
                    .substr(0, r.get_header_value("Content-Type").find(';')) !=
                "application/json") {
          json(response, {{"detail", "Content-Type must be application/json."}}, 415);
          return httplib::Server::HandlerResponse::Handled;
        }
      }
      return httplib::Server::HandlerResponse::Unhandled;
    });
    server.set_post_routing_handler([](const auto &r, auto &response) {
      auto origin = r.get_header_value("Origin");
      if (origin == "http://127.0.0.1:3000" || origin == "http://localhost:3000") {
        response.set_header("Access-Control-Allow-Origin", origin);
        response.set_header("Vary", "Origin");
      }
    });
    server.Options(R"(/.*)", [](const auto &, auto &r) {
      r.set_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE");
      r.set_header("Access-Control-Allow-Headers", "Content-Type");
      r.status = 204;
    });
    auto health = [](const auto &, auto &r) {
      json(r, {{"status", "ok"},
               {"data_mode", config.mode},
               {"trading_mode", "paper"},
               {"account_mode", "local-single-user"},
               {"backend", "cpp"},
               {"engine_version", "3.0.0-native-cpp"}});
    };
    server.Get("/health", health);
    server.Get("/", health);
    server.Get("/api/market/status", [](const auto &, auto &r) {
      json(r, config.mode == "ibkr" ? ib_status()
                                    : J{{"provider", config.mode},
                                        {"connected", nullptr},
                                        {"price_type", "completed_daily_close"}});
    });
    server.Get("/api/search", [](const auto &q, auto &r) {
      auto term = query(q, "ticker");
      if (term.size() > 100)
        throw Error(422, "Search query is too long.");
      json(r, search(term, query_int(q, "limit", 20, 1, 100)));
    });
    server.Post("/api/search", [](const auto &q, auto &r) {
      auto j = object(q);
      json(r, search(text(j, "ticker", "", 100), number(j, "limit", 20, 1, 100, true)));
    });
    server.Get(R"(/api/stocks/([^/]+))", [](const auto &q, auto &r) {
      auto interval = query(q, "interval", "1d"), before = query(q, "before");
      resolution(interval);
      if (!before.empty()) {
        double t = parse_time(before);
        if (interval == "1d" && before.size() != 10)
          throw Error(422, "Daily history uses a date cursor.");
        if (interval != "1d")
          before = stamp(t);
      }
      json(r, history(symbol(q.matches[1]), query_int(q, "days", 90, 2, 250), before, interval));
    });
    server.Get(R"(/api/stocks/([^/]+)/ticks)", [](const auto &q, auto &r) {
      auto before = query(q, "before");
      if (!before.empty())
        before = stamp(parse_time(before));
      json(r, ticks(symbol(q.matches[1]), before));
    });
    auto alert_list = [](const auto &, auto &r) {
      Db d;
      json(r, d.rows("SELECT * FROM alerts ORDER BY created_at DESC"));
    };
    server.Get("/api/alerts", alert_list);
    server.Post("/api/alert/list", alert_list);
    auto add_alert = [](const auto &q, auto &r) { json(r, create_alert(object(q)), 201); };
    server.Post("/api/alerts", add_alert);
    server.Post("/api/alert/add", add_alert);
    server.Put(R"(/api/alerts/([^/]+))",
               [](const auto &q, auto &r) { json(r, create_alert(object(q), q.matches[1])); });
    server.Delete(R"(/api/alerts/([^/]+))", [](const auto &q, auto &r) {
      Db d;
      d.exec("DELETE FROM alerts WHERE id=?", {q.matches[1].str()});
      if (!d.changes())
        throw Error(404, "Alert not found.");
      r.status = 204;
    });
    server.Get("/api/portfolio", [](const auto &, auto &r) { json(r, portfolio()); });
    server.Post("/api/order", [](const auto &q, auto &r) { json(r, place_order(object(q)), 201); });
    server.Get("/api/strategies/examples", [](const auto &, auto &r) { json(r, examples()); });
    server.Get("/api/strategies", [](const auto &, auto &r) {
      Db d;
      auto rows = d.rows("SELECT * FROM strategies ORDER BY updated_at DESC");
      for (auto &row : rows) {
        row["params"] = J::parse(row["params_json"].get<std::string>());
        row.erase("params_json");
      }
      json(r, rows);
    });
    server.Post("/api/strategies", [](const auto &q, auto &r) {
      auto j = validate_strategy(object(q));
      Db d;
      j["id"] = id();
      d.exec("INSERT INTO strategies VALUES(?,?,?,?,?,?)",
             {j["id"], j["name"], j["language"], j["code"], j["params"].dump(), stamp()});
      json(r, j, 201);
    });
    server.Post("/api/scanner", [](const auto &q, auto &r) { json(r, scan(object(q))); });
    server.Get("/api/backtests", [](const auto &, auto &r) { json(r, backtest_list()); });
    server.Get(R"(/api/backtests/([^/]+))",
               [](const auto &q, auto &r) { json(r, backtest_get(q.matches[1])); });
    server.Post("/api/backtests", [](const auto &q, auto &r) {
      json(r, backtest_submit(validate_backtest(object(q))), 202);
    });
    server.Post("/api/pairs/preview",
                [](const auto &q, auto &r) { json(r, analyze_pair(validate_pair(object(q)))); });
    server.Get("/api/pairs/alerts", [](const auto &, auto &r) { json(r, pair_list()); });
    server.Post("/api/pairs/alerts",
                [](const auto &q, auto &r) { json(r, pair_save(validate_pair(object(q))), 201); });
    server.Put(R"(/api/pairs/alerts/([^/]+))", [](const auto &q, auto &r) {
      json(r, pair_save(validate_pair(object(q)), q.matches[1]));
    });
    server.Delete(R"(/api/pairs/alerts/([^/]+))", [](const auto &q, auto &r) {
      pair_delete(q.matches[1]);
      r.status = 204;
    });
    server.Post(R"(/api/pairs/alerts/([^/]+)/state)", [](const auto &q, auto &r) {
      auto j = object(q);
      if (!j.contains("active"))
        throw Error(422, "active is required.");
      json(r, pair_state(q.matches[1], boolean(j, "active")));
    });
    server.Get("/api/pairs/events", [](const auto &, auto &r) { json(r, pair_events()); });
    std::signal(SIGTERM, on_signal);
    std::signal(SIGINT, on_signal);
    std::signal(SIGPIPE, SIG_IGN);
    std::jthread alerts([](std::stop_token stop) {
      std::mutex m;
      std::condition_variable_any wait;
      while (!stop.stop_requested()) {
        try {
          check_price_alerts();
          check_pair_alerts();
        } catch (const std::exception &e) {
          std::cerr << "Alert check: " << e.what() << '\n';
        }
        std::unique_lock lock(m);
        wait.wait_for(lock, stop, std::chrono::seconds(30), [] { return false; });
      }
    });
    std::jthread monitor([&](std::stop_token stop) {
      while (!stop.stop_requested() && !shutdown_requested)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      if (shutdown_requested)
        server.stop();
    });
    std::cout << "StockAssist C++ backend on http://127.0.0.1:" << config.port << " ("
              << config.mode << ")\n"
              << std::flush;
    bool success = server.listen("127.0.0.1", config.port);
    monitor.request_stop();
    alerts.request_stop();
    alerts.join();
    backtests_stop();
    if (config.mode == "ibkr")
      ib_close();
    return success ? 0 : 1;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
