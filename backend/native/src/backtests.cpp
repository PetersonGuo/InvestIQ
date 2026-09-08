#include "backtests.hpp"
#include <condition_variable>
#include <queue>
namespace sa {
namespace {
J presets = J::array(
    {{{"template_id", "moving_average"},
      {"name", "Moving average crossover"},
      {"description", "Trend following: invest when the fast closing-price average is above "
                      "the slow average; exit otherwise. Can whipsaw in sideways markets."},
      {"params", {{"fast", 10}, {"slow", 30}}},
      {"warmup", 30}},
     {{"template_id", "buy_and_hold"},
      {"name", "Buy and hold"},
      {"description", "Baseline: invest at the first eligible next open, then hold. Useful "
                      "for checking whether an active strategy adds value after costs."},
      {"params", J::object()},
      {"warmup", 0}},
     {{"template_id", "channel_breakout"},
      {"name", "Price-channel breakout"},
      {"description", "Momentum: buy when the close breaks above the prior "
                      "entry-window high; exit below the prior exit-window "
                      "low. The current bar is excluded from both channels."},
      {"params", {{"entry_period", 20}, {"exit_period", 10}}},
      {"warmup", 20}},
     {{"template_id", "mean_reversion"},
      {"name", "Z-score mean reversion"},
      {"description", "Buy after a close falls at least entry_z standard deviations below its "
                      "rolling mean; exit when its z-score reaches exit_z. Can struggle "
                      "during persistent declines."},
      {"params", {{"lookback", 20}, {"entry_z", 2}, {"exit_z", 0}}},
      {"warmup", 20}}});
std::pair<J, J> load_bars(const J &q) {
  std::string interval = q["interval"], start = q["start_date"], end = q["end_date"],
              ticker = q["ticker"];
  double a = parse_time(start), b = parse_time(end);
  if (interval == "1d") {
    auto data = history(ticker, 500);
    J bars = J::array();
    for (auto &bar : data["bars"])
      if (bar["time"] >= start && bar["time"] <= end)
        bars.push_back(bar);
    return {data, bars};
  }
  std::map<std::string, J> collected;
  std::string before = end;
  for (int page = 0; page < 40; page++) {
    auto data = history(ticker, 500, before, interval);
    auto &bars = data["bars"];
    if (bars.empty())
      throw Error(422, "No bars returned before the requested start was "
                       "reached. Choose a narrower window.");
    for (auto &bar : bars) {
      double t = parse_time(bar["time"]);
      if (t >= a && t < b)
        collected[bar["time"]] = bar;
    }
    if (collected.size() > 50000)
      throw Error(422, "Intraday backtests support up to 50,000 bars. Narrow "
                       "the window or choose a larger interval.");
    std::string oldest = bars[0]["time"];
    if (parse_time(oldest) >= parse_time(before))
      throw Error(502, "Historical paging did not advance.");
    if (parse_time(oldest) <= a) {
      J sorted = J::array();
      for (auto &[_, bar] : collected)
        sorted.push_back(bar);
      return {data, sorted};
    }
    before = oldest;
  }
  throw Error(422, "This window needs more than 40 historical pages. Choose a "
                   "shorter window or larger interval.");
}
J execute(const J &q, const J &bars) {
  TempDir dir;
  bool cpp = q["language"] == "cpp";
  auto source = dir.path / (cpp ? "strategy.cpp" : "strategy.py");
  write(source, q["code"]);
  auto module = source;
  if (cpp) {
    fs::copy_file(config.root / "simulation/strategy_api.h", dir.path / "strategy_api.h");
    module = dir.path / "strategy.so";
    std::vector<std::string> args = {env("CXX", "c++"), "-std=c++17", "-O3"};
#ifdef __APPLE__
    args.push_back("-dynamiclib");
#else
    args.push_back("-shared");
    args.push_back("-fPIC");
#endif
    args.insert(args.end(), {source.string(), "-o", module.string()});
    process(args, dir.path, 30, dir.path / "compile.log");
  }
  J settings = J::object();
  for (auto key : {"initial_cash", "commission", "slippage_bps", "warmup", "interval"})
    settings[key] = q[key];
  J input = {{"language", q["language"]},
             {"module_path", module.string()},
             {"params", q["params"]},
             {"bars", bars},
             {"settings", settings}};
  write(dir.path / "request.json", input.dump());
  process({config.worker.string(), (dir.path / "request.json").string(),
           (dir.path / "result.json").string()},
          dir.path, 12, dir.path / "worker.log");
  if (!fs::exists(dir.path / "result.json"))
    throw Error(422, "Strategy exited without producing a result.");
  auto result = J::parse(read(dir.path / "result.json"));
  auto logs = read(dir.path / "worker.log").substr(0, 4000);
  if (result.contains("error"))
    throw Error(422, result["error"].get<std::string>() + "\n" + logs);
  result = result["result"];
  result["logs"] = logs;
  return result;
}
struct Work {
  std::string id, key;
  J request;
};
class Jobs {
  std::mutex mutex;
  std::condition_variable cv;
  std::queue<Work> queue;
  std::vector<std::thread> workers;
  int active = 0;
  bool stopping = false;
  void run(const Work &work) {
    auto &q = work.request;
    try {
      Db d;
      d.exec("UPDATE backtest_runs SET status='running' WHERE id=?", {work.id});
      auto [data, bars] = load_bars(q);
      if (bars.size() < 2 || bars.size() <= size_t(q["warmup"].get<int>() + 1))
        throw Error(422, "Not enough available bars in this date range after warmup.");
      std::string data_hash = hash(bars.dump()),
                  data_key = hash(work.key + data["source"].get<std::string>() + data_hash);
      J result = nullptr;
      if (!q["force_rerun"].get<bool>()) {
        auto cached = d.rows("SELECT r.result_json FROM backtest_cache c JOIN backtest_runs r "
                             "ON r.id=c.run_id WHERE c.key=? AND r.status='completed'",
                             {data_key});
        if (!cached.empty())
          result = J::parse(cached[0]["result_json"].get<std::string>());
      }
      if (result.is_null())
        result = execute(q, bars);
      std::string interval = q["interval"];
      result["engine_version"] = "3.0.0-native-cpp";
      result["interval"] = interval;
      result["data"] = {{"ticker", q["ticker"]},
                        {"source", data["source"]},
                        {"start", bars[0]["time"]},
                        {"end", bars.back()["time"]},
                        {"bar_count", bars.size()},
                        {"sha256", data_hash},
                        {"bars", bars}};
      result["assumptions"] = J::array(
          {interval + " completed regular-session bars; long-only, one symbol, "
                      "whole shares, no leverage.",
           "Signals at close fill at the next available bar open, with fees "
           "and adverse slippage.",
           "Buy-and-hold benchmark enters on the first tradable open after "
           "warmup using the same costs.",
           "Open positions are marked to the final close; they are not "
           "forcibly sold.",
           interval == "1d" ? "Sharpe uses 252 sessions/year and zero risk-free rate; win "
                              "rate is per realized sell fill."
                            : "UTC timestamps and exclusive end time. Annualized return and "
                              "Sharpe are omitted for intraday runs. Win rate is per "
                              "realized sell fill.",
           "No dividends, borrow, taxes, liquidity caps, or delisted-universe "
           "correction. Historical bars may be split-adjusted.",
           "Today's scanner candidates are not a point-in-time historical "
           "universe. Avoid interpreting selection-biased results as expected "
           "returns."});
      d.exec("BEGIN IMMEDIATE");
      d.exec("UPDATE backtest_runs SET status='completed',result_json=? WHERE "
             "id=?",
             {result.dump(), work.id});
      for (auto key : {work.key, data_key})
        d.exec("INSERT OR REPLACE INTO backtest_cache VALUES(?,?,?,?)",
               {key, work.id, now() + 3600, ny_date()});
      d.exec("COMMIT");
    } catch (const std::exception &e) {
      try {
        Db d;
        d.exec("UPDATE backtest_runs SET status='failed',error=? WHERE id=?",
               {std::string(e.what()).substr(0, 8000), work.id});
      } catch (...) {
      }
    }
  }

public:
  Jobs() {
    for (int i = 0; i < 2; i++)
      workers.emplace_back([this] {
        for (;;) {
          Work work;
          {
            std::unique_lock lock(mutex);
            cv.wait(lock, [this] { return stopping || !queue.empty(); });
            if (stopping && queue.empty())
              return;
            work = queue.front();
            queue.pop();
          }
          run(work);
          {
            std::lock_guard lock(mutex);
            --active;
          }
        }
      });
  }
  ~Jobs() { stop(); }
  void stop() {
    {
      std::lock_guard lock(mutex);
      stopping = true;
    }
    cv.notify_all();
    for (auto &t : workers)
      if (t.joinable())
        t.join();
  }
  J submit(const J &q) {
    std::lock_guard lock(mutex);
    J input = q;
    input.erase("name");
    input.erase("force_rerun");
    std::string key = hash(J::array({input, config.mode, SA_BUILD_ID}).dump());
    Db d;
    if (!q["force_rerun"].get<bool>()) {
      auto rows = d.rows("SELECT r.id,r.status FROM backtest_cache c JOIN backtest_runs r ON "
                         "r.id=c.run_id WHERE c.key=? AND c.expires>? AND c.day=? AND "
                         "r.status IN ('completed','queued','running')",
                         {key, now(), ny_date()});
      if (!rows.empty()) {
        auto result = backtest_get(rows[0]["id"]);
        result["cache_hit"] = rows[0]["status"] == "completed";
        return result;
      }
    }
    if (stopping)
      throw Error(503, "Backend is shutting down.");
    if (active >= 2)
      throw Error(429, "Two backtests are already running. Wait for one to finish.");
    auto run_id = id();
    d.exec("BEGIN IMMEDIATE");
    d.exec("INSERT INTO backtest_runs VALUES(?,?,?,?,NULL,NULL)",
           {run_id, "queued", stamp(), q.dump()});
    d.exec("INSERT OR REPLACE INTO backtest_cache VALUES(?,?,?,?)",
           {key, run_id, now() + 3600, ny_date()});
    d.exec("COMMIT");
    ++active;
    queue.push({run_id, key, q});
    cv.notify_one();
    return {{"id", run_id}, {"status", "queued"}, {"cache_hit", false}};
  }
};
Jobs &jobs() {
  static Jobs instance;
  return instance;
}
} // namespace
J examples() {
  J result = J::array();
  for (auto &p : presets)
    for (auto language : {"python", "cpp"}) {
      J entry = p;
      entry["language"] = language;
      entry["code"] = read(config.root / "simulation/examples" /
                           (p["template_id"].get<std::string>() +
                            (std::string(language) == "python" ? ".py" : ".cpp")));
      result.push_back(entry);
    }
  return result;
}
J backtest_get(const std::string &id) {
  Db d;
  auto rows = d.rows("SELECT * FROM backtest_runs WHERE id=?", {id});
  if (rows.empty())
    throw Error(404, "Backtest not found.");
  auto r = rows[0];
  r["request"] = J::parse(r["request_json"].get<std::string>());
  r["result"] =
      r["result_json"].is_null() ? J(nullptr) : J::parse(r["result_json"].get<std::string>());
  r.erase("request_json");
  r.erase("result_json");
  return r;
}
J backtest_list() {
  Db d;
  auto rows = d.rows("SELECT id,status,created_at,request_json,error FROM "
                     "backtest_runs ORDER BY created_at DESC LIMIT 30");
  for (auto &row : rows) {
    auto q = J::parse(row["request_json"].get<std::string>());
    row.erase("request_json");
    for (auto key : {"name", "language", "ticker"})
      row[key] = q[key];
  }
  return rows;
}
J backtest_submit(const J &q) { return jobs().submit(q); }
void backtests_stop() { jobs().stop(); }
} // namespace sa
