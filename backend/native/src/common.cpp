#include "common.hpp"
#include <fcntl.h>
#include <iomanip>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <regex>
#include <signal.h>
#include <spawn.h>
#include <sstream>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;
namespace sa {
Config config;
std::string read(const fs::path &p) {
  std::ifstream f(p, std::ios::binary);
  if (!f)
    throw Error(500, "Could not read " + p.filename().string());
  return {std::istreambuf_iterator<char>(f), {}};
}
void write(const fs::path &p, const std::string &s) {
  std::ofstream f(p, std::ios::binary);
  if (!(f << s))
    throw Error(500, "Could not write local file.");
}
std::string env(const std::string &k, const std::string &fallback) {
  const char *v = getenv(k.c_str());
  return v ? std::string(v) : fallback;
}
void load_config() {
  config.root = fs::weakly_canonical(SA_BACKEND_ROOT);
  std::ifstream f(config.root / ".env");
  std::string line;
  std::regex assignment(R"(^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$)");
  while (std::getline(f, line)) {
    std::smatch m;
    if (!std::regex_match(line, m, assignment))
      continue;
    std::string v = m[2];
    if (v.size() > 1 &&
        ((v.front() == '\'' && v.back() == '\'') || (v.front() == '"' && v.back() == '"')))
      v = v.substr(1, v.size() - 2);
    if (!getenv(m[1].str().c_str()))
      setenv(m[1].str().c_str(), v.c_str(), 0);
  }
  config.mode = env("STOCKASSIST_DATA_MODE", "demo");
  if (config.mode != "demo" && config.mode != "ibkr" && config.mode != "massive")
    throw Error(500, "Invalid data mode.");
  config.db = env("STOCKASSIST_DB", (config.root / "data" / (config.mode + ".sqlite3")).string());
  config.worker = config.root / "build/native/stockassist-worker";
  config.python = config.root / ".venv/bin/python";
  config.ib_host = env("IBKR_HOST", "127.0.0.1");
  config.ib_port = std::stoi(env("IBKR_PORT", "7497"));
  config.ib_client = std::stoi(env("IBKR_CLIENT_ID", "71"));
  if (config.ib_port < 1 || config.ib_port > 65535 || config.ib_client < 1)
    throw Error(500, "Invalid IBKR port or client ID.");
  setenv("TZ", "America/New_York", 1);
  tzset();
}
std::string hash(const std::string &s) {
  unsigned char bytes[32];
  unsigned int n = 0;
  EVP_Digest(s.data(), s.size(), bytes, &n, EVP_sha256(), nullptr);
  std::ostringstream o;
  for (unsigned i = 0; i < n; ++i)
    o << std::hex << std::setw(2) << std::setfill('0') << int(bytes[i]);
  return o.str();
}
std::string id() {
  unsigned char b[16];
  if (RAND_bytes(b, 16) != 1)
    throw Error(500, "Random identifier unavailable.");
  b[6] = (b[6] & 15) | 64;
  b[8] = (b[8] & 63) | 128;
  std::ostringstream o;
  for (int i = 0; i < 16; i++) {
    if (i == 4 || i == 6 || i == 8 || i == 10)
      o << '-';
    o << std::hex << std::setw(2) << std::setfill('0') << int(b[i]);
  }
  return o.str();
}
double now() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}
std::string stamp(double s, bool day) {
  bool generated = s == -1;
  if (generated)
    s = now();
  time_t t = std::floor(s);
  std::tm tm{};
  gmtime_r(&t, &tm);
  char b[40];
  strftime(b, sizeof(b), day ? "%Y-%m-%d" : "%Y-%m-%dT%H:%M:%SZ", &tm);
  std::string result = b;
  if ((generated || s != std::floor(s)) && !day) {
    std::ostringstream fraction;
    fraction << "." << std::setw(6) << std::setfill('0') << int((s - std::floor(s)) * 1e6) << "Z";
    result.pop_back();
    result += fraction.str();
  }
  return result;
}
std::tm ny_time(double s) {
  time_t t = s;
  std::tm tm{};
  localtime_r(&t, &tm);
  return tm;
}
std::string ny_date(double s) {
  if (s < 0)
    s = now();
  auto tm = ny_time(s);
  char b[20];
  strftime(b, sizeof(b), "%Y-%m-%d", &tm);
  return b;
}
double parse_time(const std::string &s) {
  static const std::regex pattern(
      R"(^(\d{4})-(\d\d)-(\d\d)(?:T(\d\d):(\d\d)(?::(\d\d)(\.\d+)?)?(Z|[+-]\d\d:\d\d)?)?$)");
  std::smatch m;
  if (!std::regex_match(s, m, pattern))
    throw Error(422, "Use an ISO date or timestamp.");
  std::tm t{};
  if (std::stoi(m[1]) < 1)
    throw Error(422, "Invalid year.");
  t.tm_year = std::stoi(m[1]) - 1900;
  t.tm_mon = std::stoi(m[2]) - 1;
  t.tm_mday = std::stoi(m[3]);
  t.tm_hour = m[4].matched ? std::stoi(m[4]) : 0;
  t.tm_min = m[5].matched ? std::stoi(m[5]) : 0;
  t.tm_sec = m[6].matched ? std::stoi(m[6]) : 0;
  auto copy = t;
  auto result = timegm(&copy);
  if (copy.tm_year != t.tm_year || copy.tm_mon != t.tm_mon || copy.tm_mday != t.tm_mday ||
      copy.tm_hour != t.tm_hour || copy.tm_min != t.tm_min || copy.tm_sec != t.tm_sec)
    throw Error(422, "Invalid date or time.");
  double value = result;
  if (m[7].matched)
    value += std::stod(m[7]);
  std::string zone = m[8];
  if (zone.size() == 6) {
    int h = std::stoi(zone.substr(1, 2)), min = std::stoi(zone.substr(4, 2));
    if (h > 23 || min > 59)
      throw Error(422, "Invalid UTC offset.");
    value -= (zone[0] == '-' ? -1 : 1) * (h * 3600 + min * 60);
  }
  return value;
}
double rounded(double v, int places) {
  double f = std::pow(10, places);
  return std::nearbyint(v * f) / f;
}
std::string text(const J &j, const std::string &k, const std::string &d, size_t max) {
  if (!j.contains(k))
    return d;
  if (!j[k].is_string())
    throw Error(422, k + " must be text.");
  auto v = j[k].get<std::string>();
  if (v.size() > max)
    throw Error(422, k + " is too long.");
  return v;
}
double number(const J &j, const std::string &k, double d, double min, double max, bool integral) {
  if (!j.contains(k)) {
    if (!std::isfinite(d) || d < min || d > max)
      throw Error(422, k + " is required.");
    return d;
  }
  if (!j[k].is_number() || (integral && !j[k].is_number_integer()))
    throw Error(422, k + " must be a valid number.");
  double v = j[k];
  if (!std::isfinite(v) || v < min || v > max)
    throw Error(422, k + " is outside its allowed range.");
  return v;
}
bool boolean(const J &j, const std::string &k, bool d) {
  if (!j.contains(k))
    return d;
  if (!j[k].is_boolean())
    throw Error(422, k + " must be true or false.");
  return j[k];
}
std::string symbol(std::string s) {
  s = std::regex_replace(s, std::regex(R"(^\s+|\s+$)"), "");
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::toupper(c); });
  if (!std::regex_match(s, std::regex(R"([A-Z][A-Z0-9.\-]{0,14})")))
    throw Error(422, "Invalid ticker.");
  return s;
}
std::string choice(const J &j, const std::string &k, const std::string &d,
                   const std::vector<std::string> &allowed) {
  auto v = text(j, k, d);
  if (std::find(allowed.begin(), allowed.end(), v) == allowed.end())
    throw Error(422, "Invalid " + k + ".");
  return v;
}
Db::Db() {
  fs::create_directories(config.db.parent_path());
  if (sqlite3_open(config.db.c_str(), &db) != SQLITE_OK)
    throw Error(500, "Could not open database.");
  sqlite3_busy_timeout(db, 20000);
}
Db::~Db() {
  if (db)
    sqlite3_close(db);
}
J Db::rows(const std::string &sql, const J &args) {
  sqlite3_stmt *s = nullptr;
  if (sqlite3_prepare_v2(db, sql.c_str(), -1, &s, nullptr) != SQLITE_OK)
    throw Error(500, sqlite3_errmsg(db));
  struct Final {
    sqlite3_stmt *s;
    ~Final() { sqlite3_finalize(s); }
  } final{s};
  for (size_t i = 0; i < args.size(); i++) {
    const auto &v = args[i];
    if (v.is_null())
      sqlite3_bind_null(s, i + 1);
    else if (v.is_string()) {
      const auto &str = v.get_ref<const std::string &>();
      sqlite3_bind_text(s, i + 1, str.c_str(), str.size(), SQLITE_TRANSIENT);
    } else if (v.is_number_float())
      sqlite3_bind_double(s, i + 1, v.get<double>());
    else if (v.is_boolean())
      sqlite3_bind_int(s, i + 1, v.get<bool>());
    else
      sqlite3_bind_int64(s, i + 1, v.get<int64_t>());
  }
  J out = J::array();
  int rc;
  while ((rc = sqlite3_step(s)) == SQLITE_ROW) {
    J row = J::object();
    for (int i = 0; i < sqlite3_column_count(s); i++) {
      std::string k = sqlite3_column_name(s, i);
      switch (sqlite3_column_type(s, i)) {
      case SQLITE_INTEGER:
        row[k] = sqlite3_column_int64(s, i);
        break;
      case SQLITE_FLOAT:
        row[k] = sqlite3_column_double(s, i);
        break;
      case SQLITE_TEXT:
        row[k] = reinterpret_cast<const char *>(sqlite3_column_text(s, i));
        break;
      default:
        row[k] = nullptr;
      }
    }
    out.push_back(row);
  }
  if (rc != SQLITE_DONE)
    throw Error(500, sqlite3_errmsg(db));
  return out;
}
void Db::exec(const std::string &s, const J &a) { rows(s, a); }
int Db::changes() { return sqlite3_changes(db); }
void initialize() {
  Db d;
  for (const char *s : {"CREATE TABLE IF NOT EXISTS account (id INTEGER PRIMARY KEY "
                        "CHECK(id=1),cash_cents INTEGER NOT NULL)",
                        "INSERT OR IGNORE INTO account VALUES(1,10000000)",
                        "CREATE TABLE IF NOT EXISTS positions(ticker TEXT PRIMARY KEY,quantity "
                        "INTEGER NOT NULL,cost_cents INTEGER NOT NULL)",
                        "CREATE TABLE IF NOT EXISTS orders(id TEXT PRIMARY KEY,ticker TEXT NOT "
                        "NULL,side TEXT NOT NULL,quantity INTEGER NOT NULL,price_cents INTEGER "
                        "NOT NULL,created_at TEXT NOT NULL,price_as_of TEXT NOT NULL)",
                        "CREATE TABLE IF NOT EXISTS alerts(id TEXT PRIMARY KEY,ticker TEXT NOT "
                        "NULL,direction TEXT NOT NULL,threshold REAL NOT NULL,active INTEGER "
                        "NOT NULL DEFAULT 1,triggered_at TEXT,created_at TEXT NOT NULL)",
                        "CREATE TABLE IF NOT EXISTS strategies(id TEXT PRIMARY KEY,name TEXT "
                        "NOT NULL,language TEXT NOT NULL,code TEXT NOT NULL,params_json TEXT "
                        "NOT NULL,updated_at TEXT NOT NULL)",
                        "CREATE TABLE IF NOT EXISTS backtest_runs(id TEXT PRIMARY KEY,status "
                        "TEXT NOT NULL,created_at TEXT NOT NULL,request_json TEXT NOT "
                        "NULL,result_json TEXT,error TEXT)",
                        "CREATE TABLE IF NOT EXISTS backtest_cache(key TEXT PRIMARY KEY,run_id "
                        "TEXT NOT NULL,expires REAL NOT NULL,day TEXT NOT NULL)",
                        "CREATE TABLE IF NOT EXISTS pair_alerts(id TEXT PRIMARY KEY,config_json "
                        "TEXT NOT NULL,active INTEGER NOT NULL DEFAULT 1,version INTEGER NOT "
                        "NULL DEFAULT 1,created_at TEXT NOT NULL,updated_at TEXT NOT "
                        "NULL,last_checked_at TEXT,last_as_of TEXT,last_matched INTEGER NOT "
                        "NULL DEFAULT 0,latest_json TEXT,last_error TEXT,triggered_at TEXT)",
                        "CREATE TABLE IF NOT EXISTS pair_events(id TEXT PRIMARY KEY,alert_id "
                        "TEXT NOT NULL,version INTEGER NOT NULL,as_of TEXT NOT NULL,created_at "
                        "TEXT NOT NULL,event_json TEXT NOT NULL,UNIQUE(alert_id,version,as_of))",
                        "UPDATE backtest_runs SET status='failed',error='Backend restarted "
                        "before this run completed. Run it again.' WHERE status IN "
                        "('queued','running')"})
    d.exec(s);
}
TempDir::TempDir() {
  std::string t = (fs::temp_directory_path() / "stockassist-native-XXXXXX").string();
  std::vector<char> b(t.begin(), t.end());
  b.push_back(0);
  if (!mkdtemp(b.data()))
    throw Error(500, "Could not create worker directory.");
  path = b.data();
}
TempDir::~TempDir() {
  std::error_code e;
  fs::remove_all(path, e);
}
void process(const std::vector<std::string> &args, const fs::path &cwd, int timeout,
             const fs::path &log) {
  std::vector<char *> argv;
  for (auto &s : args)
    argv.push_back(const_cast<char *>(s.c_str()));
  argv.push_back(nullptr);
  std::vector<std::string> environment = {"PATH=" + env("PATH", "/usr/bin:/bin"),
                                          "LANG=en_US.UTF-8", "PYTHONUNBUFFERED=1",
                                          "PYTHONDONTWRITEBYTECODE=1"};
  std::vector<char *> envp;
  for (auto &s : environment)
    envp.push_back(s.data());
  envp.push_back(nullptr);
  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_addchdir_np(&actions, cwd.c_str());
  posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, log.c_str(),
                                   O_CREAT | O_WRONLY | O_TRUNC, 0600);
  posix_spawn_file_actions_adddup2(&actions, STDOUT_FILENO, STDERR_FILENO);
  posix_spawnattr_t attr;
  posix_spawnattr_init(&attr);
  posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETPGROUP);
  posix_spawnattr_setpgroup(&attr, 0);
  pid_t pid;
  int rc = posix_spawnp(&pid, args[0].c_str(), &actions, &attr, argv.data(), envp.data());
  posix_spawn_file_actions_destroy(&actions);
  posix_spawnattr_destroy(&attr);
  if (rc)
    throw Error(422, "Could not start strategy process or compiler.");
  int status = 0;
  auto until = std::chrono::steady_clock::now() + std::chrono::seconds(timeout);
  bool timed = false;
  while (waitpid(pid, &status, WNOHANG) == 0) {
    if (std::chrono::steady_clock::now() >= until) {
      timed = true;
      kill(-pid, SIGKILL);
      waitpid(pid, &status, 0);
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  kill(-pid, SIGKILL);
  if (timed)
    throw Error(422, "Strategy process exceeded its time limit.");
  if (!WIFEXITED(status) || WEXITSTATUS(status))
    throw Error(422, "Strategy process exited with an error. " + read(log).substr(0, 4000));
}
const std::map<std::string, Resolution> &resolutions() {
  static std::map<std::string, Resolution> r = {
      {"1s", {"1 secs", "1800 S", 1}},     {"5s", {"5 secs", "3600 S", 5}},
      {"10s", {"10 secs", "14400 S", 10}}, {"15s", {"15 secs", "14400 S", 15}},
      {"30s", {"30 secs", "1 D", 30}},     {"1m", {"1 min", "1 D", 60}},
      {"2m", {"2 mins", "2 D", 120}},      {"3m", {"3 mins", "1 W", 180}},
      {"5m", {"5 mins", "1 W", 300}},      {"15m", {"15 mins", "1 W", 900}},
      {"30m", {"30 mins", "1 M", 1800}},   {"1h", {"1 hour", "1 M", 3600}},
      {"4h", {"4 hours", "1 M", 14400}},   {"1d", {"1 day", "2 Y", 86400}}};
  return r;
}
const Resolution &resolution(const std::string &s) {
  auto &r = resolutions();
  if (!r.count(s))
    throw Error(422, "Unsupported candle interval.");
  return r.at(s);
}
J validate_strategy(J j) {
  if (!j.is_object())
    throw Error(422, "Expected an object.");
  J p = j.value("params", J::object());
  if (!p.is_object() || p.size() > 50)
    throw Error(422, "Parameters must be a numeric object with at most 50 entries.");
  for (auto &v : p)
    if (!v.is_number() || !std::isfinite(v.get<double>()))
      throw Error(422, "Strategy parameters must be finite numbers.");
  auto name = text(j, "name", "My strategy", 100), code = text(j, "code", "", 64000);
  if (name.empty() || code.empty())
    throw Error(422, "Name and code are required.");
  return {{"name", name},
          {"code", code},
          {"language", choice(j, "language", "", {"python", "cpp"})},
          {"params", p}};
}
J validate_backtest(J j) {
  J out = validate_strategy(j);
  std::string i = text(j, "interval", "1d"), start = text(j, "start_date"),
              end = text(j, "end_date");
  resolution(i);
  double a = parse_time(start), b = parse_time(end);
  if (a >= b)
    throw Error(422, "Start date must be before end date.");
  if (i == "1d" && (start.size() != 10 || end.size() != 10))
    throw Error(422, "Daily runs use dates.");
  if (i != "1d" && b - a > 31 * 86400)
    throw Error(422, "Intraday windows support up to 31 calendar days.");
  out.update({{"ticker", symbol(text(j, "ticker"))},
              {"interval", i},
              {"start_date", i == "1d" ? start : stamp(a)},
              {"end_date", i == "1d" ? end : stamp(b)},
              {"initial_cash", number(j, "initial_cash", 100000, 100, 1e8)},
              {"commission", number(j, "commission", 1, 0, 1000)},
              {"slippage_bps", number(j, "slippage_bps", 5, 0, 1000)},
              {"warmup", int(number(j, "warmup", 30, 0, 250, true))},
              {"force_rerun", boolean(j, "force_rerun")}});
  if (out["commission"].get<double>() > out["initial_cash"].get<double>())
    throw Error(422, "Commission cannot exceed capital.");
  return out;
}
J validate_pair(J j) {
  J c = {{"ticker_a", symbol(text(j, "ticker_a"))},
         {"ticker_b", symbol(text(j, "ticker_b"))},
         {"metric", choice(j, "metric", "zscore", {"zscore", "ratio"})},
         {"condition", choice(j, "condition", "outside", {"above", "below", "inside", "outside"})},
         {"threshold", number(j, "threshold", 2, -1e5, 1e5)},
         {"lookback", int(number(j, "lookback", 60, 20, 250, true))},
         {"hedge_ratio", number(j, "hedge_ratio", 1, 1e-12, 100)},
         {"repeat", boolean(j, "repeat")}};
  double t = c["threshold"];
  if (c["ticker_a"] == c["ticker_b"])
    throw Error(422, "Choose two different stocks.");
  if (c["metric"] == "ratio" &&
      (t <= 0 || (c["condition"] != "above" && c["condition"] != "below")))
    throw Error(422, "Ratio alerts require above/below and a positive threshold.");
  if (c["metric"] == "zscore" && std::abs(t) > 20)
    throw Error(422, "Z-score threshold must be between -20 and 20.");
  if ((c["condition"] == "inside" || c["condition"] == "outside") && t < 0)
    throw Error(422, "Band thresholds cannot be negative.");
  return c;
}
} // namespace sa
