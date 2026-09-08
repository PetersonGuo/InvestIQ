#pragma once
#include "build_config.hpp"
#include "json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <sqlite3.h>
#include <string>
#include <thread>
#include <vector>
namespace sa {
using J = nlohmann::json;
namespace fs = std::filesystem;
struct Error : std::runtime_error {
  int status;
  Error(int code, const std::string &message) : runtime_error(message), status(code) {}
};
struct Config {
  fs::path root, db, worker, python;
  std::string mode, ib_host;
  int ib_port, ib_client, port = 8000;
};
extern Config config;
std::string read(const fs::path &path);
void write(const fs::path &, const std::string &);
std::string env(const std::string &, const std::string &fallback = "");
void load_config();
std::string id();
std::string hash(const std::string &);
std::string stamp(double seconds = -1, bool date_only = false);
double parse_time(const std::string &);
double now();
std::string ny_date(double seconds = -1);
std::tm ny_time(double seconds);
double rounded(double, int = 6);
std::string text(const J &, const std::string &, const std::string &fallback = "",
                 size_t max = 64000);
double number(const J &, const std::string &, double fallback, double min, double max,
              bool integral = false);
bool boolean(const J &, const std::string &, bool = false);
std::string symbol(std::string);
std::string choice(const J &, const std::string &, const std::string &,
                   const std::vector<std::string> &);
class Db {
  sqlite3 *db = nullptr;

public:
  Db();
  ~Db();
  Db(const Db &) = delete;
  void exec(const std::string &, const J &args = J::array());
  J rows(const std::string &, const J &args = J::array());
  int changes();
};
void initialize();
struct TempDir {
  fs::path path;
  TempDir();
  ~TempDir();
};
void process(const std::vector<std::string> &args, const fs::path &cwd, int timeout,
             const fs::path &log);
struct Resolution {
  std::string size, duration;
  int seconds;
};
const std::map<std::string, Resolution> &resolutions();
const Resolution &resolution(const std::string &);
J validate_strategy(J);
J validate_backtest(J);
J validate_pair(J);
} // namespace sa
