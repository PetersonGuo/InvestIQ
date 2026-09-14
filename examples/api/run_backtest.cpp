// Standalone HTTP client: compile with the repository's vendored headers.
// This is a main() program, not a strategy callback loaded by the worker.
#include "httplib.h"
#include "json.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>
using Json = nlohmann::json;

Json decode(const httplib::Result &response) {
  if (!response) throw std::runtime_error("Cannot reach StockAssist: " + httplib::to_string(response.error()));
  Json value = Json::parse(response->body);
  if (response->status >= 400)
    throw std::runtime_error("HTTP " + std::to_string(response->status) + ": " + value.value("detail", response->body));
  return value;
}
int main(int argc, char **argv) {
  if (argc < 6 || argc > 7) {
    std::cerr << "Usage: stockassist-client SOURCE TICKERS START_DATE END_DATE OUTPUT_JSON [PORT]\n"
                 "TICKERS is comma-separated. Defaults: daily bars, warmup 30, cash $100000,\n"
                 "$1 commission, 5 bps slippage, empty numeric parameters. Edit payload below as needed.\n";
    return 2;
  }
  try {
    const std::string filename = argv[1];
    const auto dot = filename.rfind('.');
    const auto extension = dot == std::string::npos ? "" : filename.substr(dot);
    if (extension != ".cpp" && extension != ".py") throw std::runtime_error("Source must be .cpp or .py");
    std::ifstream source(filename);
    if (!source) throw std::runtime_error("Cannot read strategy source");
    std::ostringstream code;
    code << source.rdbuf();
    Json tickers = Json::array();
    std::istringstream symbols(argv[2]);
    for (std::string ticker; std::getline(symbols, ticker, ',');) tickers.push_back(ticker);
    if (tickers.empty()) throw std::runtime_error("At least one ticker is required");
    int port = argc == 7 ? std::stoi(argv[6]) : 8000;
    if (port < 1 || port > 65535) throw std::runtime_error("Invalid port");
    httplib::Client client("127.0.0.1", port);
    client.set_connection_timeout(5);
    client.set_read_timeout(45);
    auto health = decode(client.Get("/health"));
    std::cout << "Backend: " << health.at("backend") << " · provider: " << health.at("data_mode") << '\n';
    Json payload = {{"name", filename}, {"language", extension == ".cpp" ? "cpp" : "python"},
                    {"code", code.str()}, {"ticker", tickers[0]}, {"tickers", tickers},
                    {"start_date", argv[3]}, {"end_date", argv[4]}, {"interval", "1d"},
                    {"params", Json::object()}, {"warmup", 30}, {"initial_cash", 100000},
                    {"commission", 1}, {"slippage_bps", 5}, {"force_rerun", false}};
    auto run = decode(client.Post("/api/backtests", payload.dump(), "application/json"));
    const auto id = run.at("id").get<std::string>();
    std::cout << "Run: " << id << " · " << run.at("status") << '\n';
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(10);
    while (run.at("status") == "queued" || run.at("status") == "running") {
      if (std::chrono::steady_clock::now() >= deadline)
        throw std::runtime_error("Polling timed out; resume GET /api/backtests/" + id);
      std::this_thread::sleep_for(std::chrono::seconds(1));
      run = decode(client.Get("/api/backtests/" + id));
    }
    if (run.at("status") != "completed")
      throw std::runtime_error("Run failed: " + run.value("error", std::string("Unknown error")));
    std::ofstream output(argv[5]);
    output << run.dump(2) << '\n';
    if (!output) throw std::runtime_error("Could not save result");
    std::cout << run.at("result").at("risk_comparison").dump(2) << '\n';
    std::cout << "Saved report to " << argv[5] << '\n';
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
