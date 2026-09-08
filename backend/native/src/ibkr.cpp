// Read-only adapter over the user's installed IBKR C++ SDK. No broker order
// calls.
#include "Contract.h"
#include "DefaultEWrapper.h"
#include "EClientSocket.h"
#include "EReader.h"
#include "EReaderOSSignal.h"
#include "HistoricalTickLast.h"
#include "ScannerSubscription.h"
#include "bar.h"
#include "market.hpp"
#include <atomic>
#include <deque>
#include <iomanip>
#include <sstream>
namespace sa {
namespace {
class Broker : public DefaultEWrapper {
  std::mutex mutex;
  EReaderOSSignal signal{50};
  EClientSocket client{this, &signal};
  std::unique_ptr<EReader> reader;
  std::atomic<bool> connected{false};
  int serial = 1000, current = -1;
  bool ready = false, done = false;
  std::string error_message, last_error;
  J values = J::array();
  std::vector<Contract> found;
  std::map<std::string, Contract> contracts;
  std::map<std::string, std::pair<double, J>> cache;
  std::deque<double> requests;
  std::map<std::string, double> identical;
  double last_small = 0, last_search = 0, last_connect_failure = 0;
  std::string interval = "1d", before;
  void disconnect() {
    connected = false;
    client.eDisconnect();
    if (reader) {
      reader->stop();
      reader.reset();
    }
    ready = false;
    contracts.clear();
    cache.clear();
  }
  void wait(bool handshake = false) {
    double deadline = now() + (handshake ? 4 : 9);
    while (!(handshake ? ready : done) && error_message.empty() && client.isConnected() &&
           now() < deadline) {
      signal.waitForSignal();
      reader->processMsgs();
    }
    if (!error_message.empty())
      throw Error(502, error_message);
    if (!(handshake ? ready : done)) {
      disconnect();
      throw Error(504, "IBKR request timed out. Check TWS and retry.");
    }
  }
  void connect() {
    if (client.isConnected() && ready)
      return;
    if (now() - last_connect_failure < 3)
      throw Error(503, "IBKR is offline. Open TWS/IB Gateway and enable socket API access.");
    disconnect();
    error_message.clear();
    if (!client.eConnect(config.ib_host.c_str(), config.ib_port, config.ib_client)) {
      last_connect_failure = now();
      throw Error(503, "IBKR is offline. Open TWS/IB Gateway and enable socket "
                       "API access with Read-Only API enabled.");
    }
    reader = std::make_unique<EReader>(&client, &signal);
    reader->start();
    try {
      wait(true);
      connected = true;
    } catch (...) {
      last_connect_failure = now();
      disconnect();
      throw;
    }
  }
  int start() {
    current = ++serial;
    done = false;
    error_message.clear();
    values = J::array();
    found.clear();
    return current;
  }
  Contract qualify(const std::string &s) {
    if (contracts.count(s))
      return contracts.at(s);
    Contract c;
    c.symbol = s;
    c.secType = "STK";
    c.exchange = "SMART";
    c.currency = "USD";
    int req = start();
    client.reqContractDetails(req, c);
    wait();
    if (found.size() != 1)
      throw Error(404, "IBKR could not uniquely identify this USD stock.");
    c = found.front();
    c.exchange = "SMART";
    contracts[s] = c;
    return c;
  }
  std::string end_time(const std::string &s) {
    if (s.empty())
      return "";
    time_t t = parse_time(s);
    std::tm tm{};
    gmtime_r(&t, &tm);
    char b[40];
    strftime(b, sizeof(b), "%Y%m%d %H:%M:%S UTC", &tm);
    return b;
  }
  void pace(const std::string &key, bool small) {
    if (!small)
      return;
    double n = now();
    while (!requests.empty() && n - requests.front() >= 600)
      requests.pop_front();
    if (requests.size() >= 55)
      throw Error(429, "IBKR historical request budget reached. Retry in " +
                           std::to_string(int(601 - (n - requests.front()))) + " seconds.");
    double delay = std::max(0.0, .45 - (n - last_small));
    if (identical.count(key))
      delay = std::max(delay, 15.1 - (n - identical[key]));
    if (delay > 3)
      throw Error(429, "IBKR requires 15 seconds between identical small-bar requests.");
    std::this_thread::sleep_for(std::chrono::duration<double>(delay));
    last_small = now();
    requests.push_back(last_small);
    for (auto it = identical.begin(); it != identical.end();)
      if (now() - it->second > 16)
        it = identical.erase(it);
      else
        ++it;
    identical[key] = last_small;
  }

public:
  ~Broker() { disconnect(); }
  J status() {
    std::unique_lock lock(mutex, std::try_to_lock);
    return {{"provider", "ibkr"},
            {"connected", connected.load()},
            {"host", config.ib_host},
            {"port", config.ib_port},
            {"client_id", config.ib_client},
            {"last_error", lock.owns_lock() && !last_error.empty() ? J(last_error) : J(nullptr)},
            {"price_type", "completed_daily_close"},
            {"transport", "native-cpp-sdk"}};
  }
  void close() {
    std::lock_guard lock(mutex);
    disconnect();
  }
  J request(const std::string &op, const J &v) {
    std::lock_guard lock(mutex);
    try {
      connect();
      std::string key = J::array({op, v, ny_date()}).dump();
      auto it = cache.find(key);
      if (it != cache.end() && now() - it->second.first < 60)
        return it->second.second;
      if (op == "search") {
        double delay = std::max(0.0, 1.1 - (now() - last_search));
        std::this_thread::sleep_for(std::chrono::duration<double>(delay));
        last_search = now();
        int req = start();
        client.reqMatchingSymbols(req, v.get<std::string>());
        wait();
      } else if (op == "scanner") {
        ScannerSubscription sub;
        sub.instrument = "STK";
        sub.locationCode = "STK.US.MAJOR";
        sub.scanCode = v.at("scan_code");
        sub.numberOfRows = v.at("limit");
        sub.abovePrice = v.at("min_price");
        sub.belowPrice = v.at("max_price");
        sub.aboveVolume = v.at("min_volume");
        int req = start();
        client.reqScannerSubscription(req, sub, TagValueListSPtr{}, TagValueListSPtr{});
        try {
          wait();
        } catch (...) {
          client.cancelScannerSubscription(req);
          throw;
        }
        client.cancelScannerSubscription(req);
      } else {
        auto s = v.at("symbol").get<std::string>();
        interval = v.value("interval", std::string("1d"));
        before = v.value("before", std::string(""));
        auto r = resolution(interval);
        if (op != "ticks" && r.seconds <= 30 && !before.empty() &&
            parse_time(before) < now() - 183 * 86400)
          throw Error(422, "IBKR small bars are limited to roughly six months. "
                           "Choose a coarser interval for older data.");
        auto contract = qualify(s);
        pace(key, op == "ticks" || r.seconds <= 30);
        int req = start();
        if (op == "ticks") {
          client.reqHistoricalTicks(req, contract, "", end_time(before.empty() ? stamp() : before),
                                    1000, "TRADES", 1, false, TagValueListSPtr{});
          wait();
        } else {
          client.reqHistoricalData(req, contract, end_time(before), r.duration, r.size, "TRADES", 1,
                                   interval == "1d" ? 1 : 2, false, TagValueListSPtr{});
          try {
            wait();
          } catch (...) {
            client.cancelHistoricalData(req);
            throw;
          }
        }
      }
      if (cache.size() >= 128)
        cache.clear();
      cache[key] = {now(), values};
      last_error.clear();
      return values;
    } catch (const std::exception &e) {
      last_error = e.what();
      throw;
    }
  }
  void nextValidId(OrderId) override { ready = true; }
  void connectionClosed() override {
    connected = false;
    ready = false;
  }
  void error(int req, time_t, int code, const std::string &msg, const std::string &) override {
    if (code == 2104 || code == 2106 || code == 2158 || code == 2108 || code == 2176 ||
        code == 2107)
      return;
    if (req == current || code == 326 || code == 502 || code == 504) {
      error_message = "IBKR rejected the data request (code " + std::to_string(code) + "). " + msg +
                      " Check market-data subscriptions/API permissions and "
                      "request pacing.";
      done = true;
    }
  }
  void contractDetails(int req, const ContractDetails &d) override {
    if (req == current)
      found.push_back(d.contract);
  }
  void contractDetailsEnd(int req) override {
    if (req == current)
      done = true;
  }
  void symbolSamples(int req, const std::vector<ContractDescription> &descriptions) override {
    if (req != current)
      return;
    for (auto &d : descriptions)
      if (d.contract.secType == "STK" && d.contract.currency == "USD")
        values.push_back({{"ticker", d.contract.symbol},
                          {"name", d.contract.description.empty() ? d.contract.symbol
                                                                  : d.contract.description}});
    done = true;
  }
  void historicalData(TickerId req, const Bar &bar) override {
    if (req != current)
      return;
    try {
      std::string time;
      if (interval == "1d") {
        if (bar.time.size() != 8)
          throw Error(502, "Invalid IBKR daily timestamp.");
        time = bar.time.substr(0, 4) + "-" + bar.time.substr(4, 2) + "-" + bar.time.substr(6, 2);
        if (time >= ny_date())
          return;
      } else {
        double start = std::stod(bar.time);
        if (start + resolution(interval).seconds > now())
          return;
        time = stamp(start);
      }
      if (!before.empty() && parse_time(time) >= parse_time(before))
        return;
      double volume = DecimalFunctions::decimalToDouble(bar.volume);
      for (double p : {bar.open, bar.high, bar.low, bar.close})
        if (!std::isfinite(p) || p <= 0)
          throw Error(502, "Invalid IBKR price bar.");
      if (!std::isfinite(volume) || bar.low > std::min(bar.open, bar.close) ||
          bar.high < std::max(bar.open, bar.close))
        throw Error(502, "Invalid IBKR OHLCV range.");
      values.push_back({{"time", time},
                        {"open", bar.open},
                        {"high", bar.high},
                        {"low", bar.low},
                        {"close", bar.close},
                        {"volume", std::max(0.0, volume)}});
    } catch (const std::exception &e) {
      error_message = e.what();
      done = true;
    }
  }
  void historicalDataEnd(int req, const std::string &, const std::string &) override {
    if (req != current)
      return;
    std::map<std::string, J> sorted;
    for (auto &b : values)
      sorted[b["time"]] = b;
    values = J::array();
    for (auto &[_, bar] : sorted)
      values.push_back(bar);
    done = true;
  }
  void scannerData(int req, int rank, const ContractDetails &d, const std::string &,
                   const std::string &, const std::string &, const std::string &) override {
    if (req == current)
      values.push_back({{"rank", rank},
                        {"ticker", d.contract.symbol},
                        {"name", d.longName.empty() ? d.contract.symbol : d.longName},
                        {"exchange", d.contract.primaryExchange}});
  }
  void scannerDataEnd(int req) override {
    if (req == current)
      done = true;
  }
  void historicalTicksLast(int req, const std::vector<HistoricalTickLast> &ticks,
                           bool finished) override {
    if (req != current)
      return;
    for (auto &t : ticks) {
      double size = DecimalFunctions::decimalToDouble(t.size);
      if (!std::isfinite(t.price) || t.price <= 0 || !std::isfinite(size) || size < 0) {
        error_message = "Invalid IBKR trade record.";
        done = true;
        return;
      }
      values.push_back({{"time", stamp(t.time)},
                        {"price", t.price},
                        {"size", size},
                        {"exchange", t.exchange},
                        {"conditions", t.specialConditions},
                        {"past_limit", t.tickAttribLast.pastLimit},
                        {"unreported", t.tickAttribLast.unreported}});
    }
    done = finished;
  }
};
Broker &broker() {
  static Broker b;
  return b;
}
} // namespace
J ib_request(const std::string &op, const J &v) { return broker().request(op, v); }
J ib_status() { return broker().status(); }
void ib_close() { broker().close(); }
} // namespace sa
