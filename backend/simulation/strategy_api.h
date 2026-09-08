#pragma once
#include <cstdint>
#include <cstdlib>
#include <string>

// Timestamp is UTC epoch milliseconds. bars[0..count-1] contains no future
// bars.
struct SA_Bar {
  int64_t timestamp;
  double open, high, low, close, volume;
};

// Convenience for flat numeric JSON parameters (e.g. {"fast": 10}).
inline double sa_parameter(const char *json, const char *key, double fallback) {
  std::string text(json);
  auto position = text.find(std::string("\"") + key + "\"");
  if (position == std::string::npos)
    return fallback;
  position = text.find(':', position);
  if (position == std::string::npos)
    return fallback;
  char *end = nullptr;
  const char *start = text.c_str() + position + 1;
  double value = std::strtod(start, &end);
  return end == start ? fallback : value;
}

// Return -1 to hold (no order), or 0..1 as target invested fraction.
// Static/global C++ state persists for this run only.
extern "C" double on_bar(const SA_Bar *bars, int count, double cash,
                         int64_t shares, const char *params_json);

// Multi-asset ABI: histories share timestamps; only [0, count) is visible.
struct SA_Asset {
  const char *symbol;
  const SA_Bar *bars;
  int64_t shares; // Negative for a short position.
};
// Return 0 to hold; return 1 after writing one target equity weight per asset.
// Positive weights are long, negative weights short. Total absolute weights
// must not exceed the run's max_gross_exposure setting.
extern "C" int on_portfolio(const SA_Asset *assets, int asset_count, int count,
                            double cash, double equity, const char *params_json,
                            double *target_weights);
