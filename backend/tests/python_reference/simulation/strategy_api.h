#pragma once
#include <cstdint>
#include <cstdlib>
#include <string>

// Date is midnight UTC in milliseconds. bars[0..count-1] contains no future
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
