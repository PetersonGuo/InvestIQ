#include "strategy_api.h"
#include <algorithm>
#include <cmath>

extern "C" double on_bar(const SA_Bar *bars, int count, double cash,
                         int64_t shares, const char *params_json) {
  double entry_value = sa_parameter(params_json, "entry_period", 20);
  double exit_value = sa_parameter(params_json, "exit_period", 10);
  if (entry_value < 1 || entry_value > 250 ||
      std::floor(entry_value) != entry_value || exit_value < 1 ||
      exit_value > 250 || std::floor(exit_value) != exit_value)
    return 2.0;
  int entry = static_cast<int>(entry_value),
      exit_period = static_cast<int>(exit_value);
  if (count <= std::max(entry, exit_period))
    return -1.0;
  double high = bars[count - entry - 1].high;
  double low = bars[count - exit_period - 1].low;
  for (int i = count - entry - 1; i < count - 1; ++i)
    high = std::max(high, bars[i].high);
  for (int i = count - exit_period - 1; i < count - 1; ++i)
    low = std::min(low, bars[i].low);
  if (bars[count - 1].close > high)
    return 1.0;
  if (bars[count - 1].close < low)
    return 0.0;
  return -1.0;
}
