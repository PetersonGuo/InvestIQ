#include "strategy_api.h"
#include <cmath>

extern "C" double on_bar(const SA_Bar *bars, int count, double cash,
                         int64_t shares, const char *params_json) {
  double period = sa_parameter(params_json, "lookback", 20);
  double entry_z = sa_parameter(params_json, "entry_z", 2);
  double exit_z = sa_parameter(params_json, "exit_z", 0);
  if (period < 2 || period > 250 || std::floor(period) != period ||
      entry_z <= 0 || exit_z <= -entry_z)
    return 2.0;
  int lookback = static_cast<int>(period);
  if (count < lookback)
    return -1.0;
  double average = 0.0;
  for (int i = count - lookback; i < count; ++i)
    average += bars[i].close;
  average /= lookback;
  double variance = 0.0;
  for (int i = count - lookback; i < count; ++i) {
    double distance = bars[i].close - average;
    variance += distance * distance;
  }
  double deviation = std::sqrt(variance / lookback);
  if (deviation == 0)
    return -1.0;
  double z_score = (bars[count - 1].close - average) / deviation;
  if (z_score <= -entry_z)
    return 1.0;
  if (z_score >= exit_z)
    return 0.0;
  return -1.0;
}
