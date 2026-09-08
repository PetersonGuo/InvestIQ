#include "strategy_api.h"
#include <stdexcept>

extern "C" double on_bar(const SA_Bar *bars, int count, double cash,
                         int64_t shares, const char *params_json) {
  int fast = static_cast<int>(sa_parameter(params_json, "fast", 10));
  int slow = static_cast<int>(sa_parameter(params_json, "slow", 30));
  if (fast < 1 || slow <= fast)
    return 2.0; // Engine reports invalid output.
  if (count < slow)
    return -1.0; // Hold during indicator warmup.
  double fast_sum = 0.0, slow_sum = 0.0;
  for (int i = count - slow; i < count; ++i)
    slow_sum += bars[i].close;
  for (int i = count - fast; i < count; ++i)
    fast_sum += bars[i].close;
  return fast_sum / fast > slow_sum / slow ? 1.0 : 0.0;
}
