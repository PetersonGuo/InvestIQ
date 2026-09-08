#include "strategy_api.h"

extern "C" double on_bar(const SA_Bar *bars, int count, double cash,
                         int64_t shares, const char *params_json) {
  return shares == 0 ? 1.0 : -1.0;
}
