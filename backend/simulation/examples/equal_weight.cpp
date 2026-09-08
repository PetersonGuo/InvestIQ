#include "strategy_api.h"
extern "C" int on_portfolio(const SA_Asset *assets, int n, int, double, double,
                            const char *, double *weights) {
  for (int a = 0; a < n; ++a)
    if (assets[a].shares)
      return 0;
  for (int a = 0; a < n; ++a)
    weights[a] = 1.0 / n;
  return 1;
}
