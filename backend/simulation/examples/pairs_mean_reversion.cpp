#include "pair_math.h"
#include "strategy_api.h"
extern "C" int on_portfolio(const SA_Asset *assets, int n, int count, double,
                            double, const char *params, double *weights) {
  if (n != 2)
    return -1;
  int window = sa_parameter(params, "lookback", 60);
  if (window < 20 || count < window + 1)
    return 0;
  std::vector<double> a, b;
  for (int i = count - window - 1; i < count - 1; ++i) {
    a.push_back(std::log(assets[0].bars[i].close));
    b.push_back(std::log(assets[1].bars[i].close));
  }
  auto fit = sa_pair_fit(a, b);
  if (!fit.valid || !fit.test_valid || std::abs(fit.beta) > 10)
    return 0;
  bool held = assets[0].shares || assets[1].shares;
  double z =
      (std::log(assets[0].bars[count - 1].close) -
       fit.beta * std::log(assets[1].bars[count - 1].close) - fit.intercept) /
      fit.sd;
  if (held && (std::abs(z) <= sa_parameter(params, "exit_z", .5) ||
               std::abs(z) >= sa_parameter(params, "stop_z", 4) ||
               fit.adf >= fit.critical)) {
    weights[0] = weights[1] = 0;
    return 1;
  }
  if (held || fit.adf >= fit.critical ||
      std::abs(z) < sa_parameter(params, "entry_z", 2) ||
      std::abs(z) >= sa_parameter(params, "stop_z", 4))
    return 0;
  double direction = z > 0 ? -1 : 1;
  weights[0] = direction / (1 + std::abs(fit.beta));
  weights[1] = -direction * fit.beta / (1 + std::abs(fit.beta));
  return 1;
}
