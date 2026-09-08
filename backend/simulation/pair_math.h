#pragma once
#include <algorithm>
#include <cmath>
#include <vector>
// Engle–Granger: log(A)=intercept+beta*log(B)+residual; residual ADF has
// no constant and one fixed lag. MacKinnon (2010), N=2, constant trend,
// finite-sample 5% critical value. Assumes both log-price series are I(1).
struct SA_PairFit {
  bool valid = false, test_valid = false;
  double beta = 0, intercept = 0, mean = 0, sd = 0, adf = 0, critical = 0,
         half_life = 0;
};
inline SA_PairFit sa_pair_fit(const std::vector<double> &a,
                              const std::vector<double> &b) {
  SA_PairFit fit;
  int n = a.size();
  if (n < 20 || b.size() != a.size())
    return fit;
  double ma = 0, mb = 0;
  for (int i = 0; i < n; ++i) {
    ma += a[i] / n;
    mb += b[i] / n;
  }
  double cov = 0, var = 0;
  for (int i = 0; i < n; ++i) {
    cov += (a[i] - ma) * (b[i] - mb);
    var += (b[i] - mb) * (b[i] - mb);
  }
  if (var < 1e-18)
    return fit;
  fit.beta = cov / var;
  fit.intercept = ma - fit.beta * mb;
  std::vector<double> residual(n);
  double ss = 0;
  for (int i = 0; i < n; ++i) {
    residual[i] = a[i] - fit.beta * b[i] - fit.intercept;
    ss += residual[i] * residual[i];
  }
  fit.sd = std::sqrt(ss / n);
  fit.valid = std::isfinite(fit.sd) && fit.sd > 1e-10;
  if (!fit.valid)
    return fit;
  double xx = 0, xz = 0, zz = 0, xy = 0, zy = 0;
  for (int i = 2; i < n; ++i) {
    double x = residual[i - 1], z = residual[i - 1] - residual[i - 2],
           y = residual[i] - residual[i - 1];
    xx += x * x;
    xz += x * z;
    zz += z * z;
    xy += x * y;
    zy += z * y;
  }
  double determinant = xx * zz - xz * xz;
  if (determinant <= 1e-14 * xx * zz)
    return fit;
  double rho = (xy * zz - zy * xz) / determinant,
         lag = (zy * xx - xy * xz) / determinant, sse = 0;
  for (int i = 2; i < n; ++i) {
    double error = residual[i] - residual[i - 1] - rho * residual[i - 1] -
                   lag * (residual[i - 1] - residual[i - 2]);
    sse += error * error;
  }
  double se = std::sqrt(sse / (n - 4) * zz / determinant);
  if (se <= 1e-14)
    return fit;
  fit.adf = rho / se;
  double t = n - 1;
  fit.critical = -3.33613 - 6.1101 / t - 6.823 / (t * t);
  fit.test_valid = std::isfinite(fit.adf);
  // Descriptive AR(1) half-life, separate from the augmented test.
  double arxx = 0, arxy = 0;
  for (int i = 1; i < n; ++i) {
    arxx += residual[i - 1] * residual[i - 1];
    arxy += residual[i - 1] * residual[i];
  }
  double phi = arxx > 0 ? arxy / arxx : 0;
  if (phi > 0 && phi < 1)
    fit.half_life = -std::log(2.0) / std::log(phi);
  return fit;
}
