#pragma once
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

// Centered, standardized, twice-reorthogonalized QR. No ridge penalty:
// fail closed on rank deficiency rather than silently retaining factor risk.
struct SA_RVFit {
  std::vector<double> beta, residual;
  double intercept = 0, sd = 0, r2 = 0;
};
inline SA_RVFit sa_rv_fit(const std::vector<double> &y,
                          const std::vector<std::vector<double>> &x) {
  const int n = y.size(), p = x.size();
  if (n < std::max(20, 5 * (p + 1)))
    throw std::runtime_error("Insufficient observations for the number of "
                             "exposures (need at least 5 per coefficient).");
  SA_RVFit f;
  f.beta.resize(p);
  f.residual = y;
  double ym = std::accumulate(y.begin(), y.end(), 0.0) / n;
  std::vector<double> means(p), scales(p), rhs(p);
  std::vector<std::vector<double>> q(p, std::vector<double>(n)),
      r(p, std::vector<double>(p));
  for (int j = 0; j < p; ++j) {
    if (x[j].size() != y.size())
      throw std::runtime_error("Unaligned regression data.");
    means[j] = std::accumulate(x[j].begin(), x[j].end(), 0.0) / n;
    for (int i = 0; i < n; ++i)
      scales[j] += std::pow(x[j][i] - means[j], 2);
    scales[j] = std::sqrt(scales[j]);
    if (scales[j] < 1e-10)
      throw std::runtime_error("An exposure has no usable variation.");
    for (int i = 0; i < n; ++i)
      q[j][i] = (x[j][i] - means[j]) / scales[j];
    for (int pass = 0; pass < 2; ++pass)
      for (int k = 0; k < j; ++k) {
        double dot =
            std::inner_product(q[k].begin(), q[k].end(), q[j].begin(), 0.0);
        r[k][j] += dot;
        for (int i = 0; i < n; ++i)
          q[j][i] -= dot * q[k][i];
      }
    r[j][j] = std::sqrt(
        std::inner_product(q[j].begin(), q[j].end(), q[j].begin(), 0.0));
    if (r[j][j] < 1e-5)
      throw std::runtime_error("Exposures are collinear; choose independent "
                               "market, sector and factor proxies.");
    for (int i = 0; i < n; ++i) {
      q[j][i] /= r[j][j];
      rhs[j] += q[j][i] * (y[i] - ym);
    }
  }
  for (int j = p - 1; j >= 0; --j) {
    double b = rhs[j];
    for (int k = j + 1; k < p; ++k)
      b -= r[j][k] * f.beta[k];
    f.beta[j] = b / r[j][j];
  }
  f.intercept = ym;
  for (int j = 0; j < p; ++j) {
    f.beta[j] /= scales[j];
    f.intercept -= f.beta[j] * means[j];
  }
  double ss = 0, total = 0;
  for (int i = 0; i < n; ++i) {
    f.residual[i] -= f.intercept;
    for (int j = 0; j < p; ++j)
      f.residual[i] -= f.beta[j] * x[j][i];
    ss += f.residual[i] * f.residual[i];
    total += std::pow(y[i] - ym, 2);
  }
  f.sd = std::sqrt(ss / (n - p - 1));
  f.r2 = total > 1e-20 ? 1 - ss / total : 0;
  return f;
}
struct SA_RVResult {
  double score = 0, innovation = 0, sd = 0, r2 = 0, expected = 0,
         gross_scale = 1;
  std::vector<double> weights, before, after, path, reference_beta;
};
// prices: target, references, then market/sector/factor hedge instruments.
// mode 0 fixed return spread; 1 fitted basket; 2 one-bar residual lead-lag.
// All estimation ends BEFORE the latest return. Weights are dollar weights.
inline SA_RVResult sa_rv(const std::vector<std::vector<double>> &prices,
                         int refs, int window, int mode,
                         const std::vector<double> &fixed) {
  int n = prices.size(), count = prices.empty() ? 0 : prices[0].size(),
      factors = n - refs - 1;
  if (refs < 1 || refs > 20 || factors < 1 || factors > 8 || count < window + 3)
    throw std::runtime_error(
        "Not enough aligned history or invalid relationship dimensions.");
  int start = count - window - 2;
  std::vector<std::vector<double>> returns(n, std::vector<double>(window + 2));
  for (int a = 0; a < n; ++a) {
    if (prices[a].size() != size_t(count))
      throw std::runtime_error("Unaligned prices.");
    for (int i = 0; i < window + 2; ++i) {
      double prev = prices[a][start + i - 1], next = prices[a][start + i];
      if (prev <= 0 || next <= 0 || !std::isfinite(prev + next))
        throw std::runtime_error("Invalid price.");
      returns[a][i] = next / prev - 1;
    }
  }
  SA_RVResult result;
  result.weights.assign(n, 0);
  result.weights[0] = 1;
  std::vector<std::vector<double>> fx;
  for (int a = 1 + refs; a < n; ++a)
    fx.emplace_back(returns[a].begin() + 1, returns[a].end() - 1);
  std::vector<double> y(returns[0].begin() + 1, returns[0].end() - 1);
  SA_RVFit model;
  if (mode == 2) {
    // Remove contemporaneous factors from each security first, then fit
    // residual target[t] against residual leaders[t-1]. FWL factors and lag
    // model share a prior-only estimation window.
    auto target = sa_rv_fit(y, fx);
    std::vector<std::vector<double>> lagged;
    std::vector<double> current;
    for (int a = 1; a <= refs; ++a) {
      std::vector<double> ry(returns[a].begin() + 1, returns[a].end() - 1);
      auto leader = sa_rv_fit(ry, fx);
      std::vector<double> lag;
      for (int i = 0; i < window; ++i) {
        double v = returns[a][i] - leader.intercept;
        for (int k = 0; k < factors; ++k)
          v -= leader.beta[k] * returns[refs + 1 + k][i];
        lag.push_back(v);
      }
      lagged.push_back(lag);
      double last = returns[a].back() - leader.intercept;
      for (int k = 0; k < factors; ++k)
        last -= leader.beta[k] * returns[refs + 1 + k].back();
      current.push_back(last);
    }
    model = sa_rv_fit(target.residual, lagged);
    result.expected = model.intercept;
    for (int a = 0; a < refs; ++a)
      result.expected += model.beta[a] * current[a];
    if (model.sd < 1e-10)
      throw std::runtime_error("Lead-lag residual variance is too small.");
    result.score = result.expected / model.sd;
    result.innovation = result.expected;
    result.sd = model.sd;
    result.before = target.beta;
    result.reference_beta = model.beta;
    for (int k = 0; k < factors; ++k)
      result.weights[refs + 1 + k] = -target.beta[k];
  } else {
    std::vector<std::vector<double>> x = fx;
    if (mode == 1)
      for (int a = refs; a >= 1; --a)
        x.insert(x.begin(), std::vector<double>(returns[a].begin() + 1,
                                                returns[a].end() - 1));
    else {
      if (fixed.size() != size_t(refs))
        throw std::runtime_error("Reference weights required.");
      for (int a = 0; a < refs; ++a)
        for (int i = 0; i < window; ++i)
          y[i] -= fixed[a] * returns[a + 1][i + 1];
    }
    model = sa_rv_fit(y, x);
    for (int a = 0; a < refs; ++a) {
      double b = mode == 1 ? model.beta[a] : fixed[a];
      result.weights[a + 1] = -b;
      result.reference_beta.push_back(b);
    }
    for (int k = 0; k < factors; ++k)
      result.weights[refs + 1 + k] = -model.beta[(mode == 1 ? refs : 0) + k];
    // Before/after betas use the same multivariate factor model.
    std::vector<double> spread(returns[0].begin() + 1, returns[0].end() - 1);
    for (int a = 0; a < refs; ++a)
      for (int i = 0; i < window; ++i)
        spread[i] += result.weights[a + 1] * returns[a + 1][i + 1];
    result.before = sa_rv_fit(spread, fx).beta;
    double path = 0, mean = 0;
    for (double value : model.residual) {
      path += value;
      result.path.push_back(path);
      mean += path / window;
    }
    double variance = 0;
    for (double v : result.path)
      variance += std::pow(v - mean, 2) / window;
    result.sd = std::sqrt(variance);
    if (result.sd < 1e-10)
      throw std::runtime_error("Residual spread variance is too small.");
    result.innovation = -model.intercept;
    for (int a = 0; a < n; ++a)
      result.innovation += result.weights[a] * returns[a].back();
    result.score = (path + result.innovation - mean) / result.sd;
    for (double &v : result.path)
      v = (v - mean) / result.sd;
    result.path.push_back(result.score);
  }
  result.r2 = model.r2;
  std::vector<double> hedged(window, 0);
  for (int a = 0; a < n; ++a)
    for (int i = 0; i < window; ++i)
      hedged[i] += result.weights[a] * returns[a][i + 1];
  result.after = sa_rv_fit(hedged, fx).beta;
  double gross = 0;
  for (double w : result.weights)
    gross += std::abs(w);
  if (!std::isfinite(result.score + gross) || gross > 100)
    throw std::runtime_error("Unstable fitted hedge; no signal produced.");
  result.gross_scale = gross;
  for (double &w : result.weights)
    w /= gross;
  for (double &b : result.before)
    b /= gross;
  for (double &b : result.after)
    b /= gross;
  return result;
}

// Cost estimates are fractions of equity, using signal-close quantities.
// Actual next-open fills, rounding and borrowing can differ.
inline double sa_rv_cost(double turnover, int orders, double short_weight,
                         double days, double equity, double fee,
                         double slippage_bps, double borrow_percent) {
  if (equity <= 0 || turnover < 0 || orders < 0 || short_weight < 0 ||
      days < 0 || fee < 0 || slippage_bps < 0 || borrow_percent < 0)
    throw std::runtime_error("Invalid relative-value cost inputs.");
  return turnover * slippage_bps / 10000 + orders * fee / equity +
         short_weight * borrow_percent / 100 * days / 365;
}
inline double sa_rv_capture(const SA_RVResult &model, int mode, double exit_z,
                            double fraction) {
  // Scenario: capture a fraction of the distance to the exit band. This
  // is not a fitted expected return or a probability of mean reversion.
  double move = mode == 2
                    ? std::abs(model.expected)
                    : std::max(0., std::abs(model.score) - exit_z) * model.sd;
  return fraction * move / model.gross_scale;
}
