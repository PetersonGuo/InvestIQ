#include "valuation.hpp"
#include "relative_value.hpp"
#include "stock_lists.hpp"
#include <array>
#include <numeric>
#include <set>
namespace sa {
namespace {
const std::vector<std::string> names = {"Revenue growth", "Net profit margin",
                                        "Operating cash flow / revenue", "Liabilities / assets"};
J pick(const J &company, const std::vector<std::string> &tags, const std::string &unit,
       const std::string &asof, bool annual, const std::string &end = "") {
  J best = nullptr;
  auto gaap = company.value("facts", J::object()).value("us-gaap", J::object());
  for (auto &tag : tags) {
    auto rows = gaap.value(tag, J::object()).value("units", J::object()).value(unit, J::array());
    for (auto row : rows) {
      auto form = row.value("form", "");
      if (form != "10-K" && form != "10-K/A")
        continue;
      if (!row.contains("val") || !row["val"].is_number() ||
          !std::isfinite(row["val"].get<double>()))
        continue;
      auto filed = row.value("filed", ""), e = row.value("end", "");
      if (filed.empty() || filed >= asof || e.empty() || e >= asof || (!end.empty() && e != end))
        continue;
      if (annual) {
        if (!row.contains("start"))
          continue;
        auto days = (parse_time(e) - parse_time(row["start"])) / 86400;
        if (days < 330 || days > 400)
          continue;
      } else if (row.contains("start"))
        continue;
      row["tag"] = tag;
      row["unit"] = unit;
      if (best.is_null() || e > best["end"].get<std::string>() ||
          (e == best["end"].get<std::string>() && filed > best["filed"].get<std::string>()))
        best = row;
    }
  }
  if (best.is_null())
    throw Error(422, "Missing comparable annual " + tags[0] + " available before " + asof + ".");
  return best;
}
struct Fit {
  double base;
  std::array<double, 4> mean{}, scale{}, beta{};
};
Fit ridge(const std::vector<J> &rows) {
  Fit f{};
  const double n = rows.size();
  for (auto &r : rows) {
    f.base += std::log(r["multiple"].get<double>()) / n;
    for (int j = 0; j < 4; ++j)
      f.mean[j] += r["features"][j].get<double>() / n;
  }
  for (auto &r : rows)
    for (int j = 0; j < 4; ++j)
      f.scale[j] += std::pow(r["features"][j].get<double>() - f.mean[j], 2) / n;
  for (auto &v : f.scale)
    v = std::max(1e-8, std::sqrt(v));
  double a[4][5]{};
  for (int j = 0; j < 4; ++j)
    a[j][j] = 1; // fixed ridge penalty, unpenalized intercept
  for (auto &r : rows) {
    std::array<double, 4> x{};
    for (int j = 0; j < 4; ++j)
      x[j] = (r["features"][j].get<double>() - f.mean[j]) / f.scale[j];
    for (int j = 0; j < 4; ++j) {
      a[j][4] += x[j] * (std::log(r["multiple"].get<double>()) - f.base);
      for (int k = 0; k < 4; ++k)
        a[j][k] += x[j] * x[k];
    }
  }
  for (int j = 0; j < 4; ++j) {
    double v = a[j][j];
    for (int k = j; k < 5; ++k)
      a[j][k] /= v;
    for (int i = 0; i < 4; ++i)
      if (i != j) {
        double b = a[i][j];
        for (int k = j; k < 5; ++k)
          a[i][k] -= b * a[j][k];
      }
  }
  for (int j = 0; j < 4; ++j)
    f.beta[j] = a[j][4];
  return f;
}
double predict(const Fit &f, const J &r) {
  double v = f.base;
  for (int j = 0; j < 4; ++j)
    v += f.beta[j] * (r["features"][j].get<double>() - f.mean[j]) / f.scale[j];
  return v;
}
} // namespace
J valuation_facts(const J &company, const std::string &asof) {
  const std::vector<std::string> revenue = {"RevenueFromContractWithCustomerExcludingAssessedTax",
                                            "Revenues", "SalesRevenueNet"};
  auto rev = pick(company, revenue, "USD", asof, true);
  auto end = rev["end"].get<std::string>();
  if (parse_time(asof) - parse_time(end) > 550 * 86400)
    throw Error(422, "Annual financials are more than 550 days old.");
  // Previous annual report, not an overlapping year-to-date period.
  auto prior_company = company;
  for (auto &tag : revenue) {
    auto &units = prior_company["facts"]["us-gaap"][tag]["units"]["USD"];
    if (units.is_array()) {
      J kept = J::array();
      for (auto &fact : units)
        if (fact.value("end", "") < rev["start"].get<std::string>())
          kept.push_back(fact);
      units = kept;
    }
  }
  auto prev = pick(prior_company, revenue, "USD", asof, true);
  auto spacing = (parse_time(end) - parse_time(prev["end"])) / 86400;
  if (spacing < 330 || spacing > 400)
    throw Error(422, "Missing consecutive annual revenue periods.");
  auto ni = pick(company, {"NetIncomeLoss", "ProfitLoss"}, "USD", asof, true, end);
  auto eps = pick(company, {"EarningsPerShareDiluted"}, "USD/shares", asof, true, end);
  auto ocf = pick(company, {"NetCashProvidedByUsedInOperatingActivities"}, "USD", asof, true, end);
  auto assets = pick(company, {"Assets"}, "USD", asof, false, end);
  J liabilities, liability_components = J::array();
  try {
    liabilities = pick(company, {"Liabilities"}, "USD", asof, false, end);
  } catch (const Error &) {
    auto current = pick(company, {"LiabilitiesCurrent"}, "USD", asof, false, end);
    auto noncurrent = pick(company, {"LiabilitiesNoncurrent"}, "USD", asof, false, end);
    liabilities = current;
    liabilities["val"] = current["val"].get<double>() + noncurrent["val"].get<double>();
    liabilities["tag"] = "Liabilities (current + noncurrent)";
    liabilities["filed"] =
        std::max(current["filed"].get<std::string>(), noncurrent["filed"].get<std::string>());
    liability_components = J::array({current, noncurrent});
  }
  for (auto &r : {ni, eps, ocf})
    if (r["start"] != rev["start"])
      throw Error(422, "Annual reporting periods do not align.");
  double sales = rev["val"], prior = prev["val"], income = ni["val"], earnings = eps["val"],
         asset = assets["val"], liab = liabilities["val"];
  if (sales <= 0 || prior <= 0 || income <= 0 || earnings <= 0 || asset <= 0 || liab < 0)
    throw Error(
        422,
        "Model requires positive revenue, earnings, EPS and assets, and nonnegative liabilities.");
  J features =
      J::array({sales / prior - 1, income / sales, ocf["val"].get<double>() / sales, liab / asset});
  if (std::abs(features[0].get<double>()) > 2 || features[1].get<double>() > 1 ||
      std::abs(features[2].get<double>()) > 2 || features[3].get<double>() > 2)
    throw Error(422, "Extreme financial ratios require manual review; excluded from peer fitting.");
  J evidence = J::array();
  J source_facts = J::array({rev, prev, ni, eps, ocf, assets});
  if (liability_components.empty())
    source_facts.push_back(liabilities);
  else
    for (auto &f : liability_components)
      source_facts.push_back(f);
  for (auto r : source_facts) {
    auto acc = r.value("accn", "");
    auto compact = acc;
    compact.erase(std::remove(compact.begin(), compact.end(), '-'), compact.end());
    if (!compact.empty() && compact.find_first_not_of("0123456789") == std::string::npos)
      r["filing_url"] = "https://www.sec.gov/Archives/edgar/data/" +
                        std::to_string(company.at("cik").get<int64_t>()) + "/" + compact + "/" +
                        acc + "-index.html";
    evidence.push_back(r);
  }
  return {{"features", features},
          {"eps", earnings},
          {"period_end", end},
          {"evidence", evidence},
          {"name", company.value("entityName", "")},
          {"cik", company.at("cik")}};
}
J fit_peer_valuation(const J &input) {
  J output = J::array();
  for (auto &target : input) {
    std::vector<J> peers;
    for (auto &r : input)
      if (r["cik"] != target["cik"] && r["group"] == target["group"])
        peers.push_back(r);
    J row = target;
    row["action"] = {
        {"state", "wait"}, {"label", "Wait — no confirmed valuation trade"}, {"legs", J::array()}};
    row["peers"] = J::array();
    for (auto &r : peers)
      row["peers"].push_back(r["ticker"]);
    if (peers.size() < 10) {
      row["status"] = "insufficient_peers";
      row["reason"] = "Needs at least 10 other eligible issuers in the same SEC industry group.";
      output.push_back(row);
      continue;
    }
    auto fit = ridge(peers);
    auto expected = predict(fit, target);
    std::vector<double> errors;
    for (size_t i = 0; i < peers.size(); ++i) {
      auto train = peers;
      train.erase(train.begin() + i);
      errors.push_back(
          std::abs(std::log(peers[i]["multiple"].get<double>()) - predict(ridge(train), peers[i])));
    }
    std::sort(errors.begin(), errors.end());
    double radius = std::max(
        .15, errors[std::min(errors.size() - 1, size_t(std::ceil(.9 * (errors.size() + 1)) - 1))]);
    bool outside = false;
    J contributions = J::array();
    for (int j = 0; j < 4; ++j) {
      double z = (target["features"][j].get<double>() - fit.mean[j]) / fit.scale[j];
      outside |= std::abs(z) > 3;
      contributions.push_back({{"feature", names[j]},
                               {"value", target["features"][j]},
                               {"peer_mean", fit.mean[j]},
                               {"log_multiple_contribution", fit.beta[j] * z}});
    }
    row["status"] = outside ? "extrapolation" : "estimated";
    row["reason"] = outside ? "A financial feature is more than three peer standard deviations "
                              "away; no trade instruction."
                            : "Estimate excludes this issuer; uncertainty uses held-out peer "
                              "errors, not a guaranteed fair-value interval.";
    // Bound extreme outputs before exponentiation; do not turn unstable estimates into signals.
    if (!std::isfinite(expected) || expected - radius < -5 || expected + radius > 10) {
      row["status"] = "unstable";
      row["reason"] = "Unstable valuation estimate; no trade instruction.";
      output.push_back(row);
      continue;
    }
    row["expected_multiple"] = std::exp(expected);
    row["lower_multiple"] = std::exp(expected - radius);
    row["upper_multiple"] = std::exp(expected + radius);
    row["gap"] = target["multiple"].get<double>() / std::exp(expected) - 1;
    row["baseline_multiple"] = std::exp(fit.base);
    row["contributions"] = contributions;
    row["valuation_direction"] =
        target["multiple"].get<double>() < std::exp(expected - radius)   ? "cheap"
        : target["multiple"].get<double>() > std::exp(expected + radius) ? "expensive"
                                                                         : "within_range";
    output.push_back(row);
  }
  return output;
}
J peer_valuation(const J &request) {
  auto asof = text(request, "as_of", ny_date());
  if (asof != ny_date())
    throw Error(422,
                "Historical valuation runs are not enabled: archived split-consistent prices/EPS "
                "and historical peer membership are required. Current screening only.");
  J tickers = request.value("tickers", J::array());
  if (request.contains("stock_list_id"))
    tickers = stock_list(text(request, "stock_list_id"))["tickers"];
  if (!tickers.is_array() || tickers.size() < 2 || tickers.size() > 20)
    throw Error(422, "Supply 2–20 stock symbols or a stock_list_id.");
  double min_gap = number(request, "valuation_min_gap", .2, .05, 1);
  J inputs = J::array(), errors = J::array();
  std::set<std::string> symbols;
  std::set<int64_t> issuers;
  for (auto &t : tickers) {
    if (!t.is_string())
      throw Error(422, "Tickers must be strings.");
    symbols.insert(symbol(t.get<std::string>()));
  }
  auto traded = symbols;
  auto additional = request.value("valuation_peers", J::array());
  if (!additional.is_array() || additional.size() > 40)
    throw Error(422, "Supply up to 40 additional peer symbols.");
  for (auto &t : additional) {
    if (!t.is_string())
      throw Error(422, "Peer tickers must be strings.");
    symbols.insert(symbol(t.get<std::string>()));
  }
  for (auto &ticker : symbols)
    try {
      auto company = valuation_company(ticker);
      auto sic = company.value("sic", "");
      if (sic.size() != 4 || sic[0] == '6')
        throw Error(422, "This earnings model excludes financials, insurers, REITs and missing "
                         "industry classifications.");
      if (company.value("issuer_tickers", J::array()).size() != 1)
        throw Error(422, "Multiple listed securities or share classes need class-specific EPS "
                         "mapping; excluded.");
      auto row = valuation_facts(company, asof);
      int64_t cik = row["cik"];
      if (!issuers.insert(cik).second)
        throw Error(422, "Duplicate issuer excluded from peer observations.");
      auto prices = history(ticker, 10);
      if (prices["bars"].empty())
        throw Error(422, "No price history.");
      J bar = nullptr;
      for (auto &b : prices["bars"])
        if (b["time"].get<std::string>().substr(0, 10) < asof)
          bar = b;
      if (bar.is_null() || parse_time(asof) - parse_time(bar["time"]) > 5 * 86400)
        throw Error(422, "No recent completed daily price.");
      double close = bar["close"], pe = close / row["eps"].get<double>();
      if (pe < 2 || pe > 150)
        throw Error(422, "P/E outside 2–150; requires manual review.");
      row.update({{"ticker", ticker},
                  {"group", sic.substr(0, 3)},
                  {"industry", company.value("sic_description", "")},
                  {"multiple", pe},
                  {"price", close},
                  {"price_as_of", bar["time"]},
                  {"source", company.value("source", "sec")},
                  {"price_source", prices["source"]}});
      inputs.push_back(row);
    } catch (const std::exception &e) {
      errors.push_back({{"ticker", ticker}, {"error", e.what()}});
    }
  auto all_rows = fit_peer_valuation(inputs);
  J rows = J::array();
  for (auto &row : all_rows)
    if (traded.count(row["ticker"].get<std::string>()))
      rows.push_back(row);
  for (auto &row : rows) {
    if (row["status"] != "estimated" || row["valuation_direction"] == "within_range" ||
        std::abs(row["gap"].get<double>()) < min_gap)
      continue;
    try {
      auto sector = text(request, "sector", "auto");
      if (sector == "auto")
        throw Error(422, "Choose a sector ETF in scan settings to evaluate price confirmation and "
                         "exposure hedges.");
      J refs = J::array();
      for (auto &p : row["peers"]) {
        if (!traded.count(p.get<std::string>()))
          continue;
        if (refs.size() >= 5)
          break;
        refs.push_back({{"ticker", p}, {"weight", 1}});
      }
      if (refs.size() < 2)
        throw Error(422, "Valuation gap found, but at least two eligible reference stocks in the "
                         "tradable list are needed for a hedged basket.");
      for (auto &ref : refs)
        ref["weight"] = 1.0 / refs.size();
      J model = {{"category", "basket"},
                 {"target", row["ticker"]},
                 {"references", refs},
                 {"fit_reference", false},
                 {"market", request.value("market", "SPY")},
                 {"sector", sector},
                 {"factors", request.value("factors", J::array({"MTUM", "VLUE"}))},
                 {"lookback", request.value("lookback", 120)},
                 {"entry_z", request.value("threshold", 2.0)}};
      auto rule = validate_pair({{"relative_value", model}});
      auto snapshot = analyze_pair(rule);
      row["price_confirmation"] = snapshot;
      row["price_confirmation"].erase("points");
      double z = snapshot["value"], entry = rule["threshold"];
      bool direction = (row["valuation_direction"] == "cheap" && z < 0) ||
                       (row["valuation_direction"] == "expensive" && z > 0);
      if (direction && snapshot["matched"] == true && std::abs(z) < std::max(4., 2 * entry) &&
          parse_time(asof) - parse_time(snapshot["as_of"]) <= 5 * 86400)
        row["action"] = {{"state", "entry"},
                         {"label", "Valuation + price entry candidate — review all legs"},
                         {"legs", snapshot["relative_value"]["hedge_legs"]}};
      else
        row["action"]["label"] = "Valuation gap found — wait for matching price confirmation";
    } catch (const std::exception &e) {
      row["confirmation_error"] = e.what();
    }
  }
  return {
      {"model", "peer_earnings_v1"},
      {"as_of", asof},
      {"computed_at", stamp()},
      {"rows", rows},
      {"errors", errors},
      {"minimum_other_peers", 10},
      {"min_gap", min_gap},
      {"notes",
       J::array(
           {"Current screening only; no fundamental-strategy backtest is represented by these "
            "results.",
            "P/E uses latest annual diluted EPS. Annual facts must be filed before the analysis "
            "date. Later restatements are excluded by that cutoff.",
            "Peer groups use the first three SEC SIC digits and need business-model review. "
            "The list plus optional reference peers define the estimation universe; selection bias "
            "remains.",
            "Estimated ranges use leave-one-peer-out absolute log errors with a 0.15 floor; these "
            "are empirical uncertainty bands, not calibrated confidence guarantees.",
            "Market, sector and supplied factor hedges belong to the confirming price model. "
            "Neutrality is estimated in sample; no future neutrality or convergence is guaranteed.",
            "Review stock splits since the annual report, one-off earnings, current borrow and "
            "transaction costs before trading. No orders are submitted."})}};
}
} // namespace sa
