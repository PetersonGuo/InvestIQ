#pragma once
#include "market.hpp"
namespace sa {
J valuation_company(const std::string &);
J valuation_facts(const J &, const std::string &);
J peer_valuation(const J &);
J fit_peer_valuation(const J &);
} // namespace sa
