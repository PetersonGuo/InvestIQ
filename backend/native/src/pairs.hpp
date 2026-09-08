#pragma once
#include "market.hpp"
namespace sa {
bool matches(double, const J &);
J calculate_pair(const J &, const J &, const J &, const std::string &today = "");
J analyze_pair(const J &);
J pair_list();
J pair_get(const std::string &);
J pair_save(const J &, const std::string &id = "");
J pair_state(const std::string &, bool);
void pair_delete(const std::string &);
J pair_events();
void check_pair_alerts();
J portfolio();
J place_order(const J &);
void check_price_alerts();
} // namespace sa
