#pragma once
#include "pairs.hpp"
namespace sa {
J stock_lists();
J stock_list(const std::string &);
J save_stock_list(const J &, const std::string &existing = "");
void delete_stock_list(const std::string &);
J validate_list_search(const J &);
J discover_stock_list(const J &);
J pair_opportunities();
} // namespace sa
