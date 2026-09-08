#pragma once
#include "common.hpp"
namespace sa {
J ib_request(const std::string &operation, const J &value);
J ib_status();
void ib_close();
J history(const std::string &symbol, int days = 90, const std::string &before = "",
          const std::string &interval = "1d");
J quote(const std::string &symbol);
J fundamentals(const std::string &);
J company_news(const std::string &);
J parse_company_facts(const J &);
J massive_data(const std::string &);
J search(const std::string &query, int limit = 20);
J scan(J);
J ticks(const std::string &symbol, const std::string &before = "");
} // namespace sa
