#pragma once
#include "market.hpp"
namespace sa {
J examples();
J backtest_submit(const J &);
J backtest_get(const std::string &);
J backtest_list();
void backtests_stop();
} // namespace sa
