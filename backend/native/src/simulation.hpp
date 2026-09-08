#pragma once
#include "common.hpp"
#include "strategy_api.h"
namespace sa {
using Callback = double (*)(const SA_Bar *, int, double, int64_t, const char *);
J simulate(const J &bars, Callback, const J &params, const J &settings);
} // namespace sa
