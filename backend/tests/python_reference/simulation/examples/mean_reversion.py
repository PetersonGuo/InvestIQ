from math import sqrt


class Strategy:
    def on_bar(self, ctx):
        lookback = ctx.params.get("lookback", 20)
        entry_z = ctx.params.get("entry_z", 2)
        exit_z = ctx.params.get("exit_z", 0)
        if not 2 <= lookback <= 250 or int(lookback) != lookback:
            raise ValueError("lookback must be a whole number from 2 to 250")
        if entry_z <= 0 or exit_z <= -entry_z:
            raise ValueError("Require entry_z > 0 and exit_z > -entry_z")
        lookback = int(lookback)
        if len(ctx.history) < lookback:
            return None
        closes = [b.close for b in ctx.history[-lookback:]]
        average = sum(closes) / lookback
        deviation = sqrt(sum((price - average) ** 2 for price in closes) / lookback)
        if deviation == 0:
            return None
        z_score = (ctx.bar.close - average) / deviation
        if z_score <= -entry_z:
            return 1.0
        if z_score >= exit_z:
            return 0.0
        return None
