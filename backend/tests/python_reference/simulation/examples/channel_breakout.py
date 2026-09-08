class Strategy:
    def on_bar(self, ctx):
        entry = ctx.params.get("entry_period", 20)
        exit_period = ctx.params.get("exit_period", 10)
        if not all(1 <= p <= 250 and int(p) == p for p in (entry, exit_period)):
            raise ValueError("Channel periods must be whole numbers from 1 to 250")
        entry, exit_period = int(entry), int(exit_period)
        if len(ctx.history) <= max(entry, exit_period):
            return None
        prior_high = max(b.high for b in ctx.history[-entry - 1 : -1])
        prior_low = min(b.low for b in ctx.history[-exit_period - 1 : -1])
        if ctx.bar.close > prior_high:
            return 1.0
        if ctx.bar.close < prior_low:
            return 0.0
        return None
