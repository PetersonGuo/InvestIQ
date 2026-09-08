class Strategy:
    def on_bar(self, ctx):
        # Re-issue during warmup until the first permitted fill, then hold.
        return 1.0 if ctx.portfolio.shares == 0 else None
