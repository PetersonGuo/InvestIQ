class Strategy:
    def on_bar(self, ctx):
        if any(ctx.portfolio.positions.values()):
            return None
        return {symbol: 1 / len(ctx.histories) for symbol in ctx.histories}
