class Strategy:
    def on_bar(self, ctx):
        fast = int(ctx.params.get("fast", 10))
        slow = int(ctx.params.get("slow", 30))
        if not 1 <= fast < slow:
            raise ValueError("Require 1 <= fast < slow")
        if len(ctx.history) < slow:
            return None
        fast_mean = sum(b.close for b in ctx.history[-fast:]) / fast
        slow_mean = sum(b.close for b in ctx.history[-slow:]) / slow
        return 1.0 if fast_mean > slow_mean else 0.0
