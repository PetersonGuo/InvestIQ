import math


class Strategy:
    def on_bar(self, ctx):
        symbols = list(ctx.histories)
        if len(symbols) != 2:
            raise ValueError("This starter requires exactly two stocks")
        a, b = (ctx.histories[s] for s in symbols)
        n = int(ctx.params.get("lookback", 60))
        if n < 20 or len(a) < n + 1:
            return None
        x, y = (
            [math.log(row.close) for row in series[-n - 1 : -1]] for series in (a, b)
        )
        mx, my = sum(x) / n, sum(y) / n
        variance = sum((v - my) ** 2 for v in y)
        if variance < 1e-18:
            return None
        beta = sum((u - mx) * (v - my) for u, v in zip(x, y)) / variance
        intercept = mx - beta * my
        e = [u - beta * v - intercept for u, v in zip(x, y)]
        sd = math.sqrt(sum(v * v for v in e) / n)
        if sd <= 1e-10 or abs(beta) > 10:
            return None
        xx = xz = zz = xy = zy = 0.0
        for i in range(2, n):
            x1, z1, y1 = e[i - 1], e[i - 1] - e[i - 2], e[i] - e[i - 1]
            xx += x1 * x1
            xz += x1 * z1
            zz += z1 * z1
            xy += x1 * y1
            zy += z1 * y1
        det = xx * zz - xz * xz
        if det <= 1e-14 * xx * zz:
            return None
        rho, lag = (xy * zz - zy * xz) / det, (zy * xx - xy * xz) / det
        sse = sum(
            (e[i] - e[i - 1] - rho * e[i - 1] - lag * (e[i - 1] - e[i - 2])) ** 2
            for i in range(2, n)
        )
        se = math.sqrt(sse / (n - 4) * zz / det)
        if se <= 1e-14:
            return None
        cointegrated = rho / se < -3.33613 - 6.1101 / (n - 1) - 6.823 / (n - 1) ** 2
        z = (math.log(a[-1].close) - beta * math.log(b[-1].close) - intercept) / sd
        held = any(ctx.portfolio.positions.values())
        if held and (
            abs(z) <= ctx.params.get("exit_z", 0.5)
            or abs(z) >= ctx.params.get("stop_z", 4)
            or not cointegrated
        ):
            return dict.fromkeys(symbols, 0)
        if (
            held
            or not cointegrated
            or abs(z) < ctx.params.get("entry_z", 2)
            or abs(z) >= ctx.params.get("stop_z", 4)
        ):
            return None
        direction = -1 if z > 0 else 1
        return {
            symbols[0]: direction / (1 + abs(beta)),
            symbols[1]: -direction * beta / (1 + abs(beta)),
        }
