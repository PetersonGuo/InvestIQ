"""Starter strategies for comparison, not optimized trading recommendations."""

from pathlib import Path

PRESETS = [
    {
        "template_id": "moving_average",
        "name": "Moving average crossover",
        "description": "Trend following: invest when the fast closing-price average is above the slow average; exit otherwise. Can whipsaw in sideways markets.",
        "params": {"fast": 10, "slow": 30},
        "warmup": 30,
    },
    {
        "template_id": "buy_and_hold",
        "name": "Buy and hold",
        "description": "Baseline: invest at the first eligible next open, then hold. Useful for checking whether an active strategy adds value after costs.",
        "params": {},
        "warmup": 0,
    },
    {
        "template_id": "channel_breakout",
        "name": "Price-channel breakout",
        "description": "Momentum: buy when the close breaks above the prior entry-window high; exit below the prior exit-window low. The current bar is excluded from both channels.",
        "params": {"entry_period": 20, "exit_period": 10},
        "warmup": 20,
    },
    {
        "template_id": "mean_reversion",
        "name": "Z-score mean reversion",
        "description": "Buy after a close falls at least entry_z standard deviations below its rolling mean; exit when its z-score reaches exit_z. Can struggle during persistent declines.",
        "params": {"lookback": 20, "entry_z": 2, "exit_z": 0},
        "warmup": 20,
    },
]


def examples():
    folder = Path(__file__).with_name("examples")
    return [
        {
            **preset,
            "language": language,
            "code": (folder / f"{preset['template_id']}.{extension}").read_text(),
        }
        for preset in PRESETS
        for language, extension in [("python", "py"), ("cpp", "cpp")]
    ]
