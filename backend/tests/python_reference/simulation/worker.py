"""Trusted strategy worker process. Not a security sandbox."""

import importlib.util
import json
from pathlib import Path
import resource
import sys
import traceback

from simulation.native import simulate_native


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (8, 8))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024, 64 * 1024 * 1024))
    request_path, output_path = map(Path, sys.argv[1:3])
    payload = json.loads(request_path.read_text())
    try:
        if payload["language"] == "python":
            spec = importlib.util.spec_from_file_location(
                "user_strategy", payload["module_path"]
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            strategy = module.Strategy()
            callback = strategy.on_bar
        else:
            callback = None
        result = simulate_native(
            payload["bars"],
            payload["params"],
            payload["settings"],
            payload["engine_path"],
            callback=callback,
            strategy_path=(
                payload["module_path"] if payload["language"] == "cpp" else None
            ),
        )
        response = {"result": result}
    except Exception as exc:
        response = {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
        }
    output_path.write_text(json.dumps(response, allow_nan=False))


if __name__ == "__main__":
    main()
