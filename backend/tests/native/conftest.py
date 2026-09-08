"""Black-box tests of the C++ server and its existing HTTP contract."""

import os
from pathlib import Path
import socket
import subprocess
import time
import pytest
import requests

ROOT = Path(__file__).resolve().parents[2]


class Client:
    def __init__(self, port):
        self.base = f"http://127.0.0.1:{port}"
        self.session = requests.Session()

    def get(self, path, **kwargs):
        return self.session.get(self.base + path, timeout=20, **kwargs)

    def post(self, path, **kwargs):
        return self.session.post(self.base + path, timeout=20, **kwargs)

    def put(self, path, **kwargs):
        return self.session.put(self.base + path, timeout=20, **kwargs)

    def delete(self, path, **kwargs):
        return self.session.delete(self.base + path, timeout=20, **kwargs)


@pytest.fixture
def launch(tmp_path):
    processes = []

    def start(mode="demo", database=None, **extra):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        db = database or tmp_path / f"{port}.sqlite3"
        log = open(tmp_path / f"{port}.log", "w")
        environment = {
            **os.environ,
            "STOCKASSIST_DATA_MODE": mode,
            "STOCKASSIST_DB": str(db),
            **extra,
        }
        process = subprocess.Popen(
            [str(ROOT / "build/native/stockassist-server"), "--port", str(port)],
            env=environment,
            stdout=log,
            stderr=log,
        )
        processes.append((process, log))
        client = Client(port)
        for _ in range(100):
            if process.poll() is not None:
                pytest.fail(
                    f'Native server exited: {(tmp_path/f"{port}.log").read_text()}'
                )
            try:
                if client.get("/health").status_code == 200:
                    return client, process, db
            except requests.ConnectionError:
                pass
            time.sleep(0.05)
        pytest.fail("Native server did not start")

    yield start
    for process, log in processes:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()


@pytest.fixture
def client(launch):
    return launch()[0]


def wait_run(client, run_id):
    for _ in range(300):
        run = client.get("/api/backtests/" + run_id).json()
        if run["status"] not in ("queued", "running"):
            return run
        time.sleep(0.05)
    pytest.fail("Backtest did not finish")
