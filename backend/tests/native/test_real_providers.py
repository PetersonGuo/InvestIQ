"""Production binaries must never expose the test market-data provider."""
import os
import subprocess
from conftest import ROOT
import pytest


@pytest.mark.parametrize('mode', ['demo', 'fixture'])
def test_production_rejects_non_provider_modes(mode, tmp_path):
    db = tmp_path / 'rejected.sqlite3'
    result = subprocess.run(
        [str(ROOT / 'build/native/stockassist-server')],
        env={**os.environ, 'STOCKASSIST_DATA_MODE': mode, 'STOCKASSIST_DB': str(db)},
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode != 0
    assert 'real market data is required' in result.stderr
    assert not db.exists()


def test_production_ignores_fixture_environment_and_reports_offline(launch):
    client, _, _ = launch('ibkr', production=True, STOCKASSIST_TEST_DATA='1', IBKR_PORT='1')
    assert client.get('/health').json()['data_mode'] == 'ibkr'
    for path in ['/api/stocks/AAPL', '/api/stocks/AAPL/quote']:
        response = client.get(path)
        assert response.status_code in (502, 503, 504)
        assert 'IBKR' in response.json()['detail']
    assert client.get('/api/alerts').json() == []
