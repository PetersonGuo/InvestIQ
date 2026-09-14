import sqlite3
import time

BASE = '/api/pairs/schedules'
CONFIG = dict(source='symbols', tickers=['AAPL', 'MSFT', 'GOOGL', 'AMZN'], min_correlation=0, require_cointegration=False, relationship='either', max_alerts=2)

def completed(client, previous=None):
    for _ in range(200):
        row = client.get(BASE).json()[0]
        if row['last_finished'] and row['last_finished'] != previous and not row['running']:
            return row
        time.sleep(.1)
    raise AssertionError(row)

def test_schedule_automatic_dedup_restart_and_deleted_pair(launch):
    client, process, db = launch()
    response = client.post(BASE, json=CONFIG)
    assert response.status_code == 201, response.text
    uid = response.json()['id']
    first = completed(client)
    assert not first['last_error'], first
    assert len(first['last_result']['created']) == 2
    alerts = client.get('/api/pairs/alerts').json()
    assert len(alerts) == 2 and all(a['repeat'] for a in alerts)
    # Paused manual settings remain intact; deleted auto pairs are suppressed.
    client.post(f"/api/pairs/alerts/{alerts[0]['id']}/state", json={'active': False})
    client.delete(f"/api/pairs/alerts/{alerts[1]['id']}")
    assert client.post(f'{BASE}/{uid}/run', json={}).status_code == 202
    second = completed(client, first['last_finished'])
    assert len(second['last_result']['created']) == 1
    current = client.get('/api/pairs/alerts').json()
    assert next(a for a in current if a['id'] == alerts[0]['id'])['active'] == False
    deleted = {alerts[1]['ticker_a'], alerts[1]['ticker_b']}
    assert all({a['ticker_a'], a['ticker_b']} != deleted for a in current)
    assert client.post(f'{BASE}/{uid}/state', json={'active': False}).status_code == 200
    assert client.post(f'{BASE}/{uid}/run', json={}).status_code == 409
    process.terminate(); process.wait(timeout=20)
    # Interrupted active run is recovered and picked up automatically at restart.
    with sqlite3.connect(db) as connection:
        connection.execute('UPDATE pair_schedules SET running=1,active=1,next_run=0')
    client, _, _ = launch(database=db)
    third = completed(client, second['last_finished'])
    assert third['last_result']['created'] == []
    assert client.delete(f'{BASE}/{uid}').status_code == 204
    assert client.get(BASE).json() == []
    assert len(client.get('/api/pairs/alerts').json()) == 2

def test_scanner_schedule_and_validation(client):
    for bad in [dict(interval_hours=0), dict(source='bad'), dict(source='symbols',tickers=['AAPL']), dict(max_alerts=51), dict(min_price=10,max_price=5)]:
        assert client.post(BASE,json=bad).status_code == 422
    response = client.post(BASE,json={**CONFIG, 'source':'scanner'})
    assert response.status_code == 201
    row = completed(client)
    assert not row['last_error'], row
    assert len(row['last_result']['tickers']) >= 2
    assert row['last_result']['created']

def test_disconnected_schedule_is_saved_and_retries(launch):
    client, _, _ = launch(mode='ibkr')
    response = client.post(BASE,json={})
    assert response.status_code == 201
    row = completed(client)
    assert row['last_error']
    assert row['next_run'] > time.time()+250
    assert client.get('/api/pairs/alerts').json() == []

def test_paused_schedule_edit_and_existing_manual_pair(client):
    manual = client.post('/api/pairs/alerts', json={'ticker_a':'MSFT','ticker_b':'AAPL','lookback':60,'threshold':2,'hedge_ratio':1}).json()
    client.post(f"/api/pairs/alerts/{manual['id']}/state",json={'active':False})
    created = client.post(BASE,json={**CONFIG,'tickers':['AAPL','MSFT']}).json()
    uid = created['id']
    assert client.post(f'{BASE}/{uid}/state',json={'active':False}).status_code == 200
    time.sleep(2.2)
    assert client.get(BASE).json()[0]['last_started'] is None
    assert client.put(f'{BASE}/{uid}',json={**CONFIG,'tickers':['MSFT','AAPL'],'interval_hours':6}).json()['active'] == False
    client.post(f'{BASE}/{uid}/state',json={'active':True})
    row = completed(client)
    assert row['last_result']['candidates'] == 1
    assert row['last_result']['created'] == []
    alerts = client.get('/api/pairs/alerts').json()
    assert len(alerts) == 1 and alerts[0]['active'] == False
    assert alerts[0]['hedge_ratio'] == 1


def test_no_qualifying_pairs_is_success(client):
    client.post(BASE,json={**CONFIG,'min_correlation':1})
    row = completed(client)
    assert row['last_error'] is None
    assert row['last_result']['candidates'] == 0
    assert row['next_run'] > time.time()+23*3600
