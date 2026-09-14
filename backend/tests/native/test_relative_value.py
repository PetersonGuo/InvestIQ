import copy
import math
import time
from conftest import wait_run
from test_http import query

MODEL = dict(category='competitor', target='AAPL', references=[{'ticker':'MSFT','weight':1}], market='SPY', sector='NVDA', factors=['AMZN'], fit_reference=True, lookback=120, entry_z=.1)

def test_residual_exposures_and_exported_cpp(client):
    response = client.post('/api/pairs/preview',json={'relative_value':MODEL})
    assert response.status_code == 200, response.text
    snapshot = response.json()
    assert snapshot['source'] == 'fixture'
    assert snapshot['baseline_end'] < snapshot['as_of']
    assert all(abs(row['after_beta']) < 1e-8 for row in snapshot['relative_value']['exposures'])
    assert math.isclose(sum(abs(leg['weight']) for leg in snapshot['relative_value']['hedge_legs']),1)
    draft = client.post('/api/pairs/relative-value/strategy',json={'relative_value':MODEL}).json()
    assert draft['language'] == 'cpp'
    assert draft['tickers'] == ['AAPL','MSFT','SPY','NVDA','AMZN']
    assert draft['warmup'] == 122
    # This test exercises execution/export compatibility; cost gating has dedicated tests.
    q={**query(client),**draft,'warmup':122,'params':{**draft['params'],'cost_aware':0}}
    response=client.post('/api/backtests',json=q)
    assert response.status_code==202,response.text
    result=wait_run(client,response.json()['id'])
    assert result['status']=='completed',result
    assert result['result']['data']['benchmark_ticker']=='SPY'
    assert 'strategy' in result['result']['risk_comparison']
    assert result['result']['fills'], result['result']['metrics']
    replay=client.post('/api/backtests',json=q).json()
    assert replay['cache_hit'] and replay['id']==result['id']


def test_rv_schedule_persistence_and_model_dedup(client):
    config={'source':'relationships','relationships':[{'relative_value':MODEL}],'max_alerts':2}
    response=client.post('/api/pairs/schedules',json=config)
    assert response.status_code==201,response.text
    uid=response.json()['id']
    for _ in range(100):
        rows=client.get('/api/pairs/alerts').json()
        if rows:break
        time.sleep(.1)
    assert len(rows)==1,client.get('/api/pairs/schedules').json()
    assert rows[0]['relative_value']['category']=='competitor'
    changed=copy.deepcopy(config);changed['relationships'][0]['relative_value']['entry_z']=3
    assert client.put('/api/pairs/schedules/'+uid,json=changed).status_code==200
    time.sleep(2.5)
    alerts=client.get('/api/pairs/alerts').json()
    assert len(alerts)==1
    assert alerts[0]['relative_value']['entry_z']==.1
    assert client.delete('/api/pairs/schedules/'+uid).status_code==204
    assert len(client.get('/api/pairs/alerts').json())==1


def test_all_supported_families_and_invalid_models(client):
    catalog=client.get('/api/pairs/relative-value/catalog').json()
    assert {item['id'] for item in catalog}=={'same_company','etf','competitor','basket','lead_lag','event','cross_asset'}
    for category in ('same_company','etf','competitor','basket','lead_lag','cross_asset'):
        model={**MODEL,'category':category,'fit_reference':category in ('competitor','basket','cross_asset')}
        response=client.post('/api/pairs/preview',json={'relative_value':model})
        assert response.status_code==200,(category,response.text)
    for bad in ({'market':'AAPL'},{'factors':['SPY']},{'interval':'1s'},{'instrument_scope':'foreign_adr'},{'references':[]},{'category':'event'}):
        response=client.post('/api/pairs/relative-value/strategy',json={'relative_value':{**MODEL,**bad}})
        assert response.status_code==422,response.text
    presets=client.get('/api/strategies/examples').json()
    assert len([p for p in presets if p['template_id'].startswith('rv_')])==6
    assert all(p['language']=='cpp' for p in presets if p['template_id'].startswith('rv_'))


def test_event_requires_asof_terms_and_reports_payoff_gap(client):
    data=client.get('/api/stocks/AAPL?days=250').json()['bars']
    terms={**MODEL,'category':'event','cash_per_share':10,'shares_per_share':.5,'terms_as_of':data[0]['time'],'expires_on':'2099-01-01','terms_source':'Test-only supplied contract terms'}
    response=client.post('/api/pairs/preview',json={'relative_value':terms})
    assert response.status_code==200,response.text
    assert response.json()['relative_value']['event_gap_percent'] is not None
    assert client.post('/api/pairs/preview',json={'relative_value':{**terms,'terms_as_of':'2098-01-01'}}).status_code==422


def test_longest_window_export_has_valid_warmup(client):
    draft=client.post('/api/pairs/relative-value/strategy',json={'relative_value':{**MODEL,'lookback':250}}).json()
    assert draft['warmup']==252
    response=client.post('/api/backtests',json={**query(client),**draft})
    assert response.status_code==202,response.text
    result=wait_run(client,response.json()['id'])
    assert result['status']=='completed',result
