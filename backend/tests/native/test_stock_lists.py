import json
import sqlite3
import time

LISTS='/api/pairs/stock-lists'
SCHEDULES='/api/pairs/schedules'
SYMBOLS=['AAPL','MSFT','NVDA','AMZN','GOOGL','META','TSLA']

def wait_scan(client,uid,previous=None):
    for _ in range(150):
        row=next(r for r in client.get(SCHEDULES).json() if r['id']==uid)
        if not row['running'] and row['last_finished'] and row['last_finished']!=previous:return row
        time.sleep(.1)
    raise AssertionError(row)

def create_plan(client,uid,**extra):
    r=client.post(SCHEDULES,json={'source':'stock_list','stock_list_id':uid,'algorithms':['residual_pair','basket'],'lookback':120,'min_correlation':0,'threshold':.1,'max_alerts':5,'model_limit':8,**extra})
    assert r.status_code==201,r.text
    return r.json()['id']

def test_stock_lists_generate_models_dedup_restart_and_remove(launch):
    client,process,db=launch()
    r=client.post(LISTS,json={'name':'My research group','tickers':SYMBOLS})
    assert r.status_code==201,r.text
    saved=r.json();uid=create_plan(client,saved['id'])
    result=wait_scan(client,uid)
    assert not result['last_error'],result
    assert result['last_result']['candidates']>=2,result
    rows=client.get('/api/pairs/opportunities').json()
    assert rows and all(r['discovery']['list_name']=='My research group' for r in rows)
    assert all(r['relative_value']['sector']=='XLK' for r in rows)
    assert any(r['relative_value']['category']=='basket' for r in rows)
    count=len(rows)
    client.post(f'{SCHEDULES}/{uid}/run',json={})
    second=wait_scan(client,uid,result['last_finished'])
    assert len(client.get('/api/pairs/opportunities').json())==count
    assert second['last_result']['created']==[]
    process.terminate();process.wait(timeout=20)
    client,_,_=launch(database=db)
    assert client.get(LISTS).json()[0]['tickers']==SYMBOLS
    assert client.get('/api/pairs/opportunities').json()[0]['discovery']
    # Removal invalidates pending discovery and pauses affected generated rules.
    response=client.put(LISTS+'/'+saved['id'],json={'name':'Smaller list','tickers':['AAPL','MSFT']})
    assert response.status_code==200,response.text
    for rule in client.get('/api/pairs/alerts').json():
        traded={rule['ticker_a']}|{r['ticker'] for r in rule['relative_value']['references']}
        if not traded <= {'AAPL','MSFT'}:assert rule['active']==0
    assert client.delete(LISTS+'/'+saved['id']).status_code==204
    assert client.get(LISTS).json()==[]
    assert all(r['active']==0 for r in client.get('/api/pairs/alerts').json())
    assert client.get(SCHEDULES).json()==[]


def test_actions_do_not_show_trades_when_paused_stale_failed_or_outside_stop(launch):
    client,_,db=launch()
    model={'category':'competitor','target':'AAPL','references':[{'ticker':'MSFT','weight':1}],'market':'SPY','sector':'XLK','factors':['MTUM','VLUE'],'lookback':120,'entry_z':2}
    r=client.post('/api/pairs/alerts',json={'relative_value':model})
    assert r.status_code==201,r.text
    rule=r.json();snapshot=json.loads(json.dumps(rule['latest']));snapshot['value']=2.5;snapshot['matched']=True
    def update(**fields):
        with sqlite3.connect(db) as d:d.execute('UPDATE pair_alerts SET '+','.join(k+'=?' for k in fields)+' WHERE id=?',[*fields.values(),rule['id']])
        return client.get('/api/pairs/opportunities').json()[0]['action']
    action=update(latest_json=json.dumps(snapshot))
    assert action['state']=='entry'
    assert any(leg['role']=='factor hedge' for leg in action['legs'])
    action=update(active=0);assert action['state']=='paused' and action['legs']==[]
    action=update(active=1,last_error='Provider unavailable');assert action['state']=='error' and not action['legs']
    snapshot['value']=5;action=update(last_error=None,latest_json=json.dumps(snapshot));assert action['state']=='avoid' and not action['legs']
    snapshot['value']=2.5;snapshot['as_of']='2000-01-01';action=update(latest_json=json.dumps(snapshot));assert action['state']=='stale' and not action['legs']
    snapshot['as_of']=rule['latest']['as_of'];snapshot['matched']=False;action=update(latest_json=json.dumps(snapshot));assert action['state']=='wait' and not action['legs']


def test_list_validation_and_unknown_classification_errors(client):
    for body in [{'name':'','tickers':SYMBOLS},{'name':'x','tickers':['AAPL']},{'name':'x','tickers':[True,'AAPL']},{'name':'x','tickers':['AAPL','aapl']}]:
        assert client.post(LISTS,json=body).status_code==422
    saved=client.post(LISTS,json={'name':'Unavailable stocks','tickers':['UNKNOWN','MISSING']}).json()
    uid=create_plan(client,saved['id'])
    result=wait_scan(client,uid)
    assert result['last_error']
    assert client.get('/api/pairs/opportunities').json()==[]
    for settings in [{'stock_list_id':'missing'},{'algorithms':[]},{'algorithms':['invented']}]:
        r=client.post(SCHEDULES,json={'source':'stock_list','stock_list_id':saved['id'],'algorithms':['residual_pair'],**settings})
        assert r.status_code in (404,422),r.text


def test_list_edit_during_discovery_discards_old_universe(launch):
    client,_,_=launch(STOCKASSIST_TEST_PROFILE_DELAY_MS='150')
    saved=client.post(LISTS,json={'name':'Original group','tickers':SYMBOLS}).json()
    uid=create_plan(client,saved['id'])
    for _ in range(200):
        if next(r for r in client.get(SCHEDULES).json() if r['id']==uid)['running']:break
        time.sleep(.02)
    else:raise AssertionError('Schedule did not start')
    changed=client.put(LISTS+'/'+saved['id'],json={'name':'Updated group','tickers':['AAPL','MSFT']})
    assert changed.status_code==200,changed.text
    result=wait_scan(client,uid)
    assert result['last_result']['list']['version']==2,result
    assert result['last_result']['list']['tickers']==['AAPL','MSFT']
    for row in client.get('/api/pairs/opportunities').json():
        assert row['discovery']['list_version']==2
        assert {row['ticker_a']}|{r['ticker'] for r in row['relative_value']['references']} <= {'AAPL','MSFT'}


def test_empty_scan_explains_correlation_filter_and_small_basket(client):
    saved=client.post(LISTS,json={'name':'Small list','tickers':['AAPL','MSFT']}).json()
    uid=create_plan(client,saved['id'],min_correlation=1)
    result=wait_scan(client,uid)['last_result']
    assert result['evaluated']==0
    reasons=' '.join(item['reason'] for item in result['skipped'])
    assert 'correlation' in reasons
    assert 'at least three stocks' in reasons
