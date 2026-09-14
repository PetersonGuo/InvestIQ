import math
import time

SYMBOLS=['AAPL','MSFT','NVDA','GOOGL','AMZN','META','TSLA','XLK','MTUM','VLUE','SPY']

def analyze(client,**extra):
    r=client.post('/api/pairs/valuation',json={'tickers':SYMBOLS,**extra})
    assert r.status_code==200,r.text
    return r.json()

def test_valuation_estimates_explainable_and_future_filings_excluded(client):
    result=analyze(client)
    assert not result['errors'],result
    assert len(result['rows'])==11
    msft=next(r for r in result['rows'] if r['ticker']=='MSFT')
    assert {'LiabilitiesCurrent','LiabilitiesNoncurrent'} <= {f['tag'] for f in msft['evidence']}
    target=next(r for r in result['rows'] if r['ticker']=='AAPL')
    assert target['status']=='estimated',target
    assert target['valuation_direction']=='cheap'
    assert target['gap'] < -.5
    assert len(target['peers'])==10 and 'AAPL' not in target['peers']
    reconstructed=math.log(target['baseline_multiple'])+sum(c['log_multiple_contribution'] for c in target['contributions'])
    assert math.exp(reconstructed)==target['expected_multiple']
    assert target['lower_multiple']<target['expected_multiple']<target['upper_multiple']
    assert all(f['filed'] < result['as_of'] for r in result['rows'] for f in r['evidence'])
    assert all(r['action']['legs']==[] for r in result['rows']) # no explicit sector confirmation
    assert 'sector ETF' in target['confirmation_error']

def test_target_price_does_not_leak_into_its_peer_estimate(launch):
    c1,_,_=launch(STOCKASSIST_TEST_AAPL_PE='5')
    c2,_,_=launch(STOCKASSIST_TEST_AAPL_PE='10')
    r1=next(r for r in analyze(c1)['rows'] if r['ticker']=='AAPL')
    r2=next(r for r in analyze(c2)['rows'] if r['ticker']=='AAPL')
    assert r1['multiple']!=r2['multiple']
    assert r1['expected_multiple']==r2['expected_multiple']
    assert r1['lower_multiple']==r2['lower_multiple']
    assert r1['upper_multiple']==r2['upper_multiple']

def test_valuation_reference_universe_and_no_fake_historical_backtest(client):
    small=analyze(client,tickers=['AAPL','MSFT'])
    assert all(r['status']=='insufficient_peers' and not r['action']['legs'] for r in small['rows'])
    expanded=analyze(client,tickers=['AAPL','MSFT'],valuation_peers=SYMBOLS)
    assert len(expanded['rows'])==2
    assert expanded['rows'][0]['status']=='estimated'
    assert client.post('/api/pairs/valuation',json={'tickers':SYMBOLS,'as_of':'2020-01-01'}).status_code==422
    assert client.post('/api/pairs/valuation',json={'tickers':SYMBOLS,'valuation_peers':[True]}).status_code==422
    unavailable=analyze(client,tickers=['UNKNOWN','MISSING'])
    assert unavailable['rows']==[] and len(unavailable['errors'])==2

def test_valuation_schedule_persists_and_list_edits_invalidate(launch):
    c,p,db=launch()
    saved=c.post('/api/pairs/stock-lists',json={'name':'Valuation research','tickers':['AAPL','MSFT']}).json()
    response=c.post('/api/pairs/schedules',json={'source':'stock_list','stock_list_id':saved['id'],'algorithms':['peer_valuation'],'valuation_peers':SYMBOLS})
    assert response.status_code==201,response.text
    uid=response.json()['id']
    for _ in range(150):
        row=next(r for r in c.get('/api/pairs/schedules').json() if r['id']==uid)
        if row['last_finished']:break
        time.sleep(.1)
    assert row['last_result']['valuation']['rows'],row
    p.terminate();p.wait(timeout=20)
    c,_,_=launch(database=db)
    assert c.get('/api/pairs/schedules').json()[0]['last_result']['valuation']==row['last_result']['valuation']
    assert c.get('/api/pairs/valuation/events').status_code==200
    assert c.put('/api/pairs/stock-lists/'+saved['id'],json={'name':'Updated','tickers':['AAPL','NVDA']}).status_code==200
    assert c.get('/api/pairs/schedules').json()[0]['last_result'] is None
