from conftest import wait_run
from test_http import query
from test_relative_value import MODEL


def test_engine_execution_costs_override_user_parameters(client):
    q=query(client)
    q.update(tickers=['AAPL','MSFT'],commission=1,slippage_bps=5,borrow_rate_percent=3,warmup=2,
      params={'_sa_commission':999,'_sa_slippage_bps':999,'_sa_borrow_rate_percent':999},
      code='''#include "strategy_api.h"
extern "C" int on_portfolio(const SA_Asset*,int,int,double,double,const char*p,double*) {
 return sa_parameter(p,"_sa_commission",-1)==1 && sa_parameter(p,"_sa_slippage_bps",-1)==5 && sa_parameter(p,"_sa_borrow_rate_percent",-1)==3 ? 0 : -1;
}''')
    r=client.post('/api/backtests',json=q);assert r.status_code==202,r.text
    result=wait_run(client,r.json()['id']);assert result['status']=='completed',result


def test_cost_filter_diagnostics_and_ablation(client):
    draft=client.post('/api/pairs/relative-value/strategy',json={'relative_value':MODEL}).json()
    assert draft['params']['cost_aware']==1
    q={**query(client),**draft,'commission':1000,'slippage_bps':100,'borrow_rate_percent':50}
    r=client.post('/api/backtests',json=q);assert r.status_code==202,r.text
    run=wait_run(client,r.json()['id']);assert run['status']=='completed',run
    assert run['result']['fills']==[]
    assert run['result']['strategy_diagnostics']['entries_rejected_by_cost']>0
    assert run['result']['metrics']['total_slippage_cost']==0
    q.update(commission=1,slippage_bps=5,borrow_rate_percent=3)
    q['params']={**q['params'],'cost_aware':0}
    r=client.post('/api/backtests',json=q);run=wait_run(client,r.json()['id'])
    assert run['status']=='completed',run
    assert run['result']['fills']
    assert run['result']['strategy_diagnostics']['entries_rejected_by_cost']==0
    assert run['result']['metrics']['total_slippage_cost']>0


def test_replay_exact_dataset_works_without_provider_and_rejects_mismatch(launch):
    c,p,db=launch()
    q=query(c)
    initial=c.post('/api/backtests',json=q).json()
    saved=wait_run(c,initial['id']);assert saved['status']=='completed',saved
    p.terminate();p.wait(timeout=20)
    offline,_,_=launch(mode='ibkr',database=db)
    q.update(replay_run_id=saved['id'],commission=2)
    r=offline.post('/api/backtests',json=q);assert r.status_code==202,r.text
    replay=wait_run(offline,r.json()['id']);assert replay['status']=='completed',replay
    assert replay['result']['data']['sha256']==saved['result']['data']['sha256']
    assert replay['result']['data']['source']==saved['result']['data']['source']
    assert replay['result']['data']['replayed_from_run_id']==saved['id']
    q['tickers']=['MSFT'];q['ticker']='MSFT'
    bad=offline.post('/api/backtests',json=q).json()
    failed=wait_run(offline,bad['id']);assert failed['status']=='failed'
    assert 'Replay must preserve' in failed['error']
