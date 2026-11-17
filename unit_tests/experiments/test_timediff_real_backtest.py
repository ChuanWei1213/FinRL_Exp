import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('real_backtest',ROOT/'scripts/backtest_timediff_real_events.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)


class Policy:
    def __init__(self,action=None):
        self.action=np.zeros(11) if action is None else action
        self.seen=[]
    def predict(self,obs,deterministic=True):
        self.seen.append({k:v.copy() for k,v in obs.items()})
        return self.action[None],None


def fixture():
    m=b.Market.__new__(b.Market)
    m.dates=pd.bdate_range('2005-01-03',periods=120)
    m.prices=np.ones((120,10,3));m.prices[:,:,1]=1.01;m.prices[:,:,2]=.99
    m.market_return=pd.Series(np.zeros(120),index=m.dates)
    m.previous5=m.market_return.copy()
    date=lambda i:m.dates[i].strftime('%Y-%m-%d')
    f={'allowed_buffer_start':date(68),'allowed_buffer_end':date(94),'condition_start':date(70),
       'condition_end':date(90),'activation_start':date(72),'activation_end_exclusive':date(76),'predicted_return':-.1}
    return m,f


def test_cash_is_exactly_flat_and_first_stock_purchase_is_charged():
    m,f=fixture();cash=b.run_account(m,{},f,'cash','aligned21')
    assert len(cash)==21 and (cash.wealth==1).all() and cash.fee_fraction.sum()==0
    stock=b.run_account(m,{},f,'equal_weight','aligned21')
    assert stock.wealth.iloc[0]==pytest.approx(1/(1+b.r.FEE))
    assert stock.fee_fraction.iloc[0]>0 and stock.fee_fraction.iloc[1:].sum()==pytest.approx(0,abs=1e-12)
    v=b.metrics(stock,f['condition_start'],f['condition_end'])
    assert v['net_return']==pytest.approx(-b.r.FEE/(1+b.r.FEE))
    assert v['max_drawdown']==pytest.approx(v['net_return'])


def test_router_switches_keep_holdings_and_wealth_continuous():
    m,f=fixture();models={'R':Policy(),'C':Policy(np.r_[1.,-np.ones(10)])}
    d=b.run_account(m,models,f,'C','timed')
    assert d.active_policy.tolist().count('C')==4
    assert d.policy_switch.sum()==2
    np.testing.assert_allclose(d.wealth_before.iloc[1:],d.wealth.iloc[:-1],rtol=0,atol=0)
    switch=d[d.policy_switch]
    assert (switch.fee_fraction>0).all()
    assert (switch.wealth_before<1).all()
    assert not (switch.stock_exposure_before==0).any()
    first_C=models['C'].seen[0]['last_action'][0]
    assert first_C[1:].sum()>.8


def test_future_prices_cannot_enter_first_decision():
    a,f=fixture();c,_=fixture();index=c.dates.get_loc(pd.Timestamp(f['condition_start']))
    c.prices[index:]*=3
    p=Policy();q=Policy()
    b.run_account(a,{'R':p},f,'R','aligned21')
    b.run_account(c,{'R':q},f,'R','aligned21')
    for key in ('state','last_action'):np.testing.assert_array_equal(p.seen[0][key],q.seen[0][key])
    assert set(p.seen[0])=={'state','last_action'}


def test_event_slice_uses_existing_account_instead_of_rebuilding():
    m,f=fixture();d=b.run_account(m,{},f,'equal_weight','timed')
    v=b.metrics(d,f['condition_start'],f['condition_end'])
    assert v['net_return']==pytest.approx(0,abs=1e-12)
    assert v['cost_pct_start_capital']==pytest.approx(0,abs=1e-12)
    assert b.metrics(d,f['allowed_buffer_start'],f['allowed_buffer_end'])['net_return']<0


def test_event_slice_counts_switch_on_its_first_day():
    m,f=fixture();f['activation_start']=f['condition_start']
    d=b.run_account(m,{'R':Policy(),'C':Policy(np.r_[1.,-np.ones(10)])},f,'C','timed')
    v=b.metrics(d,f['condition_start'],f['condition_end'])
    assert v['switch_count']==2 and v['cost_pct_start_capital']>0


def test_behavior_uses_known_signal_and_excludes_price_drift():
    frame=pd.DataFrame({'decision_date':['2020-01-02']*4,'case_schedule_active':[True]*4,
        'rebound_signal':[True]*4,'pullback_signal':[False]*4,
        'exposure_change_trade':[-.006,.008,.005,0.],
        'exposure_change_price':[.1,-.1,.2,-.3]})
    v=b.behavior(frame,{'predicted_return':-.1},'2020-01-01','2020-01-03')
    assert v['trigger_count']==4 and v['expected_effective']==1 and v['opposite_effective']==1
    assert v['below_threshold']==2 and v['sample_insufficient']
    z=b.behavior(frame,{'predicted_return':-.1},'2021-01-01','2021-01-03')
    assert z['trigger_count']==0 and z['expected_effective_rate'] is None


def test_source_forecast_dates_do_not_move_with_time_error():
    m,f=fixture()
    a=b.run_account(m,{},f,'cash','aligned21')
    f['activation_start']=f['allowed_buffer_start'];f['activation_end_exclusive']=f['allowed_buffer_end']
    c=b.run_account(m,{},f,'cash','aligned21')
    assert a.date.tolist()==c.date.tolist() and len(a)==21
