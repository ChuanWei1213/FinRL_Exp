import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from finrl.experiments.timediff_scenario_pilot import (
    Controller,PilotEnv,EIIEWindowExtractor,action_weights,transact,
    parameter_hash,evaluate,observations,FEE,
)


def bank(n=16):
    values=np.ones((n,71,10,3))
    values[:,:,:,1]=1.01;values[:,:,:,2]=.99
    return values


def test_initial_purchase_and_sale_both_charge_fee():
    cash=np.r_[1.,np.zeros(10)];stock=np.r_[0.,1.,np.zeros(9)]
    factor,weights,cost,_=transact(cash,stock,np.ones(11))
    assert factor==pytest.approx(1/(1+FEE),abs=1e-13)
    assert cost>0
    factor,_,_,_=transact(stock,cash,np.ones(11))
    assert factor==pytest.approx(1-FEE,abs=1e-13)
    other=np.r_[0.,0.,1.,np.zeros(8)]
    factor,_,_,_=transact(stock,other,np.ones(11))
    assert factor==pytest.approx((1-FEE)/(1+FEE),abs=1e-13)


def test_fee_matches_independent_cash_budget_equation():
    previous=np.array([.2,.3,.5]+[0.]*8)
    target=np.array([.1,.7,.2]+[0.]*8)
    factor,_,cost,_=transact(previous,target,np.ones(11))
    trades=factor*target[1:]-previous[1:]
    remaining_cash=previous[0]-trades.sum()-FEE*np.abs(trades).sum()
    assert remaining_cash==pytest.approx(factor*target[0],abs=1e-12)
    assert cost==pytest.approx(FEE*np.abs(trades).sum(),abs=1e-12)


def test_cash_has_zero_reward_and_exactly_21_trades():
    env=PilotEnv(Controller(bank(),None,'R'))
    env.reset();rewards=[]
    for i in range(21):
        obs,reward,done,_,_=env.step_weights(np.r_[1.,np.zeros(10)])
        assert done==(i==20)
        rewards.append(reward)
    assert sum(rewards)==0
    assert env.wealth==1
    with pytest.raises(RuntimeError,match='already complete'):env.step(np.zeros(11))


def test_no_future_or_forecast_label_in_observation():
    a=bank();b=bank();b[:,50:]*=5
    x=PilotEnv(Controller(a,None,'R'));y=PilotEnv(Controller(b,None,'R'))
    ox,_=x.reset();oy,_=y.reset()
    assert set(ox)=={'state','last_action'}
    np.testing.assert_array_equal(ox['state'],oy['state'])
    np.testing.assert_array_equal(ox['last_action'],np.r_[1.,np.zeros(10)])
    assert ox['state'].shape==(3,10,50)


def test_net_reward_telescopes_and_drift_uses_executed_holdings():
    prices=bank();prices[:,50:,0,:]*=1.05
    env=PilotEnv(Controller(prices,None,'R'));env.reset()
    targets=np.r_[.5,.5,np.zeros(9)]
    total=0
    for _ in range(21):
        _,reward,_,_,info=env.step_weights(targets);total+=reward
    assert total==pytest.approx(100*np.log(env.wealth),abs=1e-11)
    assert info['previous_wealth']>0
    assert np.sum(env.weights)==pytest.approx(1)


def test_actions_are_long_only_without_leverage():
    for a in [np.full(11,-100),np.linspace(-20,20,11),np.zeros(11)]:
        weights=action_weights(a)
        assert (weights>=0).all() and weights.sum()==pytest.approx(1)


def test_sources_are_exactly_balanced_and_large_pool_keeps_16_active():
    controller=Controller(bank(100),bank(100),'D',pool_size=64)
    envs=[PilotEnv(controller,i) for i in range(16)]
    for env in envs:
        env.reset()
        for _ in range(21):env.step_weights(np.r_[1.,np.zeros(10)])
    assert sum(e.completed[0] for e in envs)==8
    assert sum(e.completed[1] for e in envs)==8
    assert len(controller.members)==64


def test_controller_rng_can_restore_selection_stream():
    controller=Controller(bank(100),bank(100),'C')
    state=controller.state()
    expected=[controller.pick(i) for i in range(16)]
    controller.restore(state)
    assert [controller.pick(i) for i in range(16)]==expected


def test_vectorized_evaluation_charges_initial_build_and_matches_environment():
    class EqualPolicy:
        def predict(self,obs,deterministic):return np.zeros((len(obs['state']),11)),None
    prices=bank(3)
    score=evaluate(EqualPolicy(),prices)
    env=PilotEnv(Controller(prices,None,'R'));env.reset()
    total=sum(env.step(np.zeros(11))[1] for _ in range(21))
    np.testing.assert_allclose(score,total,atol=1e-12)
    assert (score<0).all()


def test_gym_api_and_seeded_initial_parameters():
    env=PilotEnv(Controller(bank(),None,'R'))
    check_env(env,skip_render_check=True)
    hashes=[]
    for mode in ('R','C'):
        vector=DummyVecEnv([lambda mode=mode:PilotEnv(Controller(bank(),bank(),mode))])
        model=PPO('MultiInputPolicy',vector,seed=0,device='cpu',n_steps=21,batch_size=21,
                  policy_kwargs={'features_extractor_class':EIIEWindowExtractor,'net_arch':dict(pi=[64],vf=[64])})
        hashes.append(parameter_hash(model));vector.close()
    assert hashes[0]==hashes[1]
