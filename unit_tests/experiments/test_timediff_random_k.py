import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from finrl.experiments import timediff_random_k as r


def bank(n=100):
    x=np.ones((n,92,10,3));x[:,:,:,1]=1.01;x[:,:,:,2]=.99
    return x


def env_at(k,prices=None):
    c=r.Controller(bank() if prices is None else prices,None,'R')
    c.draw_k=lambda slot:k
    e=r.PilotEnv(c);e.reset()
    return e


@pytest.mark.parametrize('k',[0,1,13,21])
def test_exact_length_cash_reward_and_no_dummy_step(k):
    e=env_at(k)
    for t in range(k+21):
        _,rew,done,_,_=e.step_weights(np.r_[1.,np.zeros(10)])
        assert done==(t==k+20) and rew==0
    assert e.prefix_days==k and e.suffix_days==21
    assert e.wealth==1 and e.k_counts[k]==1
    with pytest.raises(RuntimeError):e.step(np.zeros(11))


def test_prelude_is_traded_and_holdings_carry_across_boundary():
    p=bank();p[:,50:,0,:]*=np.exp(np.arange(42)[None,:,None]*.002)
    e=env_at(21,p)
    w=np.r_[.3,.7,np.zeros(9)]
    first=e.step_weights(w)
    assert first[-1]['cost_fraction']>0
    for _ in range(20):e.step_weights(w)
    wealth=e.wealth;weights=e.weights.copy();obs=e._obs()
    assert e.day==21 and not np.array_equal(weights,np.r_[1.,np.zeros(10)])
    np.testing.assert_array_equal(obs['last_action'],weights.astype(np.float32))
    e.step_weights(weights)
    assert e.prefix_wealth==wealth and e.wealth!=1


def test_future_changes_do_not_change_pre_transition_observation():
    p=bank();q=bank();q[:,71:]*=5
    a=env_at(21,p);b=env_at(21,q)
    for _ in range(21):
        np.testing.assert_array_equal(a._obs()['state'],b._obs()['state'])
        a.step_weights(np.r_[1.,np.zeros(10)]);b.step_weights(np.r_[1.,np.zeros(10)])
    np.testing.assert_array_equal(a._obs()['state'],b._obs()['state'])
    assert set(a._obs())=={'state','last_action'}


def test_paired_sources_use_uniform_k_and_rng_restore():
    c=r.Controller(bank(),bank(),'C');saved=c.state()
    a=np.array([c.draw_k(0) for _ in range(22000)])
    b=np.array([c.draw_k(8) for _ in range(22000)])
    np.testing.assert_array_equal(a,b)
    assert set(a)==set(range(22))
    assert abs(a.mean()-10.5)<.2
    c.restore(saved)
    np.testing.assert_array_equal([c.draw_k(0) for _ in range(20)],a[:20])


def test_pool_rotation_never_resets_inflight_account_or_k():
    c=r.Controller(bank(),bank(),'D');e=r.PilotEnv(c);e.reset()
    e.step_weights(np.r_[0.,1.,np.zeros(9)])
    prior=e.state();c.rotate();e.rebind_idle()
    assert e.source==prior['source'] and e.index==prior['index']
    assert e.wealth==prior['wealth'] and e.day==prior['day'] and e.k==prior['k']
    while e.day<e.k+21:e.step_weights(np.r_[1.,np.zeros(10)])
    e.reset()
    assert e.generation_at_start==c.generation


def test_batch_evaluation_matches_traded_episode_including_prefix_fees():
    class Policy:
        def predict(self,obs,deterministic):return np.zeros((len(obs['state']),11)),None
    p=bank(4);p[:,50:,0,:]*=np.exp(np.arange(42)[None,:,None]*.001)
    ks=np.array([0,1,13,21]);scores=r.evaluate(Policy(),p,ks,True)
    for i,k in enumerate(ks):
        e=env_at(int(k),p[i:i+1]);total=0
        for _ in range(k+21):total+=e.step(np.zeros(11))[1]
        assert total==pytest.approx(scores['full'][i],abs=1e-10)
        assert 100*np.log(e.wealth/e.prefix_wealth)==pytest.approx(scores['suffix'][i],abs=1e-10)
        assert scores['full'][i]==pytest.approx(scores['prefix'][i]+scores['suffix'][i],abs=1e-10)


def test_gym_api():
    check_env(r.PilotEnv(r.Controller(bank(),None,'R')),skip_render_check=True)


def test_zero_k_reproduces_previous_21_day_environment_exactly():
    p=bank(1);p[:,50:,0,:]*=np.exp(np.arange(42)[None,:,None]*.001)
    new=env_at(0,p)
    old=r.original.PilotEnv(r.original.Controller(p[:,21:],None,'R'))
    old.reset()
    for _ in range(21):
        for key in ('state','last_action'):np.testing.assert_array_equal(new._obs()[key],old._obs()[key])
        a=new.step(np.linspace(-.1,.1,11));b=old.step(np.linspace(-.1,.1,11))
        assert a[1:4]==b[1:4]
        assert new.wealth==old.wealth
