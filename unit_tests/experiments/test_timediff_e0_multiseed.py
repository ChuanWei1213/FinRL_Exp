import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from finrl.experiments import timediff_e0_multiseed as m


def bank(n=100):
    x=np.ones((n,92,10,3));x[:,:,:,1]=1.01;x[:,:,:,2]=.99
    return x


def test_seed_zero_streams_match_previous_implementation():
    a=m.Controller(bank(),bank(),'D',seed=0)
    b=m.k.Controller(bank(),bank(),'D',seed=0)
    assert a.state()==b.state()


def test_seeds_change_pool_path_and_k_streams():
    a=m.Controller(bank(),bank(),'D',seed=1)
    b=m.Controller(bank(),bank(),'D',seed=2)
    assert a.members!=b.members
    assert a.state()['samplers']!=b.state()['samplers']
    assert [a.draw_k(0) for _ in range(50)]!=[b.draw_k(0) for _ in range(50)]
    c=m.Controller(bank(),bank(),'D',seed=1)
    assert c.members==a.members


def test_seeded_resets_pair_real_and_synthetic_k_and_change_with_seed():
    sequences=[]
    for seed in [1,2]:
        c=m.Controller(bank(),bank(),'C',seed=seed)
        envs=[m.PilotEnv(c,i) for i in range(16)]
        for i,e in enumerate(envs):e.reset(seed=seed+i)
        assert all(envs[i].k==envs[i+8].k for i in range(8))
        sequences.append([e.k for e in envs])
    assert sequences[0]!=sequences[1]


def test_identical_within_seed_initialization_and_different_across_seeds():
    torch.set_num_threads(1)
    hashes={}
    for seed in [1,2]:
        current=[]
        for group in ['R','C','D']:
            c=m.Controller(bank(),bank(),group,seed=seed)
            vec=DummyVecEnv([lambda:m.PilotEnv(c)])
            model=PPO('MultiInputPolicy',vec,seed=seed,device='cpu',n_steps=21,batch_size=21,
                policy_kwargs={'features_extractor_class':m.EIIEWindowExtractor,'net_arch':dict(pi=[64],vf=[64])})
            current.append(m.parameter_hash(model));vec.close()
        assert len(set(current))==1
        hashes[seed]=current[0]
    assert hashes[1]!=hashes[2]


def test_c_d_k_streams_survive_natural_pool_switches():
    controllers=[m.Controller(bank(),bank(),g,seed=3) for g in ['C','D']]
    pairs=[]
    for c in controllers:
        envs=[m.PilotEnv(c,i) for i in range(16)]
        for i,e in enumerate(envs):e.reset(seed=3+i)
        pairs.append(envs)
    for step in range(560):
        for envs in pairs:
            for e in envs:
                _,_,done,_,_=e.step_weights(np.r_[1.,np.zeros(10)])
                if done:e.reset()
        if step==503:
            controllers[1].rotate()
            for e in pairs[1]:e.rebind_idle()
    for a,b in zip(*pairs):
        assert a.k==b.k and a.day==b.day
        np.testing.assert_array_equal(a.k_counts,b.k_counts)
        np.testing.assert_array_equal(a.completed,b.completed)


def test_batched_environment_matches_reference_through_reset_and_pool_rotation():
    rng=np.random.default_rng(98)
    close=np.exp(np.cumsum(rng.normal(0,.007,(100,92,10)),axis=1))
    prices=np.stack([close,close*1.01,close*.99],axis=-1)
    for group in ['C','D']:
        cs=[m.Controller(prices,prices,group,seed=5) for _ in range(2)]
        envs=[[m.PilotEnv(c,i) for i in range(16)] for c in cs]
        vecs=[kind([lambda e=e:e for e in es]) for kind,es in zip([DummyVecEnv,m.BatchVecEnv],envs)]
        for v in vecs:v.seed(5)
        a,b=[v.reset() for v in vecs]
        for key in a:np.testing.assert_array_equal(a[key],b[key])
        for step in range(550):
            action=rng.uniform(-1,1,(16,11)).astype(np.float32)
            x,y=[v.step(action) for v in vecs]
            for key in x[0]:np.testing.assert_allclose(x[0][key],y[0][key],atol=2e-7,rtol=1e-7)
            np.testing.assert_allclose(x[1],y[1],atol=1e-6,rtol=1e-6)
            np.testing.assert_array_equal(x[2],y[2])
            for ia,ib in zip(x[3],y[3]):
                for key in ['source','path_id','k','prefix','pool_generation']:assert ia[key]==ib[key]
                for key in ['wealth','cost_fraction','turnover','exposure_change_from_trade']:
                    np.testing.assert_allclose(ia[key],ib[key],atol=1e-10,rtol=1e-10)
                if 'terminal_observation' in ia:
                    for key in ia['terminal_observation']:
                        np.testing.assert_allclose(ia['terminal_observation'][key],ib['terminal_observation'][key],atol=2e-7,rtol=1e-7)
                    np.testing.assert_allclose(ia['episode']['suffix_r'],ib['episode']['suffix_r'],atol=1e-8)
            if group=='D' and step==503:
                for c,es in zip(cs,envs):
                    c.rotate()
                    for e in es:e.rebind_idle()
        for v in vecs:v.close()
        assert cs[0].state()==cs[1].state()
