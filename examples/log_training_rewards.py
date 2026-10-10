"""Re-run one scenario-mix PPO training with full logging. Training is unchanged.

Same env / PPO / seed / eval schedule as train_scenario_mix_ppo.py, so the run reproduces
the original one step for step (check: validation curve equals the original *_valid.csv).
Extra outputs, nothing else:
    episodes.csv   one row per training episode: timesteps, reward (=100*log V_end/V_start),
                   length, source (real/synthetic), scenario, k
    valid.csv      validation final value / reward / sharpe / calmar / stock weight per eval
    ckpt/step_<t>.zip   every evaluated checkpoint

Run from FinRL_Exp/ (one group/seed per process):
    python examples/log_training_rewards.py \
        --prices ../TimeDiff/datasets/dow10/prices_2004.csv \
        --synthetic ../TimeDiff/outputs/dow10_cond/rl_paths/paths.npz \
        --group B90 --p 0.9 --seed 2 --out results/scenario_mix/v2_rewardlog/B90_seed2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from finrl.experiments import scenario_mix as S  # noqa: E402
from train_scenario_mix_ppo import TICKERS, make_ppo, validation_score  # noqa: E402


class SourceInfo(gym.Wrapper):
    """Copy the episode source into info at termination (before the VecEnv auto-reset)."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            src, scen, k = self.env.last_source
            info = {**info, "source": src, "scenario": -1 if scen is None else scen, "k": k}
        return obs, reward, terminated, truncated, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prices", type=Path, required=True)
    ap.add_argument("--synthetic", type=Path, required=True)
    ap.add_argument("--group", required=True)
    ap.add_argument("--p", type=float, required=True, help="synthetic share")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train", nargs=2, default=["2010-01-01", "2017-12-31"])
    ap.add_argument("--valid", nargs=2, default=["2018-01-01", "2019-12-31"])
    ap.add_argument("--timesteps", type=int, default=500_000)
    ap.add_argument("--eval-freq", type=int, default=25_000)
    ap.add_argument("--commission", type=float, default=0.0025, help="validation (and test) commission")
    ap.add_argument("--train-commission", type=float, default=None,
                    help="commission inside training episodes (default: same as --commission)")
    ap.add_argument("--k-max", type=int, default=21)
    ap.add_argument("--log-std-init", type=float, default=-2.0)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    cli = ap.parse_args()

    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    torch.set_num_threads(1)
    if cli.out.exists() and any(cli.out.iterdir()):
        raise FileExistsError(f"{cli.out} is not empty")
    (cli.out / "ckpt").mkdir(parents=True, exist_ok=True)
    (cli.out / "config.json").write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(cli).items()}, indent=2) + "\n")

    episodes = []

    class EpisodeLog(BaseCallback):
        def _on_step(self):
            for info in self.locals["infos"]:
                if "episode" in info:
                    episodes.append({"timesteps": self.num_timesteps, "reward": info["episode"]["r"],
                                     "length": info["episode"]["l"], "source": info.get("source"),
                                     "scenario": info.get("scenario"), "k": info.get("k")})
            return True

    market = S.MarketData(cli.prices, TICKERS)
    synthetic = S.SyntheticPaths(cli.synthetic, market)
    env = S.ScenarioMixEnv(market, synthetic, p_synthetic=cli.p, data_start=cli.train[0], data_end=cli.train[1],
                           commission=cli.commission if cli.train_commission is None else cli.train_commission,
                           k_max=cli.k_max, seed=cli.seed)
    env = Monitor(SourceInfo(env))
    model = make_ppo(env, cli.seed, "cpu", cli.log_std_init, cli.ent_coef)
    policy = lambda obs: model.predict(obs, deterministic=True)[0]  # noqa: E731
    cb, history = EpisodeLog(), []
    for done in range(cli.eval_freq, cli.timesteps + 1, cli.eval_freq):
        model.learn(cli.eval_freq, reset_num_timesteps=False, callback=cb)
        vdaily = S.run_continuous(policy, market, *cli.valid, cli.commission)
        fv = float(vdaily.value.iloc[-1])
        history.append({"timesteps": done, "valid_final_value": fv, "valid_reward": 100 * np.log(fv),
                        "valid_sharpe": validation_score(vdaily, "sharpe"),
                        "valid_calmar": validation_score(vdaily, "calmar"),
                        "valid_mean_stock_weight": float(1 - vdaily.cash_weight.mean())})
        model.save(cli.out / "ckpt" / f"step_{done:06d}")
        pd.DataFrame(history).to_csv(cli.out / "valid.csv", index=False)
        pd.DataFrame(episodes).to_csv(cli.out / "episodes.csv", index=False)
        print(f"{cli.group} seed {cli.seed} t={done}: valid {fv:.4f}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
