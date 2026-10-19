"""Matched controls: unchanged PPO; real replay replaces only generated futures."""
import sys
sys.modules.setdefault("quantstats", None)

from finrl.experiments import scenario_mix as S
from finrl.experiments import train_scenario_mix_logged as T


class RealReplayEnv(S.ScenarioMixEnv):
    """Replay the paired real future with the same path-choice RNG and eligibility."""

    def frame_synthetic(self, j, k):
        # Use the true exchange calendar as well as the true OHLC observations.
        a = int(self.syn.anchor_idx[j])
        if a + self.horizon > self.hi:
            raise ValueError("Real replay future extends beyond the training period")
        return self.frame_real(a - k, k)

    def _build(self):
        inner = super()._build()
        source, scenario, k = self.last_source
        if source == "synthetic":
            self.last_source = ("real_replay", scenario, k)
        return inner


def main():
    args = T.build_parser().parse_args()
    args.control_protocol = "matched_50pct_replacement_branch"
    args.replay_pool = args.group.startswith("REAL_REPLAY_")
    if args.replay_pool:
        S.ScenarioMixEnv = RealReplayEnv
    T.run(args)


if __name__ == "__main__":
    main()
