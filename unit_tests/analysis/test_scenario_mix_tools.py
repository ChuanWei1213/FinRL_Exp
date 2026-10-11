from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from analysis.scenario_mix import compare_selection_metrics as selection
from analysis.scenario_mix import plot_reward_curves as rewards
from analysis.scenario_mix import summarize_scenario_mix as summary


ROOT = Path(__file__).resolve().parents[2]
TOOLS = [
    "summarize_scenario_mix",
    "compare_selection_metrics",
    "compare_checkpoint_runs",
    "plot_reward_curves",
]
METRICS = summary.KEY_METRICS
PERIODS = ["2020", "2021", "2022", "2025", "2020-2022", "2020-2025"]


def run_cli(args, cwd, tmp_path):
    env = {**os.environ, "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(tmp_path / "mpl")}
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, *map(str, args)],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.parametrize("name", TOOLS)
def test_legacy_import_and_three_command_forms(name, tmp_path):
    module = importlib.import_module(f"analysis.scenario_mix.{name}")
    path = ROOT / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"legacy_{name}", path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    for symbol, value in vars(module).items():
        if symbol.isupper() or getattr(value, "__module__", None) == module.__name__:
            assert getattr(old, symbol) is value
    for command, cwd in [
        ([path, "--help"], tmp_path),
        ([ROOT / "analysis/scenario_mix" / f"{name}.py", "--help"], tmp_path),
        (["-m", module.__name__, "--help"], ROOT),
    ]:
        assert "--help" in run_cli(command, cwd, tmp_path).stdout


def write_summary_runs(root, *, score=False, offset=0.0):
    dates = pd.to_datetime(
        [
            "2020-01-02",
            "2020-02-03",
            "2020-03-02",
            "2020-05-01",
            "2021-01-04",
            "2022-01-03",
            "2022-04-01",
            "2022-10-03",
            "2025-02-17",
            "2025-04-01",
            "2025-06-02",
        ]
    )
    all_rows = []
    for group, group_delta in [("A", 0.0), ("B50", 0.1)]:
        for seed in [0, 1]:
            run = root / f"{group}_seed{seed}"
            run.mkdir(parents=True)
            cfg = {
                "groups": [group],
                "seeds": [seed],
                "out": str(run),
                "commission": 0.0025,
            }
            (run / "config.json").write_text(json.dumps(cfg))
            rows = []
            for period in PERIODS:
                rows.append(
                    {
                        "group": group,
                        "seed": seed,
                        "period": period,
                        **{
                            metric: 0.2 + group_delta + seed * 0.02 + offset
                            for metric in METRICS
                        },
                    }
                )
                for baseline in ["buy_hold_1_11", "buy_hold_1_10"]:
                    rows.append(
                        {
                            "group": baseline,
                            "seed": None,
                            "period": period,
                            **{metric: 0.15 for metric in METRICS},
                        }
                    )
            pd.DataFrame(rows).to_csv(run / "summary.csv", index=False)
            all_rows.extend(rows)
            pd.DataFrame(
                {
                    "date": dates,
                    "value": np.linspace(1, 1.2 + group_delta, len(dates)),
                    "cash_weight": np.linspace(0.1, 0.3, len(dates)),
                    "turnover": 0.02,
                }
            ).to_csv(run / f"{group}_seed{seed}_test_daily.csv", index=False)
            for baseline in ["buy_hold_1_11", "buy_hold_1_10"]:
                pd.DataFrame(
                    {"date": dates, "value": np.linspace(1, 1.15, len(dates))}
                ).to_csv(run / f"{baseline}_test_daily.csv", index=False)
            valid = pd.DataFrame(
                {
                    "timesteps": [25000, 50000, 75000],
                    "valid_final_value": [1.1, 1.1, 1.05],
                    "valid_mean_stock_weight": [0.7, 0.6, 0.5],
                    "valid_stock_weight_std": [0.1] * 3,
                }
            )
            if score:
                valid["valid_score"] = [0.5, 0.7, 0.7]
            valid.to_csv(run / f"{group}_seed{seed}_valid.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "group": group,
                        "seed": seed,
                        "source": "real",
                        "scenario": np.nan,
                        "pre": 0.8,
                        "early_crash": 0.7,
                        "late_crash": 0.6,
                        "change_late_vs_pre": -0.2,
                    }
                ]
            ).to_csv(run / f"{group}_seed{seed}_crash_response.csv", index=False)
    pd.DataFrame(all_rows).drop_duplicates(["group", "seed", "period"]).to_csv(
        root / "all_runs.csv", index=False
    )
    pd.concat([pd.read_csv(p) for p in root.glob("*/*_crash_response.csv")]).to_csv(
        root / "crash_response_all.csv", index=False
    )


def write_logged_runs(root, *, explicit_steps):
    evaluation = []
    for group in ["A", "B50"]:
        for seed in [0, 1]:
            run = root / f"{group}_seed{seed}"
            run.mkdir(parents=True)
            (run / "config.json").write_text(json.dumps({"eval_freq": 25000}))
            valid = pd.DataFrame(
                {
                    "timesteps": [25000, 50000, 75000],
                    "valid_final_value": [1.1, 1.1, 1.05],
                    "valid_sharpe": [0.5, 0.7, 0.7],
                    "valid_reward": 100 * np.log([1.1, 1.1, 1.05]),
                    "valid_mean_stock_weight": [0.7, 0.6, 0.5],
                }
            )
            if explicit_steps:
                valid["actual_timesteps"] = [26624, 53248, 79872]
            valid.to_csv(run / "valid.csv", index=False)
            pd.DataFrame(
                {
                    "timesteps": [25000, 26624, 26625, 53248, 75001, 79872],
                    "reward": [1, 3, 5, 7, 9, 11],
                    "source": (
                        ["real"] * 6 if group == "A" else ["real", "synthetic"] * 3
                    ),
                }
            ).to_csv(run / "episodes.csv", index=False)
            for step, actual in [(25000, 26624), (50000, 53248), (75000, 79872)]:
                for period in ["2020", "2022", "2025", "2020-2025", "gfc"]:
                    row = {
                        "group": group,
                        "seed": seed,
                        "timesteps": step,
                        "period": period,
                        **{metric: step / 100000 + seed * 0.01 for metric in METRICS},
                    }
                    if explicit_steps:
                        row["actual_timesteps"] = actual
                    evaluation.append(row)
    pd.DataFrame(evaluation).to_csv(root / "checkpoint_eval.csv", index=False)


def snapshot(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


@pytest.fixture
def inputs(tmp_path):
    root = tmp_path / "inputs"
    write_summary_runs(root / "final")
    write_summary_runs(root / "sharpe", score=True, offset=0.05)
    write_logged_runs(root / "legacy", explicit_steps=False)
    write_logged_runs(root / "explicit", explicit_steps=True)
    return root


@pytest.mark.parametrize("name", TOOLS)
def test_new_and_legacy_outputs_match_and_keep_analysis_rules(name, inputs, tmp_path):
    original = snapshot(inputs)
    outputs = []
    for mode in ["new", "legacy"]:
        work = tmp_path / mode
        shutil.copytree(inputs, work)
        output = work / "output"
        if name == "summarize_scenario_mix":
            output = work / "final"
            args = ["--root", output]
        elif name == "compare_selection_metrics":
            args = [
                "--run",
                f"final_value={work / 'final'}:{work / 'final'}",
                "--run",
                f"sharpe={work / 'sharpe'}:{work / 'sharpe'}",
                "--out",
                output,
            ]
        elif name == "compare_checkpoint_runs":
            args = [
                "--run",
                f"old={work / 'legacy'}",
                "--run",
                f"new={work / 'explicit'}",
                "--out",
                output,
            ]
        else:
            output = work / "legacy"
            args = ["--root", output]
        command = (
            ["-m", f"analysis.scenario_mix.{name}"]
            if mode == "new"
            else [ROOT / "examples" / f"{name}.py"]
        )
        run_cli([*command, *args], ROOT if mode == "new" else tmp_path, tmp_path)
        outputs.append(output)
    expected_files = {
        "summarize_scenario_mix": [
            "all_runs.csv",
            "by_group.csv",
            "validation_curves.csv",
            "crash_response_all.csv",
            "fig_equity.png",
            "fig_drawdown.png",
            "fig_stock_weight.png",
            "fig_covid.png",
            "fig_2022.png",
            "fig_2025.png",
        ],
        "compare_selection_metrics": [
            "compare_table.csv",
            "selected_checkpoints.csv",
            "training_identical.csv",
            "crash_response_compare.csv",
            "fig_compare.png",
        ],
        "compare_checkpoint_runs": ["picked_table.csv", "fig_steps.png"],
        "plot_reward_curves": ["picked_steps.csv", "fig_reward_vs_steps.png"],
    }[name]
    for filename in expected_files:
        a, b = [output / filename for output in outputs]
        assert a.stat().st_size > 0
        assert a.read_bytes() == b.read_bytes(), filename
    out = outputs[0]
    if name == "summarize_scenario_mix":
        table = pd.read_csv(out / "all_runs.csv")
        assert (
            len(table) == len(PERIODS) * 6
        )  # four policy runs + two deduplicated baselines
        agg = summary.aggregate(table)
        assert agg.loc[("2020", "A"), ("return", "mean")] == pytest.approx(0.21)
        assert agg.loc[("2020", "A"), ("return", "std")] == pytest.approx(
            np.sqrt(0.0002)
        )
        assert agg.loc[("2020", "A"), ("return", "count")] == 2
    elif name == "compare_selection_metrics":
        picks = pd.read_csv(out / "selected_checkpoints.csv")
        assert picks.loc[picks.label == "final_value", "timesteps"].eq(25000).all()
        assert picks.loc[picks.label == "sharpe", "timesteps"].eq(50000).all()
        assert (
            pd.read_csv(out / "training_identical.csv")
            .max_abs_diff_valid_final_value.eq(0)
            .all()
        )
        table = pd.read_csv(out / "compare_table.csv")
        changed = table[(table.label == "sharpe") & table.group.isin(["A", "B50"])]
        np.testing.assert_allclose(changed.diff_vs_final_value_return, 0.05)
    elif name == "compare_checkpoint_runs":
        table = pd.read_csv(out / "picked_table.csv")
        for rule, steps in [
            ("pick_final_value", (25000, 26624)),
            ("pick_sharpe", (50000, 53248)),
            ("last_step", (75000, 79872)),
        ]:
            rows = table[table.rule == rule]
            assert rows.timesteps.eq(steps[0]).all()
            assert rows.actual_timesteps.eq(steps[1]).all()
        assert set(table.period) == {"2020", "2022", "2025", "2020-2025", "gfc"}
    else:
        table = pd.read_csv(out / "picked_steps.csv")
        assert table.valid_final_value_step.eq(25000).all()
        assert table.valid_final_value_actual_step.eq(26624).all()
        assert table.valid_sharpe_step.eq(50000).all()
        assert table.valid_sharpe_actual_step.eq(53248).all()
        history = rewards.load(inputs / "legacy")[("A", 0)]
        np.testing.assert_allclose(history.train_real, [2, 6, 10])
    assert snapshot(inputs) == original


def test_notebook_alias_and_root_resolution(monkeypatch):
    import ast

    canonical = ROOT / "analysis/scenario_mix/scenario_mix_v1_summary.ipynb"
    old = ROOT / "analysis/scenario_mix_v1_summary.ipynb"
    assert old.is_symlink()
    assert not Path(os.readlink(old)).is_absolute()
    assert old.resolve() == canonical
    assert old.read_bytes() == canonical.read_bytes()
    notebook = json.loads(canonical.read_text())
    assert any(cell.get("outputs") for cell in notebook["cells"])
    setup = ast.parse("".join(notebook["cells"][2]["source"]))
    root_assignment = next(
        n
        for n in setup.body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "ROOT"
    )
    for cwd in [ROOT, old.parent, canonical.parent]:
        monkeypatch.chdir(cwd)
        namespace = {"Path": Path}
        exec(
            compile(
                ast.Module(body=[root_assignment], type_ignores=[]),
                str(canonical),
                "exec",
            ),
            namespace,
        )
        assert namespace["ROOT"] == ROOT


def test_notebook_full_sequential_smoke_in_isolated_workspace(tmp_path, monkeypatch):
    from IPython.core.interactiveshell import InteractiveShell
    import matplotlib.pyplot as plt

    project = tmp_path / "project"
    marker = project / "finrl/experiments/scenario_mix.py"
    marker.parent.mkdir(parents=True)
    marker.write_text("# Project root marker for the isolated notebook smoke test.\n")
    cwd = project / "analysis/scenario_mix"
    cwd.mkdir(parents=True)
    first_round = project / "results/scenario_mix/v1"
    write_summary_runs(first_round)
    expected_periods = {"2020", "2021", "2022", "2020-2022"}
    for run in first_round.iterdir():
        if not run.is_dir():
            continue
        table = pd.read_csv(run / "summary.csv")
        table[table.period.isin(expected_periods)].to_csv(
            run / "summary.csv", index=False
        )
        for daily in run.glob("*_test_daily.csv"):
            table = pd.read_csv(daily)
            table[table.date <= "2022-12-31"].to_csv(daily, index=False)
    # The real v1 dataset mixes legacy logs with newer valid_score logs.
    scored_path = first_round / "B50_seed0/B50_seed0_valid.csv"
    scored = pd.read_csv(scored_path)
    scored["valid_score"] = [0.5, 0.7, 0.7]
    scored.to_csv(scored_path, index=False)
    # A separate 2025 dataset must never supply the first-round tables or plots.
    write_summary_runs(project / "results/scenario_mix/v1_test2025", offset=99.0)
    # Exercise the notebook's existing recorded-data fallback without TimeDiff.
    monkeypatch.setenv("TIMEDIFF_ROOT", str(tmp_path / "missing_timediff"))
    monkeypatch.chdir(cwd)
    path = ROOT / "analysis/scenario_mix/scenario_mix_v1_summary.ipynb"
    before = path.read_bytes()
    notebook = json.loads(before)
    shell = InteractiveShell()
    executed = 0
    try:
        for i, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            result = shell.run_cell("".join(cell["source"]), store_history=False)
            assert (
                result.success
            ), f"cell {i}: {result.error_before_exec or result.error_in_exec}"
            executed += 1
        assert executed == sum(
            cell["cell_type"] == "code" for cell in notebook["cells"]
        )
        assert shell.user_ns["ROOT"] == project
        assert len(shell.user_ns["RUNS"]) == 4
        assert all(run.parent == first_round for run in shell.user_ns["RUNS"])
        assert set(shell.user_ns["S"].period) == expected_periods
        assert shell.user_ns["agg"].loc[
            ("2020-2022", "A"), ("return", "mean")
        ] == pytest.approx(0.21)
        assert shell.user_ns["D"].date.max() <= pd.Timestamp("2022-12-31")
        assert shell.user_ns["BH"].date.max() <= pd.Timestamp("2022-12-31")
        selected = shell.user_ns["best"].set_index(["group", "seed"])
        assert selected.loc[("A", 0), "timesteps"] == 25000
        assert selected.loc[("B50", 0), "timesteps"] == 50000
        export = project / "analysis/scenario_mix_v1_export"
        for name in [
            "calibration_grid.csv",
            "crash_exposure_comparison.csv",
            "finrl_by_group.csv",
            "finrl_stock_weight_by_run.csv",
            "calibration.png",
            "finrl_value_and_weight.png",
            "finrl_covid.png",
            "finrl_validation.png",
        ]:
            assert (export / name).stat().st_size > 0
        assert path.read_bytes() == before
    finally:
        plt.close("all")


@pytest.mark.parametrize(
    "bad_periods",
    [
        ["2020", "2021", "2022"],
        ["2020", "2021", "2022", "2023", "2024", "2025", "2020-2025"],
    ],
)
def test_notebook_rejects_inconsistent_periods_before_aggregation(
    tmp_path, bad_periods
):
    path = ROOT / "analysis/scenario_mix/scenario_mix_v1_summary.ipynb"
    code = "".join(json.loads(path.read_text())["cells"][21]["source"])
    runs = []
    for name, periods in [
        ("A_seed0", ["2020", "2021", "2022", "2020-2022"]),
        ("B50_seed0", bad_periods),
    ]:
        run = tmp_path / name
        run.mkdir()
        pd.DataFrame({"period": periods}).to_csv(run / "summary.csv", index=False)
        runs.append(run)
    with pytest.raises(ValueError, match="B50_seed0/summary.csv: 期間標籤不一致"):
        exec(compile(code, str(path), "exec"), {"RUNS": runs, "pd": pd})
