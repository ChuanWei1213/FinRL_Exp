import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from finrl.experiments.stock_intrinsic import aggregate_equity_curves
from finrl.experiments.stock_intrinsic import aggregate_evaluation_metrics
from finrl.experiments.stock_intrinsic import aggregate_intrinsic_diagnostics
from finrl.experiments.stock_intrinsic import buy_and_hold_rollout
from finrl.experiments.stock_intrinsic import plot_evaluation_summary
from finrl.experiments.stock_intrinsic import plot_intrinsic_diagnostics
from finrl.experiments.stock_intrinsic import plot_learning_curves_by_variant
from finrl.experiments.stock_intrinsic import plot_period_equity_comparison
from finrl.experiments.stock_intrinsic import plot_sharpe_seed_selection
from finrl.experiments.stock_intrinsic import plot_test_metric_comparison
from finrl.experiments.stock_intrinsic import plot_test_portfolio_weights
from finrl.experiments.stock_intrinsic import VariantRun


def test_intrinsic_diagnostics_are_aggregated_across_seeds():
    intrinsic_log = pd.DataFrame(
        {
            "variant": ["surprise"] * 4,
            "seed": [0, 0, 1, 1],
            "timesteps": [16, 32, 16, 32],
            "reward_intrinsic": [1.0, 2.0, 3.0, 4.0],
            "intrinsic_eta": [0.2, 0.1, 0.4, 0.3],
        }
    )

    curve, summary = aggregate_intrinsic_diagnostics(intrinsic_log)

    first_step = curve.loc[curve["timesteps"] == 16].iloc[0]
    assert first_step["reward_intrinsic_mean"] == 2.0
    assert np.isclose(first_step["reward_intrinsic_std"], np.sqrt(2.0))
    assert summary.loc[0, "seed_count"] == 2
    assert summary.loc[0, "reward_intrinsic_mean"] == 2.5

    figure, _, plotted_summary = plot_intrinsic_diagnostics(intrinsic_log)
    assert plotted_summary.loc[0, "seed_count"] == 2
    assert len(figure.axes[-1].tables) == 1
    plt.close(figure)


def test_evaluation_plot_uses_model_mean_and_standard_deviation():
    dates = pd.date_range("2025-01-02", periods=3, freq="D")
    evaluation_results = {
        "baseline__seed_0": {
            "dates": dates,
            "values": np.array([100.0, 110.0, 120.0]),
        },
        "baseline__seed_1": {
            "dates": dates,
            "values": np.array([100.0, 90.0, 100.0]),
        },
    }
    summary = pd.DataFrame(
        {
            "variant": ["baseline", "baseline"],
            "seed": [0, 1],
            "final_value": [120.0, 100.0],
            "cumulative_return": [0.2, 0.0],
            "cagr": [0.2, 0.0],
            "sharpe": [1.0, 0.0],
            "annualized_volatility": [0.3, 0.1],
            "max_drawdown": [-0.1, -0.2],
        }
    )

    long_frame, curve = aggregate_equity_curves(evaluation_results)
    metrics = aggregate_evaluation_metrics(summary)

    assert len(long_frame) == 6
    assert curve.loc[curve["date"] == dates[1], "mean"].iloc[0] == 100.0
    assert metrics.loc[0, "seed_count"] == 2
    assert metrics.loc[0, "sharpe_mean"] == 0.5

    figure, _, plotted_curve, plotted_metrics = plot_evaluation_summary(
        summary, evaluation_results
    )
    assert plotted_curve["seed_count"].eq(2).all()
    assert plotted_metrics.loc[0, "final_value_mean"] == 110.0
    assert len(figure.axes[1].tables) == 1
    plt.close(figure)


def test_buy_and_hold_invests_equal_capital_once():
    frame = pd.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-02", "2025-01-03", "2025-01-03"],
            "tic": ["AAA", "BBB", "AAA", "BBB"],
            "close": [10.0, 10.0, 12.0, 10.0],
        }
    )
    result = buy_and_hold_rollout(
        frame,
        {
            "initial_amount": 100.0,
            "buy_cost_pct": [0.0, 0.0],
            "reward_scaling": 1.0,
        },
    )

    assert np.allclose(result["values"], [100.0, 110.0])
    assert np.isclose(result["metrics"]["cumulative_return"], 0.1)
    assert result["weight_labels"] == ["Cash", "AAA", "BBB"]
    assert np.allclose(result["weights"].sum(axis=1), 1.0)


def test_original_experiment_style_figures_include_tables_and_benchmark():
    dates = pd.date_range("2025-01-02", periods=3, freq="D")
    summary = pd.DataFrame(
        {
            "variant": ["baseline", "baseline", "surprise", "surprise"],
            "seed": [0, 1, 0, 1],
            "final_value": [110.0, 120.0, 115.0, 125.0],
            "cumulative_return": [0.10, 0.20, 0.15, 0.25],
            "cagr": [0.10, 0.20, 0.15, 0.25],
            "sharpe": [0.7, 0.9, 0.8, 1.0],
            "annualized_volatility": [0.3, 0.4, 0.3, 0.4],
            "max_drawdown": [-0.2, -0.1, -0.15, -0.1],
            "turnover": [0.05, 0.06, 0.04, 0.05],
        }
    )
    evaluation_results = {}
    for variant_index, variant in enumerate(("baseline", "surprise")):
        for seed in (0, 1):
            values = np.array([100.0, 103.0, 110.0 + 5 * variant_index + 10 * seed])
            evaluation_results[f"{variant}__seed_{seed}"] = {
                "dates": dates,
                "values": values,
                "weights": np.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.2, 0.4, 0.4],
                        [0.1, 0.45, 0.45],
                    ]
                ),
                "weight_labels": ["Cash", "AAA", "BBB"],
            }
    benchmark = {
        "dates": dates,
        "values": np.array([100.0, 102.0, 112.0]),
        "metrics": {
            "cumulative_return": 0.12,
            "sharpe": 0.75,
            "max_drawdown": -0.08,
            "turnover": 0.01,
        },
    }

    sharpe_figure, representatives = plot_sharpe_seed_selection(summary)
    assert set(representatives["variant"]) == {"baseline", "surprise"}
    assert len(sharpe_figure.axes[1].tables) == 1
    plt.close(sharpe_figure)

    metric_figure, metric_summary = plot_test_metric_comparison(
        summary, pd.Series(benchmark["metrics"])
    )
    assert metric_summary["seed_count"].eq(2).all()
    assert len(metric_figure.axes[-1].tables) == 1
    plt.close(metric_figure)

    equity_figure, _, ranking = plot_period_equity_comparison(
        "test", evaluation_results, benchmark, "Test"
    )
    assert "Buy & Hold" in set(ranking["model"])
    assert len(equity_figure.axes[1].tables) == 1
    plt.close(equity_figure)

    weight_figure, mean_weights = plot_test_portfolio_weights(evaluation_results)
    assert set(mean_weights["variant"]) == {"baseline", "surprise"}
    assert len(weight_figure.axes[-1].tables) == 1
    plt.close(weight_figure)


def test_learning_curves_are_aggregated_by_variant():
    training_columns = {
        "timesteps": [16, 32],
        "reward_train": [0.1, 0.2],
        "reward_ext": [0.1, 0.1],
        "reward_intrinsic": [0.0, 0.1],
        "reward_surprise": [0.0, 0.1],
        "reward_dejavu": [0.0, 0.0],
        "intrinsic_eta": [0.0, 0.5],
        "intrinsic_volatility_factor": [0.0, 0.4],
        "intrinsic_surprise_loss": [0.0, 0.2],
        "intrinsic_dejavu_loss": [0.0, 0.0],
    }
    runs = [
        VariantRun(
            variant="surprise",
            seed=seed,
            model=None,
            environment=None,
            intrinsic_log=pd.DataFrame(training_columns),
            validation_log=pd.DataFrame(
                {
                    "timesteps": [16, 32],
                    "train_reward_mean": [0.2, 0.3],
                    "real_valid_reward": [0.1, 0.2],
                    "real_valid_turnover": [0.05, 0.04],
                    "real_valid_max_weight": [0.8, 0.7],
                    "sharpe": [0.5 + seed, 0.6 + seed],
                    "cumulative_return": [0.1, 0.2],
                }
            ),
            best_timestep=32,
        )
        for seed in (0, 1)
    ]

    figures = plot_learning_curves_by_variant(runs)

    assert set(figures) == {"surprise"}
    assert sum(len(axis.tables) for axis in figures["surprise"].axes) == 1
    plt.close(figures["surprise"])
