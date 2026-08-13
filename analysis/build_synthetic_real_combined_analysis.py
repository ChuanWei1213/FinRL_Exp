import json
from pathlib import Path

import nbformat as nbf
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "results/ppo_synthetic_vs_real/full/b5a9990cf90b"
OUTPUT_PATH = PROJECT_ROOT / "analysis/synthetic_real_combined_analysis.ipynb"
ARTIFACT_PATH = PROJECT_ROOT / "analysis/synthetic_real_combined_artifact.json"


def main() -> None:
    test = pd.read_csv(RESULT_DIR / "test_metrics.csv")
    benchmark = pd.read_csv(RESULT_DIR / "buy_and_hold_metrics.csv")

    metrics = [
        "cumulative_return",
        "sharpe",
        "annualized_volatility",
        "max_drawdown",
        "turnover",
    ]
    model_names = sorted(test["model_name"].dropna().unique())
    paired_rows = []
    for fee in ["no_fee", "with_fee"]:
        fee_rows = test.loc[test["commission_name"].eq(fee)].set_index(
            ["group", "seed"]
        )
        real = fee_rows.xs("real_trained", level="group")
        for model_name in model_names:
            combined = fee_rows.xs(
                f"real_synthetic::{model_name}", level="group"
            )
            synthetic = fee_rows.xs(f"synthetic::{model_name}", level="group")
            for baseline_name, baseline in [
                ("real", real),
                ("synthetic", synthetic),
            ]:
                deltas = combined[metrics] - baseline[metrics]
                for seed, row in deltas.iterrows():
                    paired_rows.append(
                        {
                            "commission_name": fee,
                            "model_name": model_name,
                            "baseline": baseline_name,
                            "seed": int(seed),
                            **{f"delta_{metric}": row[metric] for metric in metrics},
                        }
                    )
    paired = pd.DataFrame(paired_rows)

    def overall(baseline: str, metric: str) -> tuple[float, int]:
        values = paired.loc[
            paired["baseline"].eq(baseline), f"delta_{metric}"
        ]
        return float(values.mean()), int((values > 0).sum())

    return_vs_real, return_wins_real = overall("real", "cumulative_return")
    sharpe_vs_real, sharpe_wins_real = overall("real", "sharpe")
    return_vs_synthetic, return_wins_synthetic = overall(
        "synthetic", "cumulative_return"
    )
    sharpe_vs_synthetic, sharpe_wins_synthetic = overall("synthetic", "sharpe")

    aggregate = test.groupby(["commission_name", "agent"])[metrics].agg(
        ["mean", "std"]
    )
    no_fee_combined = (
        aggregate.loc["no_fee"]
        .loc[lambda frame: frame.index.str.startswith("Real +")]
        .sort_values(("sharpe", "mean"), ascending=False)
        .iloc[0]
    )
    no_fee_name = (
        aggregate.loc["no_fee"]
        .loc[lambda frame: frame.index.str.startswith("Real +")]
        .sort_values(("sharpe", "mean"), ascending=False)
        .index[0]
    )
    with_fee_combined = (
        aggregate.loc["with_fee"]
        .loc[lambda frame: frame.index.str.startswith("Real +")]
        .sort_values(("sharpe", "mean"), ascending=False)
        .iloc[0]
    )
    with_fee_name = (
        aggregate.loc["with_fee"]
        .loc[lambda frame: frame.index.str.startswith("Real +")]
        .sort_values(("sharpe", "mean"), ascending=False)
        .index[0]
    )
    no_fee_benchmark = benchmark.query(
        "period == 'test' and commission_name == 'no_fee'"
    ).iloc[0]
    with_fee_benchmark = benchmark.query(
        "period == 'test' and commission_name == 'with_fee'"
    ).iloc[0]

    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    }

    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Synthetic vs Real vs Real + Synthetic：2025 真實資料測試分析\n\n"
            "## tl;dr\n\n"
            f"- **加入 synthetic 對 real-only 有明顯的描述性改善。**跨 5 個生成模型、2 種費率與 3 個 seeds，"
            f"real+synthetic 相對 real-only 的平均測試報酬增加 **{return_vs_real:.2%}**，Sharpe 增加 "
            f"**{sharpe_vs_real:.3f}**；30 個 seed-level 配對中，報酬有 **{return_wins_real}/30** 次較高，"
            f"Sharpe 有 **{sharpe_wins_real}/30** 次較高。\n"
            f"- **但 combined 並沒有穩定超越 synthetic-only。**相對 synthetic-only，平均報酬只增加 "
            f"**{return_vs_synthetic:.2%}**、Sharpe 增加 **{sharpe_vs_synthetic:.3f}**；30 個配對中僅 "
            f"**{return_wins_synthetic}/30** 次報酬較高、**{sharpe_wins_synthetic}/30** 次 Sharpe 較高。"
            "這比較像 real data 提供正則化與風險控制，而不是全面提升報酬。\n"
            f"- **無交易成本時，最佳 combined 是 {no_fee_name}。**平均報酬 "
            f"**{no_fee_combined[('cumulative_return', 'mean')]:.2%}**、Sharpe "
            f"**{no_fee_combined[('sharpe', 'mean')]:.3f}**、最大回撤 "
            f"**{no_fee_combined[('max_drawdown', 'mean')]:.2%}**。它的 Sharpe 高於 Buy & Hold "
            f"({no_fee_benchmark['sharpe']:.3f})，但報酬低於 Buy & Hold ({no_fee_benchmark['cumulative_return']:.2%})。\n"
            f"- **含 0.25% 交易成本時，最佳 combined 是 {with_fee_name}。**平均報酬 "
            f"**{with_fee_combined[('cumulative_return', 'mean')]:.2%}**、Sharpe "
            f"**{with_fee_combined[('sharpe', 'mean')]:.3f}**、最大回撤 "
            f"**{with_fee_combined[('max_drawdown', 'mean')]:.2%}**。Sharpe 接近但略低於 Buy & Hold "
            f"({with_fee_benchmark['sharpe']:.3f})，報酬也較低 ({with_fee_benchmark['cumulative_return']:.2%})。\n"
            "- **結論屬於描述性而非統計定論。**每個條件只有 3 個 seeds，且所有模型共用單一 2025 真實測試年度；"
            "模型選擇若依此測試集進行，會有多重比較與 test-set overfitting 風險。"
        ),
        nbf.v4.new_markdown_cell(
            "## Context & Methods\n\n"
            "### Key Assumptions\n\n"
            "- 使用最新且包含 real+synthetic 的完整結果集 `full/b5a9990cf90b`。\n"
            "- 訓練期為 2024-01-01 至 2024-10-01，checkpoint 以共同的 real 2024 validation 選擇；"
            "最後在未參與訓練與選點的 real 2025 測試。\n"
            "- real+synthetic 並非把資料列直接合併，而是每個 episode 以 50/50 機率抽 real 或該生成器的 synthetic path。\n"
            "- 比較以相同 seed、相同 fee 配對；主要指標為累積報酬、Sharpe、年化波動、最大回撤與 turnover。\n"
            "- 所有結果皆為 PPO checkpoint 的 deterministic rollout。"
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from IPython.display import display\n\n"
            "def find_project_root(start: Path) -> Path:\n"
            "    for candidate in (start.resolve(), *start.resolve().parents):\n"
            "        if (candidate / 'finrl').is_dir() and (candidate / 'results').is_dir():\n"
            "            return candidate\n"
            "    raise RuntimeError('FinRL project root not found')\n\n"
            "PROJECT_ROOT = find_project_root(Path.cwd())\n"
            "RESULT_DIR = PROJECT_ROOT / 'results/ppo_synthetic_vs_real/full/b5a9990cf90b'\n"
            "TEST_PATH = RESULT_DIR / 'test_metrics.csv'\n"
            "PERIOD_PATH = RESULT_DIR / 'ppo_period_metrics.csv'\n"
            "BENCHMARK_PATH = RESULT_DIR / 'buy_and_hold_metrics.csv'\n"
            "print('Result set:', RESULT_DIR.relative_to(PROJECT_ROOT))"
        ),
        nbf.v4.new_markdown_cell("## Data\n\n### 1. Load and validate the formal result set"),
        nbf.v4.new_code_cell(
            "test = pd.read_csv(TEST_PATH)\n"
            "period = pd.read_csv(PERIOD_PATH)\n"
            "benchmark = pd.read_csv(BENCHMARK_PATH)\n\n"
            "METRICS = ['cumulative_return', 'sharpe', 'annualized_volatility', 'max_drawdown', 'turnover']\n"
            "assert len(test) == 66, f'Expected 66 test rows, got {len(test)}'\n"
            "assert test.duplicated(['group', 'commission_name', 'seed']).sum() == 0\n"
            "assert test[METRICS].notna().all().all()\n"
            "counts = test.groupby(['commission_name', 'group']).size()\n"
            "assert counts.eq(3).all(), counts[counts.ne(3)]\n"
            "assert set(test['seed']) == {0, 1, 2}\n"
            "quality_summary = pd.DataFrame({\n"
            "    'check': ['test rows', 'unique fee × group cells', 'seeds per cell', 'duplicate keys', 'missing primary metrics'],\n"
            "    'result': [len(test), len(counts), f'{counts.min()}–{counts.max()}', int(test.duplicated(['group','commission_name','seed']).sum()), int(test[METRICS].isna().sum().sum())],\n"
            "})\n"
            "display(quality_summary)"
        ),
        nbf.v4.new_markdown_cell(
            "## Key findings with visual evidence\n\n"
            "### 2. Build paired combined-minus-baseline comparisons\n\n"
            "A positive delta is favorable for return, Sharpe, and max drawdown (less negative drawdown). "
            "A negative delta is favorable for volatility and turnover."
        ),
        nbf.v4.new_code_cell(
            "MODEL_LABELS = {\n"
            "    'armd__single_path': 'ARMD (single path)',\n"
            "    'cndiff__50_paths': 'CN-Diff (50 paths)',\n"
            "    'dva__50_paths': 'DVA (50 paths)',\n"
            "    'dva__mean_of_50_paths': 'DVA (mean of 50 paths)',\n"
            "    'nsdiff__50_paths': 'NS-Diff (50 paths)',\n"
            "}\n"
            "paired_rows = []\n"
            "for fee in ['no_fee', 'with_fee']:\n"
            "    fee_rows = test.loc[test['commission_name'].eq(fee)].set_index(['group', 'seed'])\n"
            "    real = fee_rows.xs('real_trained', level='group')\n"
            "    for model_name, model_label in MODEL_LABELS.items():\n"
            "        combined = fee_rows.xs(f'real_synthetic::{model_name}', level='group')\n"
            "        synthetic = fee_rows.xs(f'synthetic::{model_name}', level='group')\n"
            "        for baseline_name, baseline in [('real', real), ('synthetic', synthetic)]:\n"
            "            deltas = combined[METRICS] - baseline[METRICS]\n"
            "            for seed, row in deltas.iterrows():\n"
            "                paired_rows.append({\n"
            "                    'commission_name': fee, 'model_name': model_name,\n"
            "                    'model_label': model_label, 'baseline': baseline_name,\n"
            "                    'seed': int(seed),\n"
            "                    **{f'delta_{metric}': row[metric] for metric in METRICS},\n"
            "                })\n"
            "paired = pd.DataFrame(paired_rows)\n"
            "paired.head()"
        ),
        nbf.v4.new_code_cell(
            "favorable_direction = {\n"
            "    'cumulative_return': 1, 'sharpe': 1, 'annualized_volatility': -1,\n"
            "    'max_drawdown': 1, 'turnover': -1,\n"
            "}\n"
            "overall_rows = []\n"
            "for baseline in ['real', 'synthetic']:\n"
            "    subset = paired.loc[paired['baseline'].eq(baseline)]\n"
            "    for metric in METRICS:\n"
            "        values = subset[f'delta_{metric}']\n"
            "        direction = favorable_direction[metric]\n"
            "        overall_rows.append({\n"
            "            'baseline': baseline, 'metric': metric,\n"
            "            'mean_delta': values.mean(),\n"
            "            'favorable_seed_pairs': int((direction * values > 0).sum()),\n"
            "            'total_seed_pairs': len(values),\n"
            "        })\n"
            "overall_summary = pd.DataFrame(overall_rows)\n"
            "display(overall_summary)"
        ),
        nbf.v4.new_markdown_cell(
            "**Interpretation.** Combined training consistently improves on real-only across most seed-level comparisons, "
            "especially on risk-adjusted performance and drawdown. Against synthetic-only, however, wins are close to a coin flip. "
            "The added real episodes therefore look more like a stabilizer than a universal performance boost."
        ),
        nbf.v4.new_code_cell(
            "condition = (\n"
            "    paired.groupby(['commission_name', 'model_label', 'baseline'])\n"
            "    .agg(\n"
            "        sharpe_delta_mean=('delta_sharpe', 'mean'),\n"
            "        sharpe_delta_std=('delta_sharpe', 'std'),\n"
            "        return_delta_mean=('delta_cumulative_return', 'mean'),\n"
            "        volatility_delta_mean=('delta_annualized_volatility', 'mean'),\n"
            "        drawdown_delta_mean=('delta_max_drawdown', 'mean'),\n"
            "        turnover_delta_mean=('delta_turnover', 'mean'),\n"
            "        sharpe_wins=('delta_sharpe', lambda values: int((values > 0).sum())),\n"
            "    )\n"
            "    .reset_index()\n"
            ")\n"
            "display(condition.sort_values(['commission_name', 'baseline', 'sharpe_delta_mean'], ascending=[True, True, False]))"
        ),
        nbf.v4.new_markdown_cell(
            "### 3. Sharpe deltas reveal where combined training helps\n\n"
            "Bars show the mean paired Sharpe change across three seeds; error bars are ±1 sample standard deviation, "
            "not confidence intervals."
        ),
        nbf.v4.new_code_cell(
            "palette = {'real': '#0057B8', 'synthetic': '#E66100'}\n"
            "fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)\n"
            "model_order = list(MODEL_LABELS.values())\n"
            "y = np.arange(len(model_order))\n"
            "offsets = {'real': -0.18, 'synthetic': 0.18}\n"
            "for axis, fee in zip(axes, ['no_fee', 'with_fee']):\n"
            "    fee_rows = condition.loc[condition['commission_name'].eq(fee)]\n"
            "    for baseline in ['real', 'synthetic']:\n"
            "        rows = fee_rows.loc[fee_rows['baseline'].eq(baseline)].set_index('model_label').reindex(model_order)\n"
            "        axis.barh(\n"
            "            y + offsets[baseline], rows['sharpe_delta_mean'], height=0.32,\n"
            "            xerr=rows['sharpe_delta_std'], color=palette[baseline], alpha=0.85,\n"
            "            label=f'Combined − {baseline}', capsize=3,\n"
            "        )\n"
            "    axis.axvline(0, color='#374151', linewidth=1)\n"
            "    axis.set_title('No fee' if fee == 'no_fee' else '0.25% transaction fee')\n"
            "    axis.set_xlabel('Mean paired test Sharpe delta (±1 SD)')\n"
            "    axis.grid(axis='x', alpha=0.2)\n"
            "axes[0].set_yticks(y, model_order)\n"
            "axes[0].invert_yaxis()\n"
            "axes[0].legend(loc='lower right')\n"
            "fig.suptitle('Real + synthetic training: paired Sharpe change on real 2025 test')\n"
            "fig.tight_layout()\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "**Interpretation.** ARMD is the clearest case where adding real episodes raises Sharpe relative to synthetic-only, "
            "while also lowering volatility. For DVA and NS-Diff, combined-vs-synthetic effects are mixed and often reverse by fee setting."
        ),
        nbf.v4.new_markdown_cell("### 4. Compare aggregate strategies with the real-only control and Buy & Hold"),
        nbf.v4.new_code_cell(
            "aggregate = (\n"
            "    test.groupby(['commission_name', 'agent'])[METRICS]\n"
            "    .agg(['mean', 'std'])\n"
            ")\n"
            "benchmark_test = benchmark.loc[benchmark['period'].eq('test')].copy()\n"
            "comparison_rows = []\n"
            "for fee in ['no_fee', 'with_fee']:\n"
            "    fee_agg = aggregate.loc[fee].copy()\n"
            "    combined = fee_agg.loc[fee_agg.index.str.startswith('Real +')].copy()\n"
            "    combined['strategy'] = combined.index\n"
            "    for _, row in combined.iterrows():\n"
            "        comparison_rows.append({\n"
            "            'fee': fee, 'strategy': row['strategy'], 'type': 'real + synthetic',\n"
            "            'return_mean': row[('cumulative_return', 'mean')],\n"
            "            'return_std': row[('cumulative_return', 'std')],\n"
            "            'sharpe_mean': row[('sharpe', 'mean')],\n"
            "            'sharpe_std': row[('sharpe', 'std')],\n"
            "            'max_drawdown_mean': row[('max_drawdown', 'mean')],\n"
            "        })\n"
            "    real = fee_agg.loc['Real data (2024)']\n"
            "    comparison_rows.append({\n"
            "        'fee': fee, 'strategy': 'Real data (2024)', 'type': 'real-only',\n"
            "        'return_mean': real[('cumulative_return','mean')], 'return_std': real[('cumulative_return','std')],\n"
            "        'sharpe_mean': real[('sharpe','mean')], 'sharpe_std': real[('sharpe','std')],\n"
            "        'max_drawdown_mean': real[('max_drawdown','mean')],\n"
            "    })\n"
            "    bh = benchmark_test.loc[benchmark_test['commission_name'].eq(fee)].iloc[0]\n"
            "    comparison_rows.append({\n"
            "        'fee': fee, 'strategy': 'Buy & Hold', 'type': 'benchmark',\n"
            "        'return_mean': bh['cumulative_return'], 'return_std': np.nan,\n"
            "        'sharpe_mean': bh['sharpe'], 'sharpe_std': np.nan,\n"
            "        'max_drawdown_mean': bh['max_drawdown'],\n"
            "    })\n"
            "comparison = pd.DataFrame(comparison_rows)\n"
            "display(comparison.sort_values(['fee', 'sharpe_mean'], ascending=[True, False]))"
        ),
        nbf.v4.new_markdown_cell(
            "**Interpretation.** The best combined policies narrow the gap to Buy & Hold by improving drawdown and Sharpe, "
            "but they generally give up total return. No-fee Real+ARMD slightly exceeds Buy & Hold on Sharpe; under fees, "
            "Real+CN-Diff is close but does not exceed it."
        ),
        nbf.v4.new_markdown_cell("## Robustness, limitations, and validation details\n\n### 5. Fee sensitivity"),
        nbf.v4.new_code_cell(
            "fee_pivot = test.pivot(index=['group', 'seed'], columns='commission_name', values=METRICS)\n"
            "fee_delta = fee_pivot.xs('with_fee', axis=1, level=1) - fee_pivot.xs('no_fee', axis=1, level=1)\n"
            "fee_delta['strategy_family'] = np.select(\n"
            "    [fee_delta.index.get_level_values('group').str.startswith('real_synthetic::'),\n"
            "     fee_delta.index.get_level_values('group').str.startswith('synthetic::')],\n"
            "    ['real + synthetic', 'synthetic-only'], default='real-only',\n"
            ")\n"
            "fee_summary = fee_delta.groupby('strategy_family')[METRICS].mean().reset_index()\n"
            "display(fee_summary)"
        ),
        nbf.v4.new_markdown_cell(
            "Combined policies show a smaller average degradation from no-fee to with-fee runs than the other families. "
            "This is encouraging, but it is not a pure transaction-cost subtraction: each fee condition trains a separate policy, "
            "so the delta includes policy adaptation and seed variation."
        ),
        nbf.v4.new_markdown_cell(
            "### 6. What the design supports—and what it does not\n\n"
            "- **Strengths:** common real validation for checkpoint selection, untouched real 2025 test, same PPO architecture/hyperparameters, "
            "paired seeds, deterministic rollout, and both no-fee/with-fee scenarios.\n"
            "- **Small sample:** only 3 seeds per condition. Standard deviations are descriptive; formal significance tests would have very low power.\n"
            "- **Single market regime:** all conclusions depend on one 2025 test year and five large-cap US stocks.\n"
            "- **Multiple comparisons:** choosing a winner among 10 synthetic/combined variants using the same test year risks test-set overfitting.\n"
            "- **Scaler confound:** combined and synthetic-only use the generator-specific scaler, while real-only uses the real-data scaler. "
            "Combined-vs-synthetic isolates added real episodes reasonably well; combined-vs-real also changes the scaler.\n"
            "- **Post-hoc representative seeds:** median-test-Sharpe representatives are useful plots, not an unbiased selection rule. "
            "This notebook bases conclusions on all-seed aggregates and paired deltas.\n"
            "- **Balanced curriculum:** real+synthetic samples the source 50/50 by episode. This tests a source-balanced curriculum, not arbitrary row-level concatenation ratios."
        ),
        nbf.v4.new_markdown_cell(
            "## Takeaways\n\n"
            "1. **Keep real+synthetic as an augmentation strategy, not as a guaranteed replacement for synthetic-only.** It is reliably better than real-only, "
            "but not reliably better than the corresponding synthetic-only policy.\n"
            "2. **Prioritize ARMD for no-fee/risk-adjusted experiments and CN-Diff for fee-aware robustness.** These are the clearest combined candidates in this run.\n"
            "3. **Do not declare a final winner yet.** Expand to at least 10–20 seeds, multiple rolling test windows, and bootstrap confidence intervals by market period.\n"
            "4. **Run an ablation on the real episode probability** (for example 0%, 25%, 50%, 75%, 100%) and a separate scaler ablation. This will identify whether gains come from real-data exposure, source balance, or normalization.\n"
            "5. **Lock a model choice before the next untouched test window.** Use 2024 validation for tuning, select one or two candidates, then evaluate once on a new holdout period to control test-set selection bias."
        ),
        nbf.v4.new_markdown_cell(
            "## Further questions\n\n"
            "- Are the combined gains concentrated in particular months or volatility regimes within 2025?\n"
            "- Does the 50/50 episode ratio remain optimal when synthetic path count changes from 1 to 50?\n"
            "- Would a scaler fitted jointly on real and synthetic train data improve ARMD without leaking future information?\n"
            "- Do the same rankings hold for broader universes, different asset classes, and rolling out-of-sample years?"
        ),
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT_PATH)
    print(OUTPUT_PATH)

    model_labels = {
        "armd__single_path": "ARMD (single path)",
        "cndiff__50_paths": "CN-Diff (50 paths)",
        "dva__50_paths": "DVA (50 paths)",
        "dva__mean_of_50_paths": "DVA (mean of 50 paths)",
        "nsdiff__50_paths": "NS-Diff (50 paths)",
    }
    paired["model_label"] = paired["model_name"].map(model_labels)
    paired["baseline_label"] = paired["baseline"].map(
        {"real": "Combined − real", "synthetic": "Combined − synthetic"}
    )
    paired_condition = (
        paired.groupby(
            ["commission_name", "model_name", "model_label", "baseline", "baseline_label"]
        )
        .agg(
            sharpe_delta_mean=("delta_sharpe", "mean"),
            sharpe_delta_std=("delta_sharpe", "std"),
            return_delta_mean=("delta_cumulative_return", "mean"),
            volatility_delta_mean=("delta_annualized_volatility", "mean"),
            drawdown_delta_mean=("delta_max_drawdown", "mean"),
            turnover_delta_mean=("delta_turnover", "mean"),
            sharpe_wins=("delta_sharpe", lambda values: int((values > 0).sum())),
            seed_count=("seed", "size"),
        )
        .reset_index()
    )

    def strategy_comparison_rows() -> list[dict]:
        rows = []
        for fee in ["no_fee", "with_fee"]:
            fee_aggregate = aggregate.loc[fee]
            selected = fee_aggregate.loc[
                fee_aggregate.index.str.startswith("Real +")
                | (fee_aggregate.index == "Real data (2024)")
            ]
            for strategy, row in selected.iterrows():
                rows.append(
                    {
                        "fee": fee,
                        "strategy": strategy,
                        "strategy_type": (
                            "real-only" if strategy == "Real data (2024)" else "real + synthetic"
                        ),
                        "return_mean": row[("cumulative_return", "mean")],
                        "return_std": row[("cumulative_return", "std")],
                        "sharpe_mean": row[("sharpe", "mean")],
                        "sharpe_std": row[("sharpe", "std")],
                        "volatility_mean": row[("annualized_volatility", "mean")],
                        "max_drawdown_mean": row[("max_drawdown", "mean")],
                        "turnover_mean": row[("turnover", "mean")],
                        "seed_count": 3,
                    }
                )
            buy_hold = benchmark.query(
                "period == 'test' and commission_name == @fee"
            ).iloc[0]
            rows.append(
                {
                    "fee": fee,
                    "strategy": "Buy & Hold",
                    "strategy_type": "benchmark",
                    "return_mean": buy_hold["cumulative_return"],
                    "return_std": None,
                    "sharpe_mean": buy_hold["sharpe"],
                    "sharpe_std": None,
                    "volatility_mean": buy_hold["annualized_volatility"],
                    "max_drawdown_mean": buy_hold["max_drawdown"],
                    "turnover_mean": buy_hold["turnover"],
                    "seed_count": None,
                }
            )
        return rows

    test_source = {
        "id": "test_metrics_source",
        "label": "Formal full-run PPO test metrics",
        "path": "results/ppo_synthetic_vs_real/full/b5a9990cf90b/test_metrics.csv",
        "query": {
            "engine": "DuckDB",
            "language": "sql",
            "description": "Paired same-seed comparison of real+synthetic against real-only and the corresponding synthetic-only policy on the common real 2025 test period.",
            "sql": """WITH test AS (
  SELECT * FROM read_csv_auto('results/ppo_synthetic_vs_real/full/b5a9990cf90b/test_metrics.csv')
), combined AS (
  SELECT commission_name, seed, model_name,
         cumulative_return, sharpe, annualized_volatility, max_drawdown, turnover
  FROM test WHERE group LIKE 'real_synthetic::%'
), real_control AS (
  SELECT commission_name, seed,
         cumulative_return, sharpe, annualized_volatility, max_drawdown, turnover
  FROM test WHERE group = 'real_trained'
), synthetic_control AS (
  SELECT commission_name, seed, model_name,
         cumulative_return, sharpe, annualized_volatility, max_drawdown, turnover
  FROM test WHERE group LIKE 'synthetic::%'
)
SELECT c.commission_name, c.seed, c.model_name, 'real' AS baseline,
       c.cumulative_return - r.cumulative_return AS delta_cumulative_return,
       c.sharpe - r.sharpe AS delta_sharpe,
       c.annualized_volatility - r.annualized_volatility AS delta_annualized_volatility,
       c.max_drawdown - r.max_drawdown AS delta_max_drawdown,
       c.turnover - r.turnover AS delta_turnover
FROM combined c
JOIN real_control r USING (commission_name, seed)
UNION ALL
SELECT c.commission_name, c.seed, c.model_name, 'synthetic' AS baseline,
       c.cumulative_return - s.cumulative_return,
       c.sharpe - s.sharpe,
       c.annualized_volatility - s.annualized_volatility,
       c.max_drawdown - s.max_drawdown,
       c.turnover - s.turnover
FROM combined c
JOIN synthetic_control s USING (commission_name, seed, model_name)
ORDER BY commission_name, model_name, baseline, seed""",
            "tables_used": [
                "results/ppo_synthetic_vs_real/full/b5a9990cf90b/test_metrics.csv"
            ],
            "filters": [
                "period = test",
                "real test window = [2025-01-01, 2026-01-01)",
                "seeds = 0, 1, 2",
                "fees = 0% and 0.25%",
            ],
            "metric_definitions": [
                "Cumulative return = final portfolio value / initial portfolio value - 1 over the continuous real 2025 test.",
                "Sharpe = annualized mean daily portfolio return divided by annualized daily-return standard deviation.",
                "Maximum drawdown is the minimum peak-to-trough portfolio decline; a less negative value is better.",
                "Mean paired delta averages real+synthetic minus baseline across the 30 model × fee × seed comparisons.",
            ],
        },
    }
    comparison_source = {
        "id": "comparison_source",
        "label": "Formal PPO test metrics and Buy & Hold benchmark",
        "path": "analysis/synthetic_real_combined_analysis.ipynb",
        "query": {
            "engine": "DuckDB",
            "language": "sql",
            "description": "All-seed PPO aggregates combined with the deterministic Buy & Hold test benchmark.",
            "sql": """WITH test AS (
  SELECT * FROM read_csv_auto('results/ppo_synthetic_vs_real/full/b5a9990cf90b/test_metrics.csv')
), benchmark AS (
  SELECT * FROM read_csv_auto('results/ppo_synthetic_vs_real/full/b5a9990cf90b/buy_and_hold_metrics.csv')
), ppo AS (
  SELECT commission_name AS fee, agent AS strategy,
         CASE WHEN group = 'real_trained' THEN 'real-only' ELSE 'real + synthetic' END AS strategy_type,
         avg(cumulative_return) AS return_mean,
         stddev_samp(cumulative_return) AS return_std,
         avg(sharpe) AS sharpe_mean,
         stddev_samp(sharpe) AS sharpe_std,
         avg(annualized_volatility) AS volatility_mean,
         avg(max_drawdown) AS max_drawdown_mean,
         avg(turnover) AS turnover_mean,
         count(*) AS seed_count
  FROM test
  WHERE group = 'real_trained' OR group LIKE 'real_synthetic::%'
  GROUP BY commission_name, agent, group
), buy_hold AS (
  SELECT commission_name AS fee, 'Buy & Hold' AS strategy, 'benchmark' AS strategy_type,
         cumulative_return AS return_mean, NULL::DOUBLE AS return_std,
         sharpe AS sharpe_mean, NULL::DOUBLE AS sharpe_std,
         annualized_volatility AS volatility_mean,
         max_drawdown AS max_drawdown_mean,
         turnover AS turnover_mean, NULL::BIGINT AS seed_count
  FROM benchmark WHERE period = 'test'
)
SELECT * FROM ppo UNION ALL SELECT * FROM buy_hold
ORDER BY fee, sharpe_mean DESC""",
            "tables_used": [
                "results/ppo_synthetic_vs_real/full/b5a9990cf90b/test_metrics.csv",
                "results/ppo_synthetic_vs_real/full/b5a9990cf90b/buy_and_hold_metrics.csv",
            ],
            "filters": [
                "period = test",
                "real test window = [2025-01-01, 2026-01-01)",
            ],
            "metric_definitions": [
                "PPO values are mean and sample standard deviation across three training seeds.",
                "Buy & Hold is deterministic and therefore has no seed standard deviation.",
            ],
        },
    }

    headline_row = {
        "return_delta_vs_real": return_vs_real,
        "return_win_rate_vs_real": return_wins_real / 30,
        "sharpe_delta_vs_real": sharpe_vs_real,
        "sharpe_win_rate_vs_real": sharpe_wins_real / 30,
        "return_delta_vs_synthetic": return_vs_synthetic,
        "return_win_rate_vs_synthetic": return_wins_synthetic / 30,
        "sharpe_delta_vs_synthetic": sharpe_vs_synthetic,
        "sharpe_win_rate_vs_synthetic": sharpe_wins_synthetic / 30,
        "comparison_count": 30,
    }

    generated_at = "2026-08-09T15:30:00Z"
    manifest = {
        "version": 1,
        "surface": "report",
        "title": "Real + Synthetic PPO：加入真實資料後的成效",
        "description": "Technical analysis of real-only, synthetic-only, and source-balanced real+synthetic PPO training on a common real 2025 test.",
        "generatedAt": generated_at,
        "sources": [test_source, comparison_source],
        "cards": [
            {
                "id": "return_vs_real",
                "dataset": "headline_metrics",
                "sourceId": "test_metrics_source",
                "description": "Across 5 generators × 2 fee settings × 3 paired seeds.",
                "metrics": [
                    {"label": "Return delta vs real", "field": "return_delta_vs_real", "format": "percent", "signed": True},
                    {"label": "Favorable pairs", "field": "return_win_rate_vs_real", "format": "percent"},
                ],
            },
            {
                "id": "sharpe_vs_real",
                "dataset": "headline_metrics",
                "sourceId": "test_metrics_source",
                "description": "Risk-adjusted improvement relative to real-only.",
                "metrics": [
                    {"label": "Sharpe delta vs real", "field": "sharpe_delta_vs_real", "format": "number", "signed": True},
                    {"label": "Favorable pairs", "field": "sharpe_win_rate_vs_real", "format": "percent"},
                ],
            },
            {
                "id": "return_vs_synthetic",
                "dataset": "headline_metrics",
                "sourceId": "test_metrics_source",
                "description": "Combined minus the corresponding synthetic-only policy.",
                "metrics": [
                    {"label": "Return delta vs synthetic", "field": "return_delta_vs_synthetic", "format": "percent", "signed": True},
                    {"label": "Favorable pairs", "field": "return_win_rate_vs_synthetic", "format": "percent"},
                ],
            },
            {
                "id": "sharpe_vs_synthetic",
                "dataset": "headline_metrics",
                "sourceId": "test_metrics_source",
                "description": "Combined minus the corresponding synthetic-only policy.",
                "metrics": [
                    {"label": "Sharpe delta vs synthetic", "field": "sharpe_delta_vs_synthetic", "format": "number", "signed": True},
                    {"label": "Favorable pairs", "field": "sharpe_win_rate_vs_synthetic", "format": "percent"},
                ],
            },
        ],
        "charts": [
            {
                "id": "no_fee_sharpe_delta",
                "title": "No-fee paired test Sharpe deltas",
                "subtitle": "Mean of three paired seeds per generator; positive values favor real+synthetic.",
                "showDescription": True,
                "intent": "comparison",
                "question": "Where does real+synthetic improve test Sharpe relative to real-only or synthetic-only without fees?",
                "rationale": "Grouped bars compare the two baselines for each generator while preserving the zero reference.",
                "comparisonContext": {"baseline": "real-only and corresponding synthetic-only", "grain": "generator × baseline", "unit": "Sharpe ratio delta"},
                "type": "bar",
                "dataset": "paired_no_fee",
                "sourceId": "test_metrics_source",
                "encodings": {
                    "x": {"field": "model_label", "type": "nominal", "label": "Synthetic generator"},
                    "y": {"field": "sharpe_delta_mean", "type": "quantitative", "label": "Mean paired Sharpe delta", "format": "number"},
                    "color": {"field": "baseline_label", "type": "nominal", "label": "Comparison"},
                    "tooltip": [
                        {"field": "sharpe_delta_std", "type": "quantitative", "label": "Seed SD", "format": "number"},
                        {"field": "sharpe_wins", "type": "quantitative", "label": "Positive seeds"},
                        {"field": "return_delta_mean", "type": "quantitative", "label": "Mean return delta", "format": "percent"},
                    ],
                },
                "yAxisTitle": "Mean paired Sharpe delta",
                "valueFormat": "number",
                "layout": "full",
                "palette": {"kind": "categorical", "name": "blue-orange"},
                "referenceLines": [{"axis": "y", "value": 0, "label": "No change", "color": "neutral", "lineStyle": "solid"}],
                "legend": {"position": "bottom", "title": "Baseline"},
                "settings": {"groupMode": "grouped", "categoryLabelPolicy": "wrap", "sort": "none", "showValues": True},
                "surface": {"surface": "card", "viewMode": "both"},
            },
            {
                "id": "with_fee_sharpe_delta",
                "title": "0.25% fee paired test Sharpe deltas",
                "subtitle": "Mean of three paired seeds per generator; positive values favor real+synthetic.",
                "showDescription": True,
                "intent": "comparison",
                "question": "Where does real+synthetic improve test Sharpe relative to real-only or synthetic-only with transaction fees?",
                "rationale": "Grouped bars compare the two baselines for each generator under the fee-aware training and test condition.",
                "comparisonContext": {"baseline": "real-only and corresponding synthetic-only", "grain": "generator × baseline", "unit": "Sharpe ratio delta"},
                "type": "bar",
                "dataset": "paired_with_fee",
                "sourceId": "test_metrics_source",
                "encodings": {
                    "x": {"field": "model_label", "type": "nominal", "label": "Synthetic generator"},
                    "y": {"field": "sharpe_delta_mean", "type": "quantitative", "label": "Mean paired Sharpe delta", "format": "number"},
                    "color": {"field": "baseline_label", "type": "nominal", "label": "Comparison"},
                    "tooltip": [
                        {"field": "sharpe_delta_std", "type": "quantitative", "label": "Seed SD", "format": "number"},
                        {"field": "sharpe_wins", "type": "quantitative", "label": "Positive seeds"},
                        {"field": "return_delta_mean", "type": "quantitative", "label": "Mean return delta", "format": "percent"},
                    ],
                },
                "yAxisTitle": "Mean paired Sharpe delta",
                "valueFormat": "number",
                "layout": "full",
                "palette": {"kind": "categorical", "name": "blue-orange"},
                "referenceLines": [{"axis": "y", "value": 0, "label": "No change", "color": "neutral", "lineStyle": "solid"}],
                "legend": {"position": "bottom", "title": "Baseline"},
                "settings": {"groupMode": "grouped", "categoryLabelPolicy": "wrap", "sort": "none", "showValues": True},
                "surface": {"surface": "card", "viewMode": "both"},
            },
        ],
        "tables": [
            {
                "id": "strategy_comparison_table",
                "title": "Real 2025 test strategy comparison",
                "subtitle": "PPO values are mean ± sample SD across three seeds; Buy & Hold is deterministic.",
                "showDescription": True,
                "dataset": "strategy_comparison",
                "defaultSort": {"field": "sharpe_mean", "direction": "desc"},
                "density": "spacious",
                "sourceId": "comparison_source",
                "layout": "full",
                "columns": [
                    {"field": "fee", "label": "Fee", "type": "text"},
                    {"field": "strategy", "label": "Strategy", "type": "text"},
                    {"field": "return_mean", "label": "Mean return", "format": "percent"},
                    {"field": "return_std", "label": "Return SD", "format": "percent"},
                    {"field": "sharpe_mean", "label": "Mean Sharpe", "format": "number"},
                    {"field": "sharpe_std", "label": "Sharpe SD", "format": "number"},
                    {"field": "max_drawdown_mean", "label": "Mean max drawdown", "format": "percent"},
                    {"field": "turnover_mean", "label": "Mean turnover", "format": "number"},
                ],
            }
        ],
        "blocks": [
            {"id": "title", "type": "markdown", "body": "# Real + Synthetic PPO：加入真實資料後的成效", "layout": "full"},
            {
                "id": "technical_summary",
                "type": "markdown",
                "body": (
                    "## Technical summary\n\n"
                    "**加入 synthetic 後，real+synthetic 相對 real-only 有一致的描述性改善，但沒有穩定超越 synthetic-only。** "
                    "跨 5 個生成模型、2 種費率與 3 個配對 seeds，combined 相對 real-only 的平均測試報酬增加 **2.47 個百分點**、Sharpe 增加 **0.105**；"
                    "報酬與 Sharpe 分別在 **23/30**、**21/30** 個 seed-level 配對中較高。相對 synthetic-only，平均報酬只增加 **0.23 個百分點**、Sharpe 增加 **0.017**，"
                    "有利配對僅 **13/30** 與 **14/30**。因此，real data 的主要效果較像穩定器與風險控制，而不是普遍提高 alpha。\n\n"
                    "**最佳 combined 仍未全面勝過 Buy & Hold。**無費率時 Real + ARMD 的平均 Sharpe 為 **1.094**，略高於 Buy & Hold 的 **1.054**，且最大回撤較小；"
                    "但平均報酬 **25.79%** 低於 Buy & Hold 的 **28.30%**。含 0.25% 費率時 Real + CN-Diff 的 Sharpe **1.022** 接近、但略低於 Buy & Hold 的 **1.031**，報酬也較低。"
                ),
                "layout": "full",
            },
            {"id": "headline_metrics_block", "type": "metric-strip", "cardIds": ["return_vs_real", "sharpe_vs_real", "return_vs_synthetic", "sharpe_vs_synthetic"], "layout": "full"},
            {
                "id": "no_fee_finding",
                "type": "markdown",
                "sourceId": "test_metrics_source",
                "body": (
                    "## 無交易成本時，ARMD 是最清楚的 combined 改善\n\n"
                    "Real + ARMD 對 real-only 的平均 Sharpe 增加 **0.200**，3/3 seeds 都改善；對 ARMD synthetic-only 也增加 **0.115**，同樣 3/3 seeds 改善。"
                    "它同時降低波動與回撤，但相對 ARMD synthetic-only 的平均報酬少 **5.25 個百分點**。這是典型的風險調整改善：收益上限下降，路徑更穩。"
                ),
                "layout": "full",
            },
            {"id": "no_fee_chart_block", "type": "chart", "chartId": "no_fee_sharpe_delta", "layout": "full"},
            {
                "id": "with_fee_finding",
                "type": "markdown",
                "sourceId": "test_metrics_source",
                "body": (
                    "## 含交易成本時，combined 對 real-only 更穩，但對 synthetic-only 仍混合\n\n"
                    "Real + CN-Diff 的平均 Sharpe 為 **1.022 ± 0.009**，是 fee-aware combined 中最高且最穩定者；相對 real-only 平均增加 **0.184**。"
                    "不過相對 CN-Diff synthetic-only 只增加 **0.016**，且只有 1/3 seeds 較高。Real + ARMD 對 synthetic-only 的 Sharpe 改善較大（**+0.249**），但 seed 變異也明顯較高。"
                ),
                "layout": "full",
            },
            {"id": "with_fee_chart_block", "type": "chart", "chartId": "with_fee_sharpe_delta", "layout": "full"},
            {
                "id": "benchmark_interpretation",
                "type": "markdown",
                "body": (
                    "## Combined 改善風險調整表現，但總報酬仍落後基準\n\n"
                    "下表保留所有 combined 策略、real-only 與 Buy & Hold 的精確值。主要取捨是：combined 往往降低回撤與 seed 變異，"
                    "但 2025 這個單一市場年度中，Buy & Hold 仍提供較高的總報酬；因此不能只依 Sharpe 排名宣稱投資績效全面勝出。"
                ),
                "layout": "full",
            },
            {"id": "strategy_table_block", "type": "table", "tableId": "strategy_comparison_table", "layout": "full"},
            {
                "id": "scope_definitions",
                "type": "markdown",
                "body": (
                    "## Scope、資料與指標定義\n\n"
                    "- **訓練／驗證／測試：**2024-01-01–2024-10-01 訓練；2024-10-01–2025-01-01 real validation 選 checkpoint；2025 全年 real test。\n"
                    "- **資產：**AAPL、AMZN、GOOGL、MSFT、NVDA；所有策略使用相同 real test dates。\n"
                    "- **Combined 定義：**不是資料列直接串接，而是每個 episode 以 50/50 機率抽 real 或該模型的 synthetic source；multi-path source 內再均勻抽 path。\n"
                    "- **比較單位：**相同 seed、相同 fee、相同 generator 的 paired delta；PPO 結果以 3 seeds 的平均與 sample SD 呈現。\n"
                    "- **指標方向：**報酬與 Sharpe 越高越好；波動、turnover 越低越好；最大回撤越接近 0 越好。"
                ),
                "layout": "full",
            },
            {
                "id": "methodology",
                "type": "markdown",
                "body": (
                    "## Methodology 與驗證\n\n"
                    "所有 PPO 使用相同架構與 hyperparameters，checkpoint 只依共同 real validation reward 選擇，最後以 deterministic rollout 在 real 2025 測試。"
                    "本分析先驗證 66 筆 test rows 完整覆蓋 22 個 fee × group cells、每格恰有 3 seeds、無重複 key、核心指標無缺值；"
                    "再計算 combined−real 與 combined−synthetic 的 seed-level paired deltas。圖中的誤差不作為顯著性證據；n=3 僅適合描述性判讀。"
                ),
                "layout": "full",
            },
            {
                "id": "limitations",
                "type": "markdown",
                "body": (
                    "## Limitations、uncertainty 與 robustness checks\n\n"
                    "- **Seeds 太少：**每條件僅 3 seeds，無法支持穩健的顯著性檢定或窄信賴區間。\n"
                    "- **單一市場 regime：**所有結論來自 2025 一年與 5 檔大型科技股，外推範圍有限。\n"
                    "- **多重比較：**若依同一 2025 test 從 10 個 synthetic/combined 候選挑 winner，會產生 test-set overfitting。\n"
                    "- **Scaler confound：**combined 與 synthetic-only 使用 generator-specific scaler，real-only 使用 real scaler；因此 combined-vs-real 同時改變資料 mix 與 scaler。\n"
                    "- **Post-hoc 圖表選 seed：**原 notebook 的 median-test-Sharpe representative 僅為描述；本報告的主結論使用全 seeds 聚合與配對值。\n"
                    "- **Fee delta 非純成本扣除：**no-fee 與 with-fee 分別訓練 policy，因此差異包含策略適應，不只是交易成本算術。"
                ),
                "layout": "full",
            },
            {
                "id": "next_steps",
                "type": "markdown",
                "body": (
                    "## Recommended next steps\n\n"
                    "1. **保留 real+synthetic 作為 augmentation，而非預設取代 synthetic-only。**目前它可靠地改善 real-only，但未可靠地改善 synthetic-only。\n"
                    "2. **優先驗證 Real + ARMD（無費率／風險調整）與 Real + CN-Diff（fee-aware 穩定性）。**先鎖定候選，再使用新的 untouched period。\n"
                    "3. **將 seeds 擴至至少 10–20，並加入 rolling out-of-sample windows。**以 period bootstrap 或跨窗口分布報告不確定性。\n"
                    "4. **做 real episode probability ablation：**0%、25%、50%、75%、100%，另做 joint-scaler 與 source-specific scaler ablation。\n"
                    "5. **正式模型選擇不得再看 2025 test。**用 validation 選 1–2 個候選，在下一個全新 holdout 一次性評估。"
                ),
                "layout": "full",
            },
            {
                "id": "further_questions",
                "type": "markdown",
                "body": (
                    "## Further questions\n\n"
                    "- Combined 的優勢是否集中在 2025 的特定月份或高波動 regime？\n"
                    "- 50/50 source ratio 是否會隨 synthetic path 數量（1 vs 50）改變？\n"
                    "- Joint scaler 是否能保留 ARMD 的風險改善並減少 input-range mismatch？\n"
                    "- 擴充資產 universe、不同資產類別與多個測試年度後，排名是否維持？"
                ),
                "layout": "full",
            },
        ],
    }
    snapshot = {
        "version": 1,
        "generatedAt": generated_at,
        "status": "ready",
        "datasets": {
            "headline_metrics": [headline_row],
            "paired_no_fee": paired_condition.loc[
                paired_condition["commission_name"].eq("no_fee")
            ].to_dict(orient="records"),
            "paired_with_fee": paired_condition.loc[
                paired_condition["commission_name"].eq("with_fee")
            ].to_dict(orient="records"),
            "strategy_comparison": strategy_comparison_rows(),
        },
    }
    payload = {
        "surface": "report",
        "manifest": manifest,
        "snapshot": snapshot,
        "sources": [test_source, comparison_source],
        "package_info": {
            "analysis_notebook": "analysis/synthetic_real_combined_analysis.ipynb",
            "validation": "Socket-free, in-process sequential smoke test required because managed environment denied Jupyter kernel socket binding.",
        },
    }
    ARTIFACT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(ARTIFACT_PATH)


if __name__ == "__main__":
    main()
