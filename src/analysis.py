"""
Main analysis pipeline integrating:
1. Synthetic validation of the Bayesian framework
2. Retrospective analysis of published alignment null results
3. Prospective LLM experiment analysis
4. Comparison with classical power analysis
5. Sensitivity analysis across prior specifications
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import json
from pathlib import Path

from bayesian_framework import (
    compute_bf01_from_cohens_d, compute_posterior_weakness,
    retrospective_power, sensitivity_analysis, analyze_study,
    classify_evidence
)
from retrospective_dataset import ALIGNMENT_STUDIES

# ---- Reproducibility ----
np.random.seed(42)

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RETRO_DIR = Path("results/retrospective_analysis")
RETRO_DIR.mkdir(parents=True, exist_ok=True)
SYNTH_DIR = Path("results/synthetic_validation")
SYNTH_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PART 1: Synthetic Validation
# ============================================================

def run_synthetic_validation():
    """
    Validate the Bayesian framework under known ground truth conditions.

    Generates datasets with:
    - True null (δ=0)
    - Small effects (δ=0.1, 0.2)
    - Medium effects (δ=0.3, 0.4)
    - Large effects (δ=0.5)

    Across sample sizes n=5, 10, 20, 30, 50, 100, 200.

    Reports:
    - Classification accuracy (BF01 > 3 when truly null, < 1/3 when effect exists)
    - Calibration: P(classification = "robustness" | true delta = 0)
    """
    print("\n=== PART 1: Synthetic Validation ===")

    effect_sizes = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    sample_sizes = [5, 10, 20, 30, 50, 100, 200]
    n_simulations = 500
    mu_prior = 0.3
    sigma_prior = 0.2

    records = []
    for true_delta in effect_sizes:
        for n in sample_sizes:
            bf01_values = []
            p_robust_values = []
            classifications = []
            power_values = []

            for sim in range(n_simulations):
                # Generate observed Cohen's d by sampling from true distribution
                # t-statistic noise ~ N(0, 1) (standard error of d is approx 1/sqrt(n_eff))
                # Observed d = true_delta + noise / sqrt(n)
                noise_sd = np.sqrt(2/n)  # SE of Cohen's d for single-group vs reference
                observed_d = abs(true_delta + np.random.normal(0, noise_sd))

                bf01 = compute_bf01_from_cohens_d(
                    observed_d, n, mu_prior=mu_prior, sigma_prior=sigma_prior
                )
                post = compute_posterior_weakness(bf01)
                power = retrospective_power(observed_d, n)

                bf01_values.append(bf01)
                p_robust_values.append(post['p_robust'])
                classifications.append(post['classification'])
                power_values.append(power)

            # Summary statistics
            records.append({
                'true_delta': true_delta,
                'n': n,
                'mean_bf01': np.mean(bf01_values),
                'median_bf01': np.median(bf01_values),
                'mean_p_robust': np.mean(p_robust_values),
                'mean_power': np.mean(power_values),
                'pct_robustness': sum(1 for c in classifications
                                       if 'robustness' in c) / n_simulations,
                'pct_weakness': sum(1 for c in classifications
                                     if 'weakness' in c) / n_simulations,
                'pct_inconclusive': sum(1 for c in classifications
                                         if c == 'inconclusive') / n_simulations,
                'pct_strong_robustness': sum(1 for c in classifications
                                              if c == 'strong_robustness') / n_simulations,
            })

    df_synth = pd.DataFrame(records)
    df_synth.to_csv(SYNTH_DIR / "synthetic_validation.csv", index=False)
    print(f"Synthetic validation complete: {len(records)} conditions × {n_simulations} sims")

    # Key calibration metrics
    null_df = df_synth[df_synth['true_delta'] == 0.0]
    print("\nWhen true effect = 0 (true robustness):")
    print(null_df[['n', 'mean_bf01', 'mean_p_robust', 'pct_robustness',
                    'pct_inconclusive']].to_string(index=False))

    weak_df = df_synth[df_synth['true_delta'].isin([0.2, 0.3])]
    print("\nWhen true effect = 0.2 or 0.3 (potential weakness):")
    print(weak_df[['true_delta', 'n', 'mean_bf01', 'pct_weakness',
                    'mean_power']].to_string(index=False))

    return df_synth


# ============================================================
# PART 2: Retrospective Analysis of Published Studies
# ============================================================

def run_retrospective_analysis(mu_prior: float = 0.3, sigma_prior: float = 0.2) -> pd.DataFrame:
    """
    Apply the Bayesian framework to the curated dataset of published alignment studies.

    Computes for each study:
    - BF01 (Bayes Factor for H0 over H1)
    - P(weakness|data) - posterior probability of experimental weakness
    - Classification (strong_robustness, moderate_robustness, inconclusive,
                       moderate_weakness, strong_weakness)
    - Retrospective classical power
    - Sensitivity range across priors
    """
    print("\n=== PART 2: Retrospective Analysis ===")

    results = []
    for study in ALIGNMENT_STUDIES:
        r = analyze_study(study, mu_prior=mu_prior, sigma_prior=sigma_prior)
        results.append(r)

    df = pd.DataFrame(results)
    df.to_csv(RETRO_DIR / "retrospective_results.csv", index=False)

    # Summary statistics
    null_df = df[df['reported_null'] == True]
    pos_df = df[df['reported_null'] == False]

    print(f"\nTotal studies: {len(df)}")
    print(f"Null results: {len(null_df)}")
    print(f"Positive results: {len(pos_df)}")

    print(f"\n--- Null Result Classifications ---")
    for cls in ["strong_robustness", "moderate_robustness", "inconclusive",
                "moderate_weakness", "strong_weakness"]:
        count = (null_df['classification'] == cls).sum()
        pct = 100 * count / len(null_df)
        print(f"  {cls}: {count} ({pct:.0f}%)")

    # Key hypothesis tests
    pct_weakness_criterion = (null_df['p_weakness'] > 0.3).mean()
    print(f"\nH1 test: P(weakness) > 0.3 for {100*pct_weakness_criterion:.1f}% "
          f"of null results")
    print(f"  (Hypothesis predicts >40%, threshold for confirmation)")

    # Correlation with power
    corr, pval = stats.spearmanr(null_df['bf01'], null_df['retrospective_power'])
    print(f"\nH2 test: Spearman ρ(BF01, power) = {corr:.3f} (p={pval:.4f})")
    print(f"  (Hypothesis predicts r > 0.7; high correlation expected)")

    return df


def run_sensitivity_analysis_global(df: pd.DataFrame) -> dict:
    """
    Run sensitivity analysis across prior specifications for all null result studies.

    Shows how classification changes as priors vary.
    """
    print("\n=== Sensitivity Analysis ===")

    mu_range = np.linspace(0.1, 0.5, 9)
    sigma_range = np.linspace(0.1, 0.35, 9)

    null_df = df[df['reported_null'] == True]
    sensitivity_results = {}

    # For each study, compute sensitivity
    pct_weak_across_priors = []
    for _, row in null_df.iterrows():
        sens = sensitivity_analysis(
            row['cohens_d'], row['n'],
            mu_prior_range=mu_range, sigma_prior_range=sigma_range
        )
        pct_weak_this = (sens['bf_grid'] < 1/3).mean() * 100
        pct_weak_across_priors.append(pct_weak_this)

    sensitivity_results['pct_weak_by_study'] = pct_weak_across_priors
    sensitivity_results['mean_pct_weak_across_priors'] = np.mean(pct_weak_across_priors)
    sensitivity_results['studies_weak_all_priors'] = sum(1 for x in pct_weak_across_priors
                                                          if x > 90)
    sensitivity_results['studies_robust_all_priors'] = sum(1 for x in pct_weak_across_priors
                                                             if x < 10)
    sensitivity_results['mu_range'] = mu_range.tolist()
    sensitivity_results['sigma_range'] = sigma_range.tolist()

    print(f"Null studies with >90% of prior settings favoring weakness: "
          f"{sensitivity_results['studies_weak_all_priors']}/{len(null_df)}")
    print(f"Null studies with <10% of prior settings favoring weakness "
          f"(consistently robust): {sensitivity_results['studies_robust_all_priors']}/{len(null_df)}")

    with open(RETRO_DIR / "sensitivity_results.json", "w") as f:
        json.dump(sensitivity_results, f, indent=2)

    return sensitivity_results


# ============================================================
# PART 3: LLM Experiment Analysis
# ============================================================

def analyze_llm_experiments(llm_results: dict) -> dict:
    """
    Apply the Bayesian framework to the prospective LLM experiments.

    Computes:
    - BF01 for "true robustness" vs "intervention too weak"
    - Decision trajectory as evidence accumulates sequentially
    """
    print("\n=== PART 3: LLM Experiment Analysis ===")

    analysis = {}

    for experiment_name, experiment_data in [
        ("jailbreak", llm_results.get("jailbreak", {})),
        ("sycophancy", llm_results.get("sycophancy", {}))
    ]:
        print(f"\n--- {experiment_name.capitalize()} Experiment ---")
        exp_analysis = {}

        # Determine conditions and rate key
        if experiment_name == "jailbreak":
            conditions = ["weak", "medium", "strong"]
            rate_key = "jailbreak_rate"
            # For jailbreak: H0 = model is robust (jailbreak_rate near 0)
            # H1 = intervention too weak (true vulnerability exists)
            # Prior: beta(1, 5) → expected rate around 0.15 if vulnerable
            mu_prior_effect = 0.3  # Expected effect size (difference from 0) if truly vulnerable
        else:
            conditions = ["low", "medium", "high"]
            rate_key = "sycophancy_rate"
            # For sycophancy: H0 = model maintains position (rate near 0)
            # H1 = susceptible (sycophancy exists but pressure was insufficient)
            mu_prior_effect = 0.3

        for cond in conditions:
            if cond not in experiment_data:
                continue
            cond_data = experiment_data[cond]
            rate = cond_data.get(rate_key, 0.0)
            n = cond_data.get("n", 10)

            # Convert proportion to Cohen's h (arcsine transformation)
            # Effect vs. null baseline of 0
            cohens_h = 2 * np.arcsin(np.sqrt(max(0.001, min(0.999, rate))))
            # Null baseline: h=0 (zero rate)

            # BF01 for this condition
            bf01 = compute_bf01_from_cohens_d(
                cohens_h, n, mu_prior=mu_prior_effect
            )
            post = compute_posterior_weakness(bf01)
            power = retrospective_power(cohens_h, n)

            exp_analysis[cond] = {
                "rate": rate,
                "n": n,
                "cohens_h": cohens_h,
                "bf01": bf01,
                "p_robust": post['p_robust'],
                "p_weakness": post['p_weakness'],
                "classification": post['classification'],
                "retrospective_power": power,
            }

            print(f"  {cond}: rate={rate:.1%}, n={n}, Cohen's h={cohens_h:.3f}, "
                  f"BF01={bf01:.3f}, P(weakness)={post['p_weakness']:.3f}, "
                  f"class={post['classification']}")

        analysis[experiment_name] = exp_analysis

    with open(RESULTS_DIR / "llm_experiments" / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    return analysis


# ============================================================
# PART 4: Sequential Analysis (Prospective)
# ============================================================

def run_sequential_analysis(results_data: dict) -> dict:
    """
    Demonstrate sequential Bayesian analysis: show how evidence accumulates
    as trials are added one by one (simulating prospective stopping rules).

    Decision rule: stop when BF01 > 10 (strong robustness) or BF01 < 0.1 (strong weakness).
    """
    print("\n=== PART 4: Sequential Bayesian Analysis ===")

    sequential_results = {}

    for exp_name, conditions in [
        ("jailbreak", ["weak", "medium", "strong"]),
        ("sycophancy", ["low", "medium", "high"])
    ]:
        exp_data = results_data.get(exp_name, {})
        sequential_results[exp_name] = {}

        for cond in conditions:
            if cond not in exp_data:
                continue

            # Recover trial-level data
            cond_data = exp_data[cond]
            trials = cond_data.get("trials", [])
            rate_key = "success" if exp_name == "jailbreak" else "sycophantic"
            observations = [t.get(rate_key, 0) for t in trials]

            # Sequential BF computation
            bf01_trajectory = []
            p_weakness_trajectory = []
            stopping_point = None

            for i in range(1, len(observations) + 1):
                partial_obs = observations[:i]
                # Rate so far
                current_rate = sum(partial_obs) / i
                cohens_h = 2 * np.arcsin(np.sqrt(max(0.001, min(0.999, current_rate))))

                bf01 = compute_bf01_from_cohens_d(cohens_h, i)
                post = compute_posterior_weakness(bf01)

                bf01_trajectory.append(bf01)
                p_weakness_trajectory.append(post['p_weakness'])

                # Check stopping rule
                if stopping_point is None:
                    if bf01 > 10 or bf01 < 0.1:
                        stopping_point = i

            sequential_results[exp_name][cond] = {
                "bf01_trajectory": bf01_trajectory,
                "p_weakness_trajectory": p_weakness_trajectory,
                "stopping_point": stopping_point,
                "final_bf01": bf01_trajectory[-1] if bf01_trajectory else None,
                "final_classification": classify_evidence(
                    bf01_trajectory[-1] if bf01_trajectory else 1.0,
                    p_weakness_trajectory[-1] if p_weakness_trajectory else 0.5
                ) if bf01_trajectory else "insufficient_data"
            }

            print(f"  {exp_name}/{cond}: final BF01={bf01_trajectory[-1]:.3f}, "
                  f"stopping={'trial ' + str(stopping_point) if stopping_point else 'did not stop'}")

    return sequential_results


# ============================================================
# PART 5: Visualizations
# ============================================================

def create_visualizations(df_synth: pd.DataFrame, df_retro: pd.DataFrame,
                           llm_analysis: dict, seq_analysis: dict,
                           sensitivity: dict):
    """Create comprehensive visualization suite."""
    print("\n=== Creating Visualizations ===")

    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams['figure.dpi'] = 100

    # ---- Figure 1: Synthetic Validation Heatmaps ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Pivot: BF01 by true_delta × n (null case)
    null_data = df_synth[df_synth['true_delta'] == 0].pivot_table(
        values='mean_bf01', index='n', columns='true_delta', aggfunc='mean'
    )
    full_data = df_synth.pivot_table(
        values='mean_bf01', index='n', columns='true_delta', aggfunc='mean'
    )

    sns.heatmap(np.log10(full_data + 0.01), ax=axes[0], cmap='RdYlGn',
                annot=True, fmt='.2f', cbar_kws={'label': 'log10(BF01)'},
                linewidths=0.5)
    axes[0].set_title('Mean BF01 (log10) by Effect Size and Sample Size\n(Green=Evidence for Robustness, Red=Evidence for Weakness)')
    axes[0].set_xlabel('True Cohen\'s d (δ=0 means genuinely null)')
    axes[0].set_ylabel('Sample Size (n)')

    # Pct classified as "robustness" when true delta = 0
    null_pct = df_synth[df_synth['true_delta'] == 0].pivot_table(
        values='pct_robustness', index='n', columns='true_delta', aggfunc='mean'
    )
    # Pct classified as "weakness/inconclusive" when true delta > 0
    alt_pct = df_synth[df_synth['true_delta'] > 0].pivot_table(
        values='pct_weakness', index='n', columns='true_delta', aggfunc='mean'
    )

    # Show overall classification rates
    cls_data = df_synth.pivot_table(
        values='pct_inconclusive', index='n', columns='true_delta', aggfunc='mean'
    )
    sns.heatmap(cls_data, ax=axes[1], cmap='YlOrBr',
                annot=True, fmt='.2f', vmin=0, vmax=1,
                cbar_kws={'label': 'P(inconclusive)'},
                linewidths=0.5)
    axes[1].set_title('Probability of Inconclusive Classification\nby Effect Size and Sample Size')
    axes[1].set_xlabel('True Cohen\'s d')
    axes[1].set_ylabel('Sample Size (n)')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_synthetic_validation.png", bbox_inches='tight')
    plt.close()
    print("  Saved fig1_synthetic_validation.png")

    # ---- Figure 2: Retrospective Analysis Results ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    null_df = df_retro[df_retro['reported_null'] == True].copy()
    pos_df = df_retro[df_retro['reported_null'] == False].copy()

    # 2a: Distribution of BF01 for null results (log scale)
    ax = axes[0, 0]
    log_bf = np.log10(null_df['bf01'].clip(0.001, 1000))
    ax.hist(log_bf, bins=15, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(np.log10(3), color='green', linestyle='--', lw=2, label='BF01=3 (Moderate Robustness)')
    ax.axvline(np.log10(1/3), color='red', linestyle='--', lw=2, label='BF01=1/3 (Moderate Weakness)')
    ax.axvline(np.log10(1), color='gray', linestyle=':', lw=1.5, label='BF01=1 (No Evidence)')
    ax.set_xlabel('log10(BF01)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Bayes Factors\nfor Published Null Results')
    ax.legend(fontsize=8)

    # 2b: Classification breakdown
    ax = axes[0, 1]
    cls_counts = null_df['classification'].value_counts()
    colors = {'strong_robustness': '#2d7a2d', 'moderate_robustness': '#82c882',
              'inconclusive': '#f0c050', 'moderate_weakness': '#e88040',
              'strong_weakness': '#cc2222'}
    cls_labels = []
    cls_values = []
    cls_colors = []
    for cls in ["strong_robustness", "moderate_robustness", "inconclusive",
                "moderate_weakness", "strong_weakness"]:
        cls_labels.append(cls.replace('_', '\n'))
        cls_values.append(cls_counts.get(cls, 0))
        cls_colors.append(colors[cls])

    bars = ax.bar(cls_labels, cls_values, color=cls_colors, edgecolor='white')
    for bar, val in zip(bars, cls_values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    str(val), ha='center', va='bottom', fontweight='bold')
    ax.set_title(f'Classification of {len(null_df)} Null Results\n(Jeffreys-Scale Evidence)')
    ax.set_ylabel('Number of Studies')
    ax.set_ylim(0, max(cls_values) + 2)

    # 2c: BF01 vs. Retrospective Power (H2 test)
    ax = axes[1, 0]
    ax.scatter(null_df['retrospective_power'], null_df['bf01'],
               c=null_df['p_weakness'], cmap='RdYlGn_r', s=60,
               alpha=0.8, edgecolors='gray', linewidths=0.5)
    ax.set_xscale('linear')
    ax.set_yscale('log')
    ax.set_xlabel('Classical Retrospective Power')
    ax.set_ylabel('BF01 (log scale)')
    ax.set_title('BF01 vs. Classical Power\n(Color = P(weakness|data))')
    ax.axhline(3, color='green', linestyle='--', lw=1.5, alpha=0.7, label='BF01=3')
    ax.axhline(1/3, color='red', linestyle='--', lw=1.5, alpha=0.7, label='BF01=1/3')
    ax.legend(fontsize=8)
    corr, pval = stats.spearmanr(null_df['bf01'], null_df['retrospective_power'])
    ax.text(0.05, 0.95, f'ρ={corr:.3f} (p={pval:.3f})',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2d: P(weakness) by intervention type
    ax = axes[1, 1]
    type_order = ["activation_steering", "rlhf", "constitutional_ai",
                  "benchmark_evaluation", "jailbreak_resistance", "prompting", "refusal_training"]
    type_labels = ["Activation\nSteering", "RLHF", "Constitutional\nAI",
                   "Benchmark\nEval", "Jailbreak\nResistance", "Prompting", "Refusal\nTraining"]

    # Filter null results only for this plot
    type_pw = []
    type_names_present = []
    type_labels_present = []
    for t, lbl in zip(type_order, type_labels):
        sub = null_df[null_df['intervention_type'] == t]
        if len(sub) > 0:
            type_pw.append(sub['p_weakness'].values)
            type_names_present.append(t)
            type_labels_present.append(lbl)

    bp = ax.boxplot(type_pw, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2))
    ax.set_xticks(range(1, len(type_labels_present) + 1))
    ax.set_xticklabels(type_labels_present)
    for patch in bp['boxes']:
        patch.set_facecolor('#7eb4e2')
        patch.set_alpha(0.8)
    ax.axhline(0.3, color='red', linestyle='--', lw=1.5,
               label='P(weakness)=0.3 threshold')
    ax.set_title('P(Experimental Weakness|Data)\nby Intervention Type (Null Results)')
    ax.set_ylabel('P(weakness|data)')
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', labelsize=8)
    ax.legend(fontsize=8)

    plt.suptitle('Retrospective Bayesian Analysis of Published Alignment Null Results',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_retrospective_analysis.png", bbox_inches='tight')
    plt.close()
    print("  Saved fig2_retrospective_analysis.png")

    # ---- Figure 3: LLM Experiment Evidence ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, (exp_name, cond_labels, rate_key_label) in enumerate([
        ("jailbreak", ["weak", "medium", "strong"], "Jailbreak Rate"),
        ("sycophancy", ["low", "medium", "high"], "Sycophancy Rate")
    ]):
        ax = axes[ax_idx]
        if exp_name not in llm_analysis:
            continue

        conds = cond_labels
        rates = [llm_analysis[exp_name].get(c, {}).get("rate", 0) for c in conds]
        bf01s = [llm_analysis[exp_name].get(c, {}).get("bf01", 1) for c in conds]
        p_weaks = [llm_analysis[exp_name].get(c, {}).get("p_weakness", 0.5) for c in conds]

        x = np.arange(len(conds))
        width = 0.35

        bars1 = ax.bar(x - width/2, rates, width, label=rate_key_label,
                        color='steelblue', alpha=0.8)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel(f'{rate_key_label}', color='steelblue')

        ax2 = ax.twinx()
        ax2.plot(x, p_weaks, 'o-', color='red', lw=2, ms=8, label='P(weakness)')
        ax2.axhline(0.3, color='red', linestyle='--', lw=1, alpha=0.5)
        ax2.set_ylabel('P(Experimental Weakness)', color='red')
        ax2.set_ylim(0, 1)

        ax.set_xticks(x)
        ax.set_xticklabels([c.capitalize() for c in conds])
        ax.set_xlabel('Intervention Strength')

        title_map = {"jailbreak": "Jailbreak Resistance", "sycophancy": "Sycophancy Susceptibility"}
        ax.set_title(f'{title_map[exp_name]}\n(Blue=Rate, Red=P(weakness))')

        # Add BF01 annotations
        for i, (c, bf) in enumerate(zip(conds, bf01s)):
            ax.text(i, max(rates[i], 0.05) + 0.05, f'BF01={bf:.2f}',
                    ha='center', fontsize=8, color='darkblue')

    plt.suptitle('Prospective LLM Experiment: Evidence Under Varying Intervention Strength',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_llm_experiments.png", bbox_inches='tight')
    plt.close()
    print("  Saved fig3_llm_experiments.png")

    # ---- Figure 4: Sequential Evidence Accumulation ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, exp_name in enumerate(["jailbreak", "sycophancy"]):
        ax = axes[ax_idx]
        if exp_name not in seq_analysis:
            continue

        cond_colors = {"weak": "green", "medium": "orange", "strong": "red",
                       "low": "green", "medium": "orange", "high": "red"}

        for cond, cond_data in seq_analysis[exp_name].items():
            traj = cond_data.get("bf01_trajectory", [])
            if traj:
                log_traj = [np.log10(max(b, 0.001)) for b in traj]
                ax.plot(range(1, len(log_traj)+1), log_traj,
                        label=f'{cond.capitalize()} intervention',
                        color=cond_colors.get(cond, 'blue'), lw=2, marker='o', ms=4)

                sp = cond_data.get("stopping_point")
                if sp:
                    ax.axvline(sp, color=cond_colors.get(cond, 'blue'),
                               linestyle=':', lw=1.5, alpha=0.5)

        ax.axhline(np.log10(10), color='green', linestyle='--', lw=1.5,
                   label='Stop: Strong Robustness (BF01=10)')
        ax.axhline(np.log10(1/10), color='red', linestyle='--', lw=1.5,
                   label='Stop: Strong Weakness (BF01=0.1)')
        ax.axhline(0, color='gray', linestyle=':', lw=1, alpha=0.5)
        ax.set_xlabel('Number of Trials')
        ax.set_ylabel('log10(BF01)')
        ax.set_title(f'Sequential Evidence Accumulation\n({exp_name.capitalize()} Experiment)')
        ax.legend(fontsize=7)

    plt.suptitle('Sequential Bayesian Analysis: Evidence Accumulation by Intervention Strength',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig4_sequential_analysis.png", bbox_inches='tight')
    plt.close()
    print("  Saved fig4_sequential_analysis.png")

    # ---- Figure 5: Sensitivity Analysis ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 5a: Sensitivity across priors for a representative study
    # Use the study "CAA sycophancy weak steering coefficient" (d=0.15, n=200)
    rep_study = next(s for s in ALIGNMENT_STUDIES
                     if "weak steering" in s["name"].lower())
    mu_range = np.linspace(0.1, 0.5, 20)
    sigma_range = np.array([0.1, 0.2, 0.3])

    ax = axes[0]
    for sigma_v in sigma_range:
        bfs = [compute_bf01_from_cohens_d(rep_study['cohens_d'], rep_study['n'],
                                           mu_prior=mu, sigma_prior=sigma_v)
               for mu in mu_range]
        ax.plot(mu_range, np.log10(bfs), label=f'σ_prior={sigma_v}', lw=2)

    ax.axhline(np.log10(3), color='green', linestyle='--', lw=1.5, label='BF01=3 threshold')
    ax.axhline(np.log10(1/3), color='red', linestyle='--', lw=1.5, label='BF01=1/3 threshold')
    ax.set_xlabel('Prior Mean Effect Size (μ_prior)')
    ax.set_ylabel('log10(BF01)')
    ax.set_title(f'Prior Sensitivity Analysis\n("{rep_study["name"][:40]}...")')
    ax.legend(fontsize=8)
    ax.fill_between(mu_range, np.log10(1/3), np.log10(3),
                    alpha=0.1, color='yellow', label='Inconclusive zone')

    # 5b: Proportion of null studies whose classification changes across priors
    ax = axes[1]
    pct_values = sensitivity.get('pct_weak_by_study', [])
    ax.hist(pct_values, bins=10, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(50, color='gray', linestyle='--', lw=1.5,
               label='50% of prior settings')
    ax.set_xlabel('% of Prior Settings Favoring Weakness (BF01 < 1/3)')
    ax.set_ylabel('Number of Null Result Studies')
    ax.set_title('Prior Sensitivity: How Robust are\nClassifications Across Prior Specs?')
    ax.legend(fontsize=8)

    plt.suptitle('Sensitivity Analysis: Robustness of Conclusions to Prior Specification',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig5_sensitivity_analysis.png", bbox_inches='tight')
    plt.close()
    print("  Saved fig5_sensitivity_analysis.png")

    print(f"\nAll figures saved to {FIGURES_DIR}")


# ============================================================
# Main execution
# ============================================================

def main():
    print("=" * 60)
    print("BAYESIAN EVIDENCE QUANTIFICATION ANALYSIS")
    print("=" * 60)

    # Part 1: Synthetic validation
    df_synth = run_synthetic_validation()

    # Part 2: Retrospective analysis
    df_retro = run_retrospective_analysis()

    # Calculate key results for H1 test
    null_df = df_retro[df_retro['reported_null'] == True]
    pct_weakness_threshold = (null_df['p_weakness'] > 0.3).mean()
    print(f"\n*** H1 RESULT: {100*pct_weakness_threshold:.1f}% of null results "
          f"have P(weakness) > 0.3 ***")
    print(f"    (Hypothesis: >40%; {'SUPPORTED' if pct_weakness_threshold > 0.4 else 'NOT SUPPORTED'})")

    # Part 3: Sensitivity analysis
    sensitivity = run_sensitivity_analysis_global(df_retro)

    # Part 4: Load LLM results
    llm_results_file = Path("results/llm_experiments/raw_results.json")
    if llm_results_file.exists():
        with open(llm_results_file) as f:
            llm_results = json.load(f)
        print(f"\nLoaded LLM experiment results from {llm_results_file}")
    else:
        print("\nLLM results not found, using fallback simulated data")
        from llm_experiments import _simulate_results
        llm_results = _simulate_results(10)

    llm_analysis = analyze_llm_experiments(llm_results)

    # Sequential analysis
    seq_analysis = run_sequential_analysis(llm_results)

    # Part 5: Visualizations
    create_visualizations(df_synth, df_retro, llm_analysis, seq_analysis, sensitivity)

    # Compile summary statistics
    corr_h2, pval_h2 = stats.spearmanr(null_df['bf01'], null_df['retrospective_power'])

    # H3: Sequential analysis agreement
    # Compare sequential stopping classification with final post-hoc classification
    seq_cls = []
    posthoc_cls = []
    for exp_name, cond_data in seq_analysis.items():
        for cond, data in cond_data.items():
            if exp_name in llm_analysis and cond in llm_analysis[exp_name]:
                seq_c = data.get("final_classification", "inconclusive")
                posthoc_c = llm_analysis[exp_name][cond].get("classification", "inconclusive")
                seq_cls.append(seq_c)
                posthoc_cls.append(posthoc_c)

    # Compute kappa if we have enough data
    kappa = _compute_kappa(seq_cls, posthoc_cls)

    summary = {
        "n_studies_total": len(df_retro),
        "n_null_results": len(null_df),
        "pct_weakness_above_threshold": float(pct_weakness_threshold),
        "h1_supported": bool(pct_weakness_threshold > 0.4),
        "spearman_corr_bf01_power": float(corr_h2),
        "h2_supported": bool(abs(corr_h2) > 0.7),
        "sequential_kappa": float(kappa) if kappa is not None else None,
        "h3_supported": bool(kappa > 0.6) if kappa is not None else None,
        "classification_breakdown": null_df['classification'].value_counts().to_dict(),
        "mean_p_weakness_null": float(null_df['p_weakness'].mean()),
        "sensitivity_robust_all_priors": sensitivity['studies_robust_all_priors'],
        "sensitivity_weak_all_priors": sensitivity['studies_weak_all_priors'],
    }

    with open(RESULTS_DIR / "summary_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== FINAL SUMMARY ===")
    print(f"H1 (>40% have P(weakness)>0.3): {pct_weakness_threshold:.1%} — "
          f"{'SUPPORTED' if summary['h1_supported'] else 'NOT SUPPORTED'}")
    print(f"H2 (correlation BF01 vs power > 0.7): ρ={corr_h2:.3f} — "
          f"{'SUPPORTED' if summary['h2_supported'] else 'NOT SUPPORTED'}")
    print(f"H3 (sequential/post-hoc agreement κ > 0.6): κ={kappa:.3f} — "
          f"{'SUPPORTED' if summary['h3_supported'] else 'NOT SUPPORTED'}")
    print(f"\nResults saved to {RESULTS_DIR}")

    return summary


def _compute_kappa(cls1: list, cls2: list) -> float:
    """Compute Cohen's kappa for two classification lists."""
    if len(cls1) < 2 or len(cls1) != len(cls2):
        return 0.0

    # Collapse to 3 categories for kappa computation
    def coarsen(c):
        if 'robustness' in c:
            return 'robust'
        elif 'weakness' in c:
            return 'weak'
        else:
            return 'inconclusive'

    c1 = [coarsen(c) for c in cls1]
    c2 = [coarsen(c) for c in cls2]

    # Simple kappa
    n = len(c1)
    observed_agree = sum(a == b for a, b in zip(c1, c2)) / n

    cats = set(c1 + c2)
    expected_agree = sum(
        (c1.count(cat) / n) * (c2.count(cat) / n)
        for cat in cats
    )

    if expected_agree >= 1.0:
        return 1.0

    kappa = (observed_agree - expected_agree) / (1.0 - expected_agree)
    return kappa


if __name__ == "__main__":
    main()
