"""
Core Bayesian Framework for Distinguishing Robustness from Experimental Weakness.

Implements a Bayesian mixture model that computes:
- Bayes Factor BF01: P(data|H0) / P(data|H1)
- Posterior probability P(H0|data) and P(H1|data)
- Sensitivity analysis across prior hyperparameters
- Decision boundaries for 'conclusive robustness' vs 'inconclusive'

H0 (True Robustness): true effect size delta ~ N(0, epsilon²), epsilon small
H1 (Experimental Weakness): true effect size delta ~ N(mu_prior, sigma_prior²)
"""

import numpy as np
from scipy import stats
from scipy import integrate
import json
import warnings
warnings.filterwarnings('ignore')


def compute_bf01_ttest(t_stat: float, n: int,
                        mu_prior: float = 0.3, sigma_prior: float = 0.2,
                        null_effect: float = 0.0, null_width: float = 0.05) -> float:
    """
    Compute Bayes Factor BF01 comparing H0 (true robustness) vs H1 (masked effect).

    Uses the JZS-inspired approach: marginal likelihood under each hypothesis.

    Under H0: observed t ~ N(noncentrality=0, df) — centered t-distribution
    Under H1: integrate over prior on effect size delta ~ N(mu_prior, sigma_prior²)
              noncentrality = delta * sqrt(n)

    Args:
        t_stat: Observed t-statistic (or Cohen's d * sqrt(n))
        n: Sample size
        mu_prior: Prior mean effect size under H1 (Cohen's d units)
        sigma_prior: Prior SD on effect size under H1
        null_effect: Point null effect size (under H0)
        null_width: Width of null region (effect within +/- null_width counts as null)

    Returns:
        BF01: Bayes Factor favoring H0 over H1 (>1 favors robustness, <1 favors weakness)
    """
    # Noncentrality parameter: lambda = delta * sqrt(n)
    # Under H0: t ~ t(df=n-1) approximately N(0,1) for large n
    # Simplify with normal approximation (valid for n > 15)

    # Under H0 (effect = 0): likelihood of observed t
    p_data_H0 = stats.norm.pdf(t_stat, loc=null_effect * np.sqrt(n), scale=1.0)

    # Under H1: marginalize over prior on delta
    # P(t|H1) = integral_delta P(t|delta) * P(delta) d_delta
    # P(t|delta) = N(t; delta*sqrt(n), 1)
    # P(delta) = N(delta; mu_prior, sigma_prior²)
    # Analytically: P(t|H1) = N(t; mu_prior*sqrt(n), 1 + n*sigma_prior²)

    marginal_mean = mu_prior * np.sqrt(n)
    marginal_var = 1.0 + n * sigma_prior ** 2
    marginal_sd = np.sqrt(marginal_var)

    p_data_H1 = stats.norm.pdf(t_stat, loc=marginal_mean, scale=marginal_sd)

    # Avoid division by zero
    if p_data_H1 < 1e-300:
        return 1e6  # Very strong evidence for H0

    bf01 = p_data_H0 / p_data_H1
    return bf01


def compute_bf01_from_cohens_d(cohens_d: float, n: int, n2: int = None,
                                mu_prior: float = 0.3, sigma_prior: float = 0.2) -> float:
    """
    Compute BF01 from Cohen's d effect size and sample size.

    Args:
        cohens_d: Observed Cohen's d (absolute value of effect)
        n: Primary sample size (per group for two-sample)
        n2: Secondary group size (if two-sample, defaults to n)
        mu_prior: Prior mean effect size under H1
        sigma_prior: Prior SD on effect size under H1

    Returns:
        BF01: Bayes Factor favoring H0 (robustness) over H1 (weakness)
    """
    if n2 is None:
        n2 = n

    # Effective sample size for t-statistic
    n_eff = (n * n2) / (n + n2)  # harmonic mean / 2

    # Convert Cohen's d to approximate t-statistic
    t_stat = cohens_d * np.sqrt(n_eff)

    return compute_bf01_ttest(t_stat, n=int(n_eff), mu_prior=mu_prior, sigma_prior=sigma_prior)


def compute_posterior_weakness(bf01: float, prior_h0: float = 0.5) -> dict:
    """
    Compute posterior probability of each hypothesis given BF01.

    Args:
        bf01: Bayes Factor favoring H0 over H1
        prior_h0: Prior probability of H0 (true robustness), default 0.5

    Returns:
        dict with 'p_robust' and 'p_weakness' posteriors
    """
    prior_h1 = 1.0 - prior_h0

    # Posterior odds = BF01 * prior odds
    posterior_odds_h0 = bf01 * (prior_h0 / prior_h1)

    p_robust = posterior_odds_h0 / (1.0 + posterior_odds_h0)
    p_weakness = 1.0 - p_robust

    return {
        'p_robust': p_robust,
        'p_weakness': p_weakness,
        'bf01': bf01,
        'classification': classify_evidence(bf01, p_weakness)
    }


def classify_evidence(bf01: float, p_weakness: float) -> str:
    """
    Classify the null result based on BF01 and posterior weakness probability.

    Evidence categories (Jeffreys' scale adapted):
        Strong robustness: BF01 > 10
        Moderate robustness: BF01 > 3
        Inconclusive: 1/3 < BF01 < 3
        Moderate weakness: BF01 < 1/3
        Strong weakness: BF01 < 0.1
    """
    if bf01 > 10:
        return "strong_robustness"
    elif bf01 > 3:
        return "moderate_robustness"
    elif bf01 > 1/3:
        return "inconclusive"
    elif bf01 > 0.1:
        return "moderate_weakness"
    else:
        return "strong_weakness"


def sensitivity_analysis(cohens_d: float, n: int,
                          mu_prior_range: np.ndarray = None,
                          sigma_prior_range: np.ndarray = None,
                          n2: int = None) -> dict:
    """
    Run sensitivity analysis varying prior hyperparameters.

    Returns BF01 across the full range of plausible priors.
    """
    if mu_prior_range is None:
        mu_prior_range = np.linspace(0.1, 0.5, 9)
    if sigma_prior_range is None:
        sigma_prior_range = np.linspace(0.1, 0.35, 9)

    bf_grid = np.zeros((len(mu_prior_range), len(sigma_prior_range)))

    for i, mu in enumerate(mu_prior_range):
        for j, sigma in enumerate(sigma_prior_range):
            bf_grid[i, j] = compute_bf01_from_cohens_d(
                cohens_d, n, n2=n2, mu_prior=mu, sigma_prior=sigma
            )

    return {
        'mu_prior_range': mu_prior_range,
        'sigma_prior_range': sigma_prior_range,
        'bf_grid': bf_grid,
        'bf_min': float(np.min(bf_grid)),
        'bf_max': float(np.max(bf_grid)),
        'bf_median': float(np.median(bf_grid)),
        'robust_across_priors': bool(np.min(bf_grid) > 3),  # BF01>3 for ALL priors
        'weak_across_priors': bool(np.max(bf_grid) < 1/3),  # BF01<1/3 for ALL priors
    }


def retrospective_power(cohens_d: float, n: int, n2: int = None,
                         alpha: float = 0.05) -> float:
    """
    Compute classical retrospective (post-hoc) statistical power.

    Power = P(reject H0 | delta=observed_d, n, alpha)

    Note: Using the observed effect size as the 'true' effect — this is the
    classical (though controversial) retrospective power calculation.
    """
    if n2 is None:
        n2 = n

    n_eff = (n * n2) / (n + n2)

    # Critical t-value
    df = n + n2 - 2
    t_crit = stats.t.ppf(1 - alpha/2, df=df)

    # Noncentrality parameter
    ncp = cohens_d * np.sqrt(n_eff)

    # Power using noncentral t-distribution
    power = 1.0 - stats.nct.cdf(t_crit, df=df, nc=ncp) + stats.nct.cdf(-t_crit, df=df, nc=ncp)

    return max(0.0, min(1.0, power))


def analyze_study(study: dict, mu_prior: float = 0.3, sigma_prior: float = 0.2) -> dict:
    """
    Apply the full Bayesian framework to a single study.

    Args:
        study: Dict with keys: name, cohens_d, n, n2 (optional),
               reported_null (bool), intervention_type
        mu_prior: Prior mean effect size under H1
        sigma_prior: Prior SD on effect size under H1

    Returns:
        Analysis result dict with BF01, posteriors, power, classification
    """
    d = study.get('cohens_d', 0.0)
    n = study.get('n', 30)
    n2 = study.get('n2', None)

    bf01 = compute_bf01_from_cohens_d(d, n, n2=n2, mu_prior=mu_prior, sigma_prior=sigma_prior)
    posteriors = compute_posterior_weakness(bf01)
    power = retrospective_power(d, n, n2=n2)
    sens = sensitivity_analysis(d, n, n2=n2)

    result = {
        'study_name': study.get('name', 'Unknown'),
        'intervention_type': study.get('intervention_type', 'Unknown'),
        'cohens_d': d,
        'n': n,
        'reported_null': study.get('reported_null', True),
        'bf01': bf01,
        'log_bf01': float(np.log10(bf01)) if bf01 > 0 else -10,
        'p_robust': posteriors['p_robust'],
        'p_weakness': posteriors['p_weakness'],
        'classification': posteriors['classification'],
        'retrospective_power': power,
        'sensitivity_bf_min': sens['bf_min'],
        'sensitivity_bf_max': sens['bf_max'],
        'robust_across_all_priors': sens['robust_across_priors'],
        'weak_across_all_priors': sens['weak_across_priors'],
    }

    return result


if __name__ == "__main__":
    # Quick sanity check
    print("=== Bayesian Framework Sanity Check ===\n")

    # Test 1: Large n, small d (should favor H0: true robustness)
    print("Test 1: n=200, d=0.01 (should indicate robustness)")
    bf01 = compute_bf01_from_cohens_d(0.01, 200)
    post = compute_posterior_weakness(bf01)
    print(f"  BF01 = {bf01:.3f}, P(weakness) = {post['p_weakness']:.3f}, "
          f"Classification: {post['classification']}")

    # Test 2: Small n, moderate d (should be inconclusive)
    print("\nTest 2: n=15, d=0.3 (should be inconclusive)")
    bf01 = compute_bf01_from_cohens_d(0.3, 15)
    post = compute_posterior_weakness(bf01)
    print(f"  BF01 = {bf01:.3f}, P(weakness) = {post['p_weakness']:.3f}, "
          f"Classification: {post['classification']}")

    # Test 3: Very small n, moderate d (should favor H1: experimental weakness)
    print("\nTest 3: n=5, d=0.2 (should indicate experimental weakness)")
    bf01 = compute_bf01_from_cohens_d(0.2, 5)
    post = compute_posterior_weakness(bf01)
    print(f"  BF01 = {bf01:.3f}, P(weakness) = {post['p_weakness']:.3f}, "
          f"Classification: {post['classification']}")

    # Test 4: Sensitivity analysis
    print("\nTest 4: Sensitivity analysis for n=20, d=0.15")
    sens = sensitivity_analysis(0.15, 20)
    print(f"  BF01 range: [{sens['bf_min']:.3f}, {sens['bf_max']:.3f}]")
    print(f"  Robust across all priors: {sens['robust_across_priors']}")
    print(f"  Weak across all priors: {sens['weak_across_priors']}")
