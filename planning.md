# Research Planning: Bayesian Evidence Quantification for AI Alignment Null Results

## Motivation & Novelty Assessment

### Why This Research Matters

AI alignment research suffers from an interpretational crisis with null results: when an intervention (e.g., activation steering, RLHF fine-tuning, constitutional AI prompting) fails to induce misalignment or shows no measurable behavioral change, this can mean either (a) the model is genuinely robust, or (b) the experiment was too weak to detect an existing effect. This ambiguity impedes cumulative science—robustness claims get published without warranted confidence, and failed interventions get dismissed without understanding why. A principled Bayesian framework that makes this distinction quantitative could transform how the alignment community designs and interprets experiments.

### Gap in Existing Work

The literature review reveals that:
- Bayes factors (BF01) exist for null hypothesis testing but are rarely applied in alignment research
- Alignment papers typically report binary null results (p > 0.05) without power analysis
- No framework currently connects intervention strength, sample size, measurement sensitivity, and prior effect sizes in a unified Bayesian model for alignment evaluation
- The "five-nines reliability" work (Kim et al., 2026) begins to quantify evaluation requirements but doesn't provide a retrospective diagnostic tool
- Computational safety hypothesis testing (Chen, 2025) addresses forward design but not retrospective interpretation

### Our Novel Contribution

We contribute: (1) a hierarchical Bayesian mixture model that explicitly separates "true null" from "masked effect" hypotheses; (2) a curated dataset of 30+ alignment null results with extracted statistical parameters; (3) prospective validation using real LLM experiments with systematic variation of intervention strength; (4) comparison between Bayesian diagnostics and classical power analysis showing the added value of the Bayesian approach.

### Experiment Justification

- **Experiment 1 (Synthetic Validation)**: Needed to verify the framework works under known ground truth before applying to noisy real data
- **Experiment 2 (Retrospective Analysis)**: Core hypothesis test—do published null results lean toward "insufficient evidence" vs. "genuine robustness"?
- **Experiment 3 (LLM Prospective)**: Demonstrates the framework working prospectively with real data; tests jailbreak/sycophancy resistance under systematically varied intervention strength
- **Experiment 4 (Sensitivity Analysis)**: Needed to show conclusions are robust to prior specification, which is a major concern with Bayesian methods

---

## Research Question

Can a Bayesian framework that models intervention strength, measurement sensitivity, and effect size priors reliably distinguish genuine model robustness from underpowered null results in AI alignment research?

**Sub-hypothesis 1**: >40% of published alignment null results have posterior P(experimental_weakness|data) > 0.3

**Sub-hypothesis 2**: Bayesian diagnostics (BF01, P_weakness) correlate with classical retrospective power (r > 0.7)

**Sub-hypothesis 3**: Prospective Bayesian sequential analysis agrees with post-hoc analysis (Cohen's kappa > 0.6)

---

## Methodology

### Bayesian Model Specification

**Hypotheses**:
- H0 (True Robustness): Effect size δ ~ N(0, σ_noise²) — genuinely null
- H1 (Experimental Weakness): Effect size δ ~ N(μ_prior, σ_prior²) — effect exists but may be undetected

**Prior on effect size for H1**: δ ~ N(0.3, 0.2²) based on alignment literature
  - Mean 0.3: Typical small-to-medium effect in behavioral interventions
  - SD 0.2: Uncertainty about effect magnitude

**Likelihood**: Given observed test statistic t = d/SE = d/(σ/√n):
  - Under H0: t ~ N(0, 1) (standardized, effect = 0)
  - Under H1: t ~ N(δ√n/σ, 1) integrated over prior on δ

**Bayes Factor**:
  BF01 = P(data|H0) / P(data|H1)
  
  BF01 > 3: Substantial evidence for H0 (true robustness)
  BF01 < 1/3: Substantial evidence for H1 (experimental weakness)
  1/3 < BF01 < 3: Inconclusive

**Posterior P(H0|data)**:
  P(H0|data) = BF01 * P(H0) / (BF01 * P(H0) + P(H1))
  with P(H0) = 0.5 (equal priors)

**Intervention Strength Model**:
  - Strong: detected effect size factor k=1.0 of maximum possible
  - Medium: k=0.5
  - Weak: k=0.2
  - Measurement sensitivity adds noise: σ_obs = σ_true / sensitivity

### Experimental Steps

1. **Framework implementation** (Python, scipy-based)
   - Bayesian mixture model with closed-form BF computation
   - Sensitivity analysis across prior hyperparameters
   - Classical power analysis comparison

2. **Synthetic validation**
   - Generate data under H0 (true null) and H1 (weak intervention)
   - Verify BF01 correctly classifies 90%+ under unambiguous conditions
   - Verify calibration: P(insuff|data) matches empirical frequency

3. **Retrospective study curation**
   - 30-50 studies extracted from papers in papers/ directory
   - Key parameters: n, d (Cohen's d or equivalent), intervention type, reported significance
   - Focus on activation steering (Papers 17-22) and RLHF (Papers 23-29)

4. **Real LLM experiments via OpenRouter**
   - Jailbreak resistance test: weak/medium/strong jailbreak attempts, 20 trials each
   - Sycophancy test: varying levels of social pressure, 20 trials each
   - Models: meta-llama/llama-3.1-8b-instruct (accessible via OpenRouter)
   - Measure: binary success/failure for jailbreak; sycophancy score for sycophancy

5. **Analysis**
   - Compute BF01 and P_weakness for each study
   - Classify null results
   - Compute correlation with retrospective power
   - Sensitivity to prior specification

### Evaluation Metrics

- BF01 per study (primary)
- P(weakness|data) per study (primary)
- Proportion classified as inconclusive (expected: >40%)
- Spearman ρ between BF01 and retrospective power (expected: r > 0.7)
- Cohen's κ between Bayesian stopping and post-hoc classification
- Sensitivity range (BF01 across prior hyperparameter range)

### Baselines

- Classical post-hoc power analysis (1-β given observed n, d, α=0.05)
- Frequentist significance test (p-value, binary)
- Naive BF with default Cauchy prior (JZS prior)

### Statistical Plan

- Report posterior probabilities with credible intervals (bootstrap uncertainty on P_weakness)
- Spearman correlation for BF01 vs. power (non-parametric, appropriate for bounded values)
- Cohen's κ for categorical agreement
- Sensitivity analysis: vary μ_prior ∈ [0.1, 0.5], σ_prior ∈ [0.1, 0.3]

---

## Expected Outcomes

- **Supporting H1**: BF01 < 1/3 for >40% of null results (experimental weakness)
- **Supporting H2**: ρ > 0.7 between BF01 and classical power
- **Supporting H3**: κ > 0.6 for sequential analysis agreement
- If these aren't met: Framework still provides value as a diagnostic tool with richer information than p-values alone

---

## Timeline

- T+0:00 - Environment setup and package installation ✓
- T+0:20 - Framework implementation (Bayesian model, power analysis)
- T+0:50 - Synthetic validation and retrospective dataset curation
- T+1:20 - LLM experiments via OpenRouter
- T+1:50 - Analysis and visualization
- T+2:20 - Documentation (REPORT.md, README.md)

---

## Potential Challenges

1. **OpenRouter API rate limits**: Mitigate by caching responses and using smaller batch sizes
2. **Missing quantitative data in papers**: Approximate effect sizes from reported statistics or use conservative estimates
3. **Prior sensitivity**: Run full sensitivity analysis to show robustness of conclusions
4. **Computational cost of MCMC**: Use closed-form BF computation (JZS Cauchy) instead

## File Structure

```
src/
├── bayesian_framework.py    # Core Bayesian model
├── power_analysis.py        # Classical power analysis for comparison
├── retrospective_dataset.py # Curated study data
├── llm_experiments.py       # OpenRouter LLM experiments
├── analysis.py              # Results analysis and visualization
results/
├── synthetic_validation/
├── retrospective_analysis/
├── llm_experiments/
├── figures/
REPORT.md                    # Final report
README.md                    # Overview
```
