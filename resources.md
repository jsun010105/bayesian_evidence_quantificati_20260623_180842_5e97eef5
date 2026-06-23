# Resources Catalog

## Summary

This document catalogs all resources gathered for the research project:
**Bayesian Evidence Quantification for Distinguishing Robustness from Experimental Weakness in AI Alignment Null Results**

---

## Papers

**Total papers downloaded: 32**

| # | Title (Short) | Authors | Year | File | Category |
|---|--------------|---------|------|------|----------|
| 1 | Efficient Bayes Factor Sensitivity Analysis | Bartoš et al. | 2026 | `papers/2604.21596_bayes_factor_sensitivity.pdf` | Bayesian Stats |
| 2 | Bayes Factors for Peri-Null Hypotheses | Ly, Wagenmakers | 2021 | `papers/2102.07162_bayes_factors_peri_null.pdf` | Bayesian Stats |
| 3 | How Accurate Are Bayes Factor Tests? | Schad et al. | 2024 | `papers/2406.08022_bayes_factor_accuracy_simulation.pdf` | Bayesian Stats |
| 4 | Can Bayes Factors Prove the Null? | — | 2019 | `papers/1907.05583_can_bayes_factors_prove_null.pdf` | Bayesian Stats |
| 5 | The Posterior Predictive Null | — | 2021 | `papers/2112.03333_posterior_predictive_null.pdf` | Bayesian Stats |
| 6 | Confirming the Null: Equivalence Testing | Dale | 2024 | `papers/2405.16331_confirming_null_equivalence_testing.pdf` | Bayesian Stats |
| 7 | Bayesian Evidence Computation | — | 2023 | `papers/2301.13783_bayesian_evidence_computation.pdf` | Bayesian Stats |
| 8 | Meta-Analysis of Bayesian Analyses | — | 2019 | `papers/1904.04484_meta_analysis_bayesian.pdf` | Bayesian Stats |
| 9 | Calibrated Model Criticism | — | 2022 | `papers/2203.15897_calibrated_model_criticism.pdf` | Bayesian Stats |
| 10 | Robustness to Changing Priors in VB | — | 2016 | `papers/1611.07469_robustness_changing_priors_vb.pdf` | Bayesian Stats |
| 11 | Optimal Stopping for Sequential BED | — | 2025 | `papers/2509.21734_optimal_stopping_bayesian_design.pdf` | Experimental Design |
| 12 | Robust Experimental Design via GBI | — | 2025 | `papers/2511.07671_robust_experimental_design.pdf` | Experimental Design |
| 13 | Deep Adaptive Design | — | 2021 | `papers/2103.02438_deep_adaptive_design.pdf` | Experimental Design |
| 14 | Alignment Verifiability (Normative Indistinguishability) | Santos-Grueiro | 2026 | `papers/2602.05656_alignment_verifiability.pdf` | Alignment Eval |
| 15 | Computational Safety: Hypothesis Testing | Chen | 2025 | `papers/2502.12445_computational_safety_hypothesis.pdf` | Alignment Eval |
| 16 | Five-Nines Reliability: LLM Evaluation | Kim et al. | 2026 | `papers/2605.11209_five_nines_reliability.pdf` | Alignment Eval |
| 17 | Activation Engineering (ActAdd) | Turner et al. | 2023 | `papers/2308.10248_activation_steering.pdf` | Activation Steering |
| 18 | Extending Activation Steering | — | 2024 | `papers/2403.05767_extending_activation_steering.pdf` | Activation Steering |
| 19 | Representation Engineering | Zou et al. | 2023 | `papers/2310.01405_representation_engineering.pdf` | Activation Steering |
| 20 | Persona Vectors vs. Targeted Steering | Kelkar et al. | 2026 | `papers/2605.21006_persona_vectors_steering.pdf` | Activation Steering |
| 21 | Depth-Wise Activation Steering | — | 2025 | `papers/2512.07667_depthwise_activation_steering.pdf` | Activation Steering |
| 22 | Sycophancy Mitigation with CAA | — | 2024 | `papers/2409.01658_sycophancy_mitigation.pdf` | Activation Steering |
| 23 | When RLHF Fails: Mechanistic Taxonomy | Abahana | 2026 | `papers/2606.03238_when_rlhf_fails.pdf` | RLHF |
| 24 | How to Evaluate Reward Models | — | 2024 | `papers/2410.14872_evaluate_reward_models_rlhf.pdf` | RLHF |
| 25 | RewardBench 2 | — | 2025 | `papers/2506.01937_rewardbench2.pdf` | RLHF |
| 26 | Reward Uncertainty in RLHF | — | 2026 | `papers/2606.09073_reward_uncertainty_rlhf.pdf` | RLHF |
| 27 | Calibration Collapse Under Sycophancy | — | 2026 | `papers/2604.10585_calibration_collapse_sycophancy.pdf` | RLHF |
| 28 | Constitutional AI | Bai et al. (Anthropic) | 2022 | `papers/2212.08073_constitutional_ai.pdf` | RLHF |
| 29 | Helpful and Harmless with RLHF | Bai et al. (Anthropic) | 2022 | `papers/2204.05862_helpful_harmless_rlhf.pdf` | RLHF |
| 30 | Reward Calibration in RLHF | — | 2024 | `papers/2410.09724_reward_calibration_rlhf.pdf` | RLHF |
| 31 | Multi-Objective Reward Modeling | — | 2024 | `papers/2406.12845_multiobjective_reward_modeling.pdf` | RLHF |
| 32 | Reward-Robust RLHF | — | 2024 | `papers/2409.15360_reward_calibration_rlhf.pdf` | RLHF |

See `papers/README.md` for detailed descriptions.

---

## Datasets

**Total datasets downloaded: 4 (samples)**

| Name | Source | Format | Task | Location |
|------|--------|--------|------|----------|
| Anthropic HH-RLHF | HuggingFace: `Anthropic/hh-rlhf` | JSON pairs | Preference learning | `datasets/anthropic_hh_rlhf/` |
| RewardBench | HuggingFace: `allenai/reward-bench` | Parquet | Reward model eval | `datasets/reward_bench/` |
| UltraFeedback Binarized | HuggingFace: `HuggingFaceH4/ultrafeedback_binarized` | Parquet | Preference learning | `datasets/ultrafeedback/` |
| HelpSteer2 | HuggingFace: `nvidia/HelpSteer2` | Parquet | Multi-attr quality | `datasets/nvidia_helpsteer2/` |

See `datasets/README.md` for detailed descriptions and download instructions.

---

## Code Repositories

**Total repositories cloned: 4**

| Name | URL | Purpose | Location |
|------|-----|---------|----------|
| RewardBench | github.com/allenai/reward-bench | Reward model evaluation framework | `code/reward_bench/` |
| CAA Sycophancy | github.com/nrimsky/CAA | Contrastive Activation Addition implementation | `code/caa_sycophancy/` |
| Alignment Benchmarks (WMDP) | github.com/centerforaisafety/wmdp | Safety benchmark + RMU unlearning | `code/alignment_benchmarks/` |
| OpenAI Evals | github.com/openai/evals | LLM evaluation framework | `code/openai_evals/` |

See `code/README.md` for detailed descriptions.

---

## Resource Gathering Notes

### Search Strategy
1. **arXiv searches** (5 rounds, 6 queries each): Targeted queries for Bayesian statistics, AI alignment, activation steering, RLHF evaluation, power analysis, sequential design
2. **Direct paper ID lookups**: 24 additional papers fetched by known arXiv IDs
3. **HuggingFace dataset API**: Searched for alignment/preference/reward datasets; verified 9+ datasets; downloaded 4 with samples
4. **GitHub search**: Cloned 4 relevant code repositories

### Selection Criteria
Papers selected for direct relevance to the research hypothesis components:
- **Bayesian evidence quantification**: Bayes factors, sensitivity analysis, peri-null, equivalence testing
- **Experimental design**: Sequential Bayesian design, optimal stopping
- **Alignment evaluation limits**: Identifiability problem, evaluation awareness, behavioral limitations
- **Intervention methodology**: Activation steering, RLHF with quantitative effect sizes
- **Measurement sensitivity**: Reliability estimation, benchmark saturation

### Challenges Encountered
1. **Paper-finder service unavailable**: Fell back to arXiv API — still found highly relevant papers
2. **Some repo URLs not found** (ActAdd main repo) — alternatives found (CAA which is related)
3. **Sycophancy-specific datasets** (PhilPapers2020 raw) require HF authentication
4. **RLHF transition data** from Abahana (2606.03238) not publicly available — may need to generate synthetic data or contact authors

### Gaps and Workarounds
1. **Missing: Published alignment null result corpus** — No systematic collection of AI alignment studies with labeled null/non-null results exists. The research will need to construct this retrospectively by extracting results from papers (especially activation steering papers 17-22).
2. **Missing: Standardized effect size estimates** from alignment interventions — Will be extracted from downloaded papers' results sections.
3. **Missing: RLHF transition-level data** — Will generate synthetic data matching statistical properties from Paper 23.

---

## Recommendations for Experiment Design

### 1. Primary Dataset(s)

**For retrospective analysis** (testing the "40% null results favor experimental insufficiency" claim):
- Extract effect size data from activation steering papers (17, 18, 19, 20, 21, 22)
- Extract pre/post intervention scores from RLHF papers (23, 24, 25, 28, 29)
- HelpSteer2 and UltraFeedback for continuous quality scores (intervention strength proxy)

**For framework validation**:
- Synthetic datasets with known true effect sizes (δ = 0, 0.1, 0.3, 0.5)
- CAA datasets for real effect size distribution estimation

### 2. Baseline Methods

- **Classical power analysis**: G*Power-style retrospective power (comparison baseline)
- **Frequentist TOST**: Two one-sided tests for equivalence (paper 6)
- **Naive Bayes factor (no sensitivity)**: BF10/BF01 without prior sensitivity analysis
- **Naive BF with default prior**: JZS Cauchy(0, √2/2) for effect size

### 3. Evaluation Metrics

**Core metrics for the Bayesian framework**:
- `BF01`: Evidence for null over H1 (primary output)
- `BF sensitivity width`: Range of BF01 across ±2 SD of prior hyperparameter
- `P(insuff | data)`: Probability of experimental insufficiency under mixture model
- `P(robust | data)`: Probability of genuine robustness

**Validation metrics**:
- Calibration: P(correctly classified | true state) via simulation
- Correlation with classical power: Spearman ρ between BF01 and retrospective power
- Interpretability: Whether diagnostics explain practitioner intuitions

### 4. Code to Adapt/Reuse

1. **`code/reward_bench/`**: Use evaluation pipeline structure for implementing the BF analysis pipeline
2. **`code/caa_sycophancy/`**: Use CAA datasets and analysis code for effect size extraction; adapt `activation_steering_interp.ipynb` for Bayesian reanalysis
3. **`code/alignment_benchmarks/`**: Extract WMDP evaluation results for retrospective analysis
4. **Python libraries**: PyMC/Bambi for Bayesian models; ArviZ for posterior analysis; SciPy for classical power analysis

### 5. Implementation Priority

```
Phase 1: Framework Implementation
  - Implement Bayesian mixture model (robust vs. insufficient)
  - Implement BF sensitivity analysis (Paper 1 method)
  - Validate on synthetic data

Phase 2: Prior Construction
  - Extract effect sizes from 22 alignment papers
  - Fit empirical prior distributions (by intervention type)
  - Conduct sensitivity analysis on prior choice

Phase 3: Retrospective Analysis
  - Apply framework to published alignment studies
  - Classify null results as "robust" vs. "insufficient evidence"
  - Compare with retrospective classical power

Phase 4: Evaluation
  - Assess >40% claim with posterior probabilities
  - Compute correlation with classical power analysis
  - Generate diagnostic visualizations
```
