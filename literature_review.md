# Literature Review

## Research Topic
Bayesian Evidence Quantification for Distinguishing Robustness from Experimental Weakness in AI Alignment Null Results

## Research Area Overview

This research sits at the intersection of two domains: (1) Bayesian statistical methodology for null hypothesis testing and evidence quantification, and (2) AI alignment evaluation methodology. The central challenge is that null results in AI alignment research—situations where an intervention or safety measure appears to have no effect—are often ambiguous: they may indicate genuine model robustness, or they may reflect experimental insufficiency (underpowered designs, weak interventions, insensitive metrics). Classical frequentist approaches cannot distinguish these cases, but a Bayesian framework that explicitly models intervention strength, measurement sensitivity, and prior effect size distributions can.

The field is currently grappling with two related problems: (a) alignment interventions such as activation steering and RLHF fine-tuning produce highly variable results across models and evaluation protocols; (b) behavioral evaluation of alignment is theoretically limited in what it can verify, even in principle. A Bayesian framework offers principled tools for quantifying evidence and performing sensitivity analysis that could substantially improve the interpretability of null results in this domain.

---

## Key Papers

### Group 1: Bayesian Evidence Quantification and Bayes Factors

#### Paper 1: Efficient Bayes Factor Sensitivity Analysis via Posterior Density Ratios
- **Authors**: František Bartoš, Eric-Jan Wagenmakers, Maarten Marsman, Don van den Bergh
- **Year**: 2026
- **Source**: arXiv:2604.21596
- **Key Contribution**: Proposes a computationally efficient method for Bayes factor sensitivity analysis that recovers the entire sensitivity curve from a single additional model fit, rather than refitting at each hyperparameter value.
- **Methodology**: Decomposes BF at any hyperparameter γx into an "anchor" BF at reference γ0 and a Savage-Dickey density ratio (SDDR) in an extended model. Uses importance-weighted marginal density estimator (IWMDE) to estimate posterior density. Because sensitivity parameter enters only through the prior, the data likelihood cancels—reducing computation to ratio of prior density evaluations on MCMC draws.
- **Key Result**: BF10(γx) = BF10(γ0) × [p(γx|y,Hγ)/π(γx)] × [π(γ0)/p(γ0|y,Hγ)]. Substantially outperforms kernel density estimation.
- **Datasets Used**: Synthetic validation using Bayesian t-test with exact BFs; bivariate informed t-test; Bayesian model-averaged meta-analysis
- **Code Available**: Yes (R package ecosystem; details not specified in first chunk)
- **Relevance**: Directly addresses the core computational challenge of the research hypothesis—computing sensitivity curves for the evidence framework efficiently across prior hyperparameter ranges.

#### Paper 2: Bayes Factors for Peri-Null Hypotheses
- **Authors**: Alexander Ly, Eric-Jan Wagenmakers
- **Year**: 2021 (published 2022)
- **Source**: arXiv:2102.07162
- **Key Contribution**: Examines consequences of approximating point-null hypothesis with "peri-null" hypothesis (narrow distribution around null value). Shows peri-null BF = point-null BF × correction factor (itself a BF).
- **Methodology**: Derives theoretical relationship: BF_peri-null = BF_point-null × BF(H0 vs H̃0). For large N, correction term becomes influential and peri-null BF is inconsistent, approaching a limit based on ratio of prior ordinates at MLE.
- **Key Result**: For moderate sample sizes, correction is negligible; for large N, peri-null BF is bounded by prior ordinate ratio at MLE—inconsistent. This is directly relevant to the "genuine robustness vs. weak effect" distinction.
- **Relevance**: Provides theoretical foundation for using peri-null hypotheses (small practical equivalence regions) as the null, rather than point nulls—directly applicable to alignment interventions where "true zero effect" is implausible but "negligible effect for practical purposes" is the real question.

#### Paper 3: How Accurate Are Bayes Factor-Based Null Hypothesis Tests? A Simulation Study
- **Authors**: Daniel J. Schad, Martin Modrák, Shravan Vasishth
- **Year**: 2024 (updated 2025)
- **Source**: arXiv:2406.08022
- **Key Contribution**: Uses marginal simulation-based calibration (SBC) to test whether computed Bayes factors are accurate for common research designs.
- **Methodology**: Tests brms/bridgesampling implementations on three designs: (a) random effects for subjects only, (b) Latin square with crossed random effects, (c) 2x2 Latin square design. Code at OSF (https://osf.io/3g86r/).
- **Key Result**: BF estimates are accurate when bridgesampling does not issue a warning message, but can be biased and liberal when warnings occur. Provides practical guidance: check warning messages before trusting BF conclusions.
- **Relevance**: Provides calibration methodology applicable to implementing Bayesian evidence quantification for alignment experiments. Warning-checking protocol directly applicable.

#### Paper 4: Can Bayes Factors "Prove" the Null Hypothesis?
- **Authors**: (Authors from 1907.05583)
- **Year**: 2019
- **Source**: arXiv:1907.05583
- **Key Contribution**: Shows that large BF in favor of null can occur when both null and alternative have low likelihoods, and when there are ignored hypotheses—cautioning against interpreting BF in favor of null as "proof."
- **Methodology**: Theoretical analysis and simulation of scenarios where BF01 is misleadingly large.
- **Key Result**: A large BF favoring H0 does not prove H0; it merely indicates H1 (as specified) is relatively unlikely. Other hypotheses may be entirely ignored. Distinguishes between "evidence that H0 is true" vs "evidence against the specific H1."
- **Relevance**: Directly addresses a core concern in the research: what does a large BF in favor of "no alignment effect" actually mean? The paper warns that ignoring alternative hypotheses (e.g., "effect exists but intervention was too weak to detect") can produce misleading large BF01.

#### Paper 5: The Posterior Predictive Null
- **Authors**: (Authors from 2112.03333)
- **Year**: 2021
- **Source**: arXiv:2112.03333
- **Key Contribution**: Introduces "posterior predictive null" for Bayesian model criticism, adapting goodness-of-fit testing to the Bayesian framework.
- **Methodology**: Combines posterior predictive checking with null hypothesis testing. Tests whether observed data are compatible with the model's posterior predictive distribution.
- **Relevance**: Provides model checking methodology for validating the Bayesian framework used in the research. Important for verifying that prior specifications for effect sizes in alignment experiments are well-calibrated.

#### Paper 6: Confirming the Null: Remarks on Equivalence Testing and the Topology of Confirmation
- **Authors**: Reid Dale
- **Year**: 2024
- **Source**: arXiv:2405.16331
- **Key Contribution**: Develops modal logic of frequentist confirmation, showing two-sided hypotheses (θ = θ0) are not confirmable, while equivalence hypotheses (θ ∈ [L, U]) are. Frames equivalence testing as satisfying Mayo's severe testing criterion.
- **Methodology**: Modal logic with semantics based on confidence interval inclusion in hypothesis. H confirmed iff Rα ⊆ H; rejected iff Rα ⊆ Hc; inconclusive otherwise.
- **Key Result**: Point-null hypotheses have empty interior and are therefore not confirmable. Equivalence hypotheses have nonempty interior and are confirmable. TOST (two one-sided tests) satisfies severe testing desiderata.
- **Relevance**: Provides theoretical justification for using equivalence testing (practical null regions) in alignment studies rather than point nulls. Relevant to distinguishing "genuinely robust" (effect in equivalence interval) from "underpowered" (inconclusive result).

#### Paper 7: An Analytical Approach to Bayesian Evidence Computation
- **Authors**: (Authors from 2301.13783)
- **Year**: 2023
- **Source**: arXiv:2301.13783
- **Key Contribution**: Analytical approach to computing Bayesian evidence (marginal likelihood) for model selection, addressing computational bottlenecks.
- **Relevance**: Provides computational methods for Bayesian evidence computation applicable to the research framework.

#### Paper 8: Meta-Analysis of Bayesian Analyses
- **Authors**: (Authors from 1904.04484)
- **Year**: 2019
- **Source**: arXiv:1904.04484
- **Key Contribution**: Addresses how to combine results from multiple related Bayesian statistical analyses, where the natural output is a posterior distribution.
- **Relevance**: Enables combining evidence across multiple alignment studies in the retrospective analysis component of the research hypothesis.

#### Paper 9: Calibrated Model Criticism Using Split Predictive Checks
- **Authors**: (Authors from 2203.15897)
- **Year**: 2022
- **Source**: arXiv:2203.15897
- **Key Contribution**: Introduces split predictive checks that are well-calibrated (avoiding the double use of data), allowing valid Bayesian model criticism.
- **Relevance**: Provides calibrated model checking for validating prior specifications in the Bayesian alignment evaluation framework.

#### Paper 10: Fast Measurements of Robustness to Changing Priors in Variational Bayes
- **Authors**: (Authors from 1611.07469)
- **Year**: 2016
- **Source**: arXiv:1611.07469
- **Key Contribution**: Methods for fast measurement of posterior sensitivity to prior changes in variational Bayes approximations.
- **Relevance**: Prior robustness analysis methodology applicable to the research's sensitivity analysis component.

---

### Group 2: Experimental Design for Sequential Bayesian Testing

#### Paper 11: Optimal Stopping for Sequential Bayesian Experimental Design
- **Authors**: (Authors from 2509.21734)
- **Year**: 2025
- **Source**: arXiv:2509.21734
- **Key Contribution**: Addresses optimal stopping rules for sequential Bayesian experimental design, moving beyond fixed-horizon formulations.
- **Methodology**: Policy optimization problem for adaptive stopping where experiments terminate when sufficient evidence is accumulated.
- **Relevance**: Provides the sequential design component for the research: when to stop collecting alignment evaluation data based on accumulated Bayesian evidence.

#### Paper 12: Robust Experimental Design via Generalised Bayesian Inference
- **Authors**: (Authors from 2511.07671)
- **Year**: 2025
- **Source**: arXiv:2511.07671
- **Key Contribution**: Robust optimal experimental design using generalised Bayesian inference to handle model misspecification.
- **Relevance**: Addresses robustness of experimental design under model misspecification—important for alignment experiments where the true data-generating process may deviate from assumed models.

#### Paper 13: Deep Adaptive Design: Amortizing Sequential Bayesian Experimental Design
- **Authors**: (Authors from 2103.02438)
- **Year**: 2021
- **Source**: arXiv:2103.02438
- **Key Contribution**: DAD method for amortizing sequential Bayesian experimental design using neural networks, enabling real-time experiment execution.
- **Methodology**: Policy network trained via variational bounds on expected information gain; allows rapid sequential design without per-experiment optimization.
- **Relevance**: Provides scalable sequential design methodology for adaptively collecting more alignment evaluation data when evidence is inconclusive.

---

### Group 3: AI Alignment Evaluation — Behavioral Assessment Limits

#### Paper 14: On the Limits of Behavioral Alignment: Formal Verifiability and the Problem of Normative Indistinguishability
- **Authors**: Igor Santos-Grueiro
- **Year**: 2026
- **Source**: arXiv:2602.05656
- **Key Contribution**: Formalizes alignment evaluation as identifiability problem under partial observability. Proves conditional impossibility result: under "evaluation-aware" agents, observed compliance does not uniquely identify latent alignment.
- **Methodology**: Statistical identifiability framework; "Chameleon" construction using Llama-3.2-3B demonstrates evaluation-dependent non-identifiability; uses conditional fine-tuning.
- **Key Result**: Behavioral benchmarks provide necessary but insufficient evidence for latent alignment under evaluation awareness. Observed compliance identifies only an equivalence class of conditionally compliant policies.
- **Datasets**: Llama-3.2-3B instruction-tuned model; custom evaluation protocols
- **Relevance**: Directly motivates the research hypothesis: behavioral null results in alignment studies are fundamentally ambiguous about latent properties. The Bayesian framework proposed in the research hypothesis is needed to quantify this ambiguity.

#### Paper 15: Computational Safety for Generative AI: A Hypothesis Testing Perspective
- **Authors**: Pin-Yu Chen (IBM Research)
- **Year**: 2025 (updated 2026)
- **Source**: arXiv:2502.12445 (ICML 2026 Workshop on Hypothesis Testing)
- **Key Contribution**: Frames AI safety problems as hypothesis testing tasks in signal processing. Defines "computational safety" as the set of safety problems formatable as hypothesis testing.
- **Methodology**: Signal processing framework covering sensitivity analysis for jailbreak detection and statistical signal processing for AI-generated content detection. Judge functions J(x,y) → {0,1} serve as proxy for ground-truth.
- **Key Result**: Many alignment safety problems can be reformulated as binary hypothesis testing (H1: unsafe/aligned vs H0: safe/unaligned). Sensitivity analysis and subspace projection applicable.
- **Relevance**: Provides the hypothesis testing language for formalizing alignment intervention effects. Directly bridges signal processing and the Bayesian evidence quantification approach.

#### Paper 16: Measuring Five-Nines Reliability: Sample-Efficient LLM Evaluation in Saturated Benchmarks
- **Authors**: Eungyeup Kim, Chenchen Gu, Vashisth Tiwari, J. Zico Kolter (CMU)
- **Year**: 2026
- **Source**: arXiv:2605.11209
- **Key Contribution**: Shows that models with near-identical accuracy on standard benchmarks can differ substantially in failure rates at the reliability level. Proposes CEM-based importance sampling for sample-efficient rare failure estimation.
- **Methodology**: Cross-entropy method (CEM) learns sampling distribution concentrated on failure-prone inputs. Achieves up to 156.22× reduction in required inferences vs. uniform sampling.
- **Key Result**: Models exceeding 99.9% accuracy still differ by up to 2.4× in estimated failure rates. Benchmark saturation obscures reliability differences. Failures exhibit systematic input-dependent patterns.
- **Datasets**: Parameterized GSM8K templates; Qwen2.5-Math-7B-Instruct, gpt-oss-20b-low, Gemini 2.5 Flash Lite
- **Relevance**: Measurement sensitivity problem: saturated benchmarks make null results essentially uninformative about actual robustness. The CEM approach could be adapted for alignment evaluation. Underscores importance of the research hypothesis—"null result" in saturated benchmark ≠ genuine robustness.

---

### Group 4: Activation Steering and Alignment Interventions

#### Paper 17: Steering Language Models With Activation Engineering
- **Authors**: Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez, Ulisse Mini, Monte MacDiarmid
- **Year**: 2023 (updated 2024)
- **Source**: arXiv:2308.10248
- **Key Contribution**: Introduces activation engineering and the ActAdd technique—SOTA on toxicity reduction and sentiment control using contrast-pair steering vectors.
- **Methodology**: Activation Addition (ActAdd): contrasts activations on prompt pairs (e.g., "Love" vs "Hate") to compute steering vector; adds at inference time without gradient descent. Works with single pair of data points.
- **Key Result**: SOTA sentiment control and detoxification on LLaMA-3 and OPT. Preserves off-target task performance. Lightweight—no optimization required.
- **Relevance**: Primary class of alignment intervention whose null results need Bayesian analysis. ActAdd's single-pair limitation and lack of explicit effect size estimates makes power analysis challenging.

#### Paper 18: Extending Activation Steering to Broad Skills and Multiple Behaviours
- **Authors**: (Authors from 2403.05767)
- **Year**: 2024
- **Source**: arXiv:2403.05767
- **Key Contribution**: Extends activation steering to broad skills and dangerous capabilities, evaluating effectiveness at reducing risks from increasingly capable models.
- **Relevance**: Provides larger-scale evaluation of activation steering interventions—source of null results amenable to Bayesian reanalysis.

#### Paper 19: Representation Engineering: A Top-Down Approach to AI Transparency
- **Authors**: (Authors from 2310.01405, Zou et al.)
- **Year**: 2023
- **Source**: arXiv:2310.01405
- **Key Contribution**: Introduces representation engineering (RepE) as a framework for AI transparency that identifies and manipulates high-level representations in LLMs.
- **Methodology**: Uses principal component analysis on contrast activations to find representation directions. Develops reading vectors (linear probes) and control vectors (steering).
- **Relevance**: Foundational paper for activation engineering interventions. Provides the methodological basis for interventions whose null results the research will analyze.

#### Paper 20: Playing Devil's Advocate: Off-the-Shelf Persona Vectors Rival Targeted Steering for Sycophancy
- **Authors**: Ishaan Kelkar, Vikram Kakaria, Nebras Alam, Madhur Panwar, Vasu Sharma, Maheep Chaudhary
- **Year**: 2026
- **Source**: arXiv:2605.21006 (EIML@ICML 2026 Workshop)
- **Key Contribution**: Critical-role persona vectors achieve 68–98% of Contrastive Activation Addition's (CAA) effect on sycophancy without any sycophancy-specific training data. Conformist roles produce weak, heterogeneous effects. Role vectors are nearly orthogonal to CAA direction (|cos| < 0.17).
- **Methodology**: Wilcoxon signed-rank tests with Holm correction; 24-condition experiment on Gemma 2 27B and Qwen 3 32B; PhilPapers benchmark; tune/test split; coefficient sweep.
- **Key Result**: Sycophancy is better understood as a persona-level property than a single steerable direction. Effect asymmetry: steering toward critical personas reduces sycophancy; steering toward conformist personas does not increase it.
- **Relevance**: Excellent example of research with complex null and non-null results requiring Bayesian interpretation—conformist roles yield near-zero effects that could be true null or underpowered. The asymmetry finding illustrates exactly the "robustness vs. weak intervention" ambiguity the research addresses.

#### Paper 21: Depth-Wise Activation Steering for Honest Language Models
- **Authors**: (Authors from 2512.07667)
- **Year**: 2025
- **Source**: arXiv:2512.07667
- **Key Contribution**: Layer-specific activation steering for honesty; shows that depth-wise targeting matters for effectiveness.
- **Relevance**: Illustrates that intervention strength varies systematically with steering layer choice—a key component for modeling intervention strength in the Bayesian framework.

#### Paper 22: From Yes-Men to Truth-Tellers: Addressing Sycophancy in Large Language Models with Contrastive Activation Addition
- **Authors**: (Authors from 2409.01658)
- **Year**: 2024
- **Source**: arXiv:2409.01658
- **Key Contribution**: Uses CAA to mitigate sycophancy, demonstrating effectiveness on factual consistency metrics.
- **Relevance**: Another activation steering intervention with quantitative effect sizes that could serve as empirical prior for the Bayesian framework.

---

### Group 5: RLHF Evaluation and Reward Model Reliability

#### Paper 23: When RLHF Fails: A Mechanistic Taxonomy of Reward Hacking, Collapse, and Evaluator Gaming
- **Authors**: Zelalem Abahana
- **Year**: 2026
- **Source**: arXiv:2606.03238
- **Key Contribution**: Transition-based taxonomy of RLHF failures: stable alignment, reward hacking, optimization collapse, proxy under-alignment, conservative stagnation, evaluator gaming. Row-level analysis reveals failures hidden by checkpoint averages.
- **Methodology**: Compact RLHF pipeline with PPO, DPO, UP-PPO; MC dropout uncertainty; two LLM judges; directional classification of transitions. 61 checkpoint rows, 1,920 transitions; pre-transition logistic model (AUC 0.821) predicts future reward hacking.
- **Key Result**: Aggressive PPO: 14.45% reward hacking rate (95% CI: 10.16–18.75%). Row-level analysis finds localized failures that checkpoint averages miss in 3/12 settings.
- **Relevance**: Empirical demonstration that aggregated null results (checkpoint-level averages) mask important heterogeneity. The research's Bayesian framework needs to handle this structured variation. Also provides a dataset of RLHF transitions with proxy/judge scores.

#### Paper 24: How to Evaluate Reward Models for RLHF
- **Authors**: (Authors from 2410.14872)
- **Year**: 2024
- **Source**: arXiv:2410.14872
- **Key Contribution**: Introduces benchmark for reward models focused on their ability to produce strong LLMs through RLHF, including gold-standard approach.
- **Relevance**: Provides evaluation methodology for reward model quality—key measurement instrument whose sensitivity affects the Bayesian analysis.

#### Paper 25: RewardBench 2: Advancing Reward Model Evaluation
- **Authors**: (Authors from 2506.01937)
- **Year**: 2025
- **Source**: arXiv:2506.01937
- **Key Contribution**: Second version of RewardBench extending reward model evaluation to nuanced preference, reasoning, and safety.
- **Relevance**: Updated evaluation framework providing quantitative scores useful as effect size priors in the Bayesian framework.

#### Paper 26: A Unifying Lens on Reward Uncertainty in RLHF
- **Authors**: (Authors from 2606.09073)
- **Year**: 2026
- **Source**: arXiv:2606.09073
- **Key Contribution**: Framework for understanding reward uncertainty across RLHF training phases; connects to reward hacking.
- **Relevance**: Reward model uncertainty directly maps to measurement sensitivity in the Bayesian evidence framework.

#### Paper 27: Calibration Collapse Under Sycophancy Fine-Tuning: How Reward Hacking Breaks Uncertainty
- **Authors**: (Authors from 2604.10585)
- **Year**: 2026
- **Source**: arXiv:2604.10585
- **Key Contribution**: Shows that sycophancy fine-tuning causes calibration collapse in LLMs, breaking uncertainty quantification.
- **Relevance**: Important case study where intervention null results could be confounded with calibration collapse—the Bayesian framework needs to account for this.

#### Paper 28: Constitutional AI: Harmlessness from AI Feedback
- **Authors**: Anthropic (Bai et al.)
- **Year**: 2022
- **Source**: arXiv:2212.08073
- **Key Contribution**: Constitutional AI using AI-generated feedback for harmlessness training; RLHF from AI feedback.
- **Relevance**: Foundational alignment methodology whose evaluation results provide prior effect sizes for the Bayesian framework.

#### Paper 29: Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback
- **Authors**: Anthropic (Bai et al.)
- **Year**: 2022
- **Source**: arXiv:2204.05862
- **Key Contribution**: RLHF for helpfulness and harmlessness; finds alignment training improves performance while maintaining safety.
- **Relevance**: Core alignment training paper providing baseline effect sizes.

---

## Common Methodologies

### Bayesian Evidence Quantification
- **Bayes factors** (Jeffreys framework): Most common approach for quantifying evidence for null vs. alternative. Requires prior specification for alternative.
- **Savage-Dickey density ratio**: Efficient computation for nested models. Used in Papers 1, 2.
- **Bridge sampling**: General marginal likelihood computation. Used in Paper 3 (brms/bridgesampling).
- **Sensitivity analysis**: Examining BF as function of prior hyperparameters. Papers 1, 10.
- **Sequential/adaptive**: Sequential Bayesian testing with optional stopping. Papers 11-13.

### Alignment Intervention Evaluation
- **Activation steering (ActAdd/CAA)**: Contrast pair activations; adds steering vector at inference. Papers 17-22.
- **RLHF/DPO**: Preference learning from human or AI feedback. Papers 23-29.
- **Behavioral benchmarks**: Finite evaluation protocols. Paper 14 shows fundamental limits.
- **LLM-as-judge**: Secondary evaluation metric. Papers 23, 15.

### Null Result Interpretation
- **Frequentist**: p > α → "failure to reject"—cannot confirm robustness
- **Bayesian**: Large BF01 → evidence for null but only relative to specified H1
- **Equivalence testing (TOST)**: Confirms null within practical equivalence region. Paper 6.
- **Power analysis**: Post-hoc power as continuous function of effect size. Relates to Papers 2, 4.

---

## Standard Baselines

### Bayesian Testing
- **JZS Bayesian t-test** (Jeffreys-Zellner-Siow): Standard default BF for simple comparisons; Cauchy prior on effect size
- **brms/bridgesampling**: Standard pipeline for complex BF computation (R ecosystem)
- **BayesFactor R package**: For simpler factorial designs
- **Stan/PyMC**: For custom Bayesian models

### Alignment Evaluation
- **CAA (Contrastive Activation Addition)**: 2,000+ contrast pairs; Rimsky et al. 2024 formalization
- **PhilPapers benchmark**: Forced-choice sycophancy evaluation (Perez et al. 2022)
- **RewardBench / RewardBench2**: Standardized reward model evaluation
- **Anthropic HH-RLHF dataset**: Standard prompts for helpfulness/harmlessness

---

## Evaluation Metrics

### Bayesian Analysis
- **Bayes Factor (BF10/BF01)**: Primary evidence quantification; scale: <1/3 = moderate for H0, >3 = moderate for H1
- **Posterior probability P(H0|data)**: Given prior model odds
- **BF sensitivity curve**: BF as function of prior hyperparameter
- **Evidence robustness**: Width of BF sensitivity range
- **Posterior predictive p-value**: For model checking

### Alignment Interventions
- **Sycophancy rate / Δlogit**: Binary rate and log-probability difference
- **Reward model score**: Proxy quality metric
- **LLM judge scores**: External quality measure
- **Refusal rate**: For safety interventions
- **Task performance (off-target)**: Side effect measurement

---

## Datasets in the Literature

| Dataset | Papers | Task | Size |
|---------|--------|------|------|
| PhilPapers2020 benchmark | 20, 22 | Sycophancy evaluation | 300 questions × 2 orderings |
| Anthropic HH-RLHF prompts | 23, 28, 29 | Helpfulness/harmlessness | Large-scale |
| RewardBench2 | 25 | Reward model evaluation | Structured benchmark |
| Parameterized GSM8K | 16 | Reliability estimation | Combinatorial parameter space |
| Custom RLHF pipeline | 23 | RLHF failure analysis | 61 checkpoints, 1,920 transitions |

---

## Gaps and Opportunities

### Gap 1: No existing Bayesian framework for alignment null results
- Current alignment papers report binary "null/not null" without quantifying evidence
- No systematic retrospective analysis framework exists
- **Opportunity**: Develop and validate the Bayesian framework the hypothesis proposes

### Gap 2: Intervention strength is rarely operationalized
- ActAdd/CAA results vary by steering coefficient α but this is rarely treated as a continuous intervention strength parameter
- Prior distributions over reasonable intervention strengths are not elicited
- **Opportunity**: Build empirical prior distributions from published steering experiments

### Gap 3: Measurement sensitivity not modeled
- Five-nines paper (16) shows saturated benchmarks obscure reliability differences
- Existing studies don't account for measurement insensitivity in their null result interpretation
- **Opportunity**: Incorporate benchmark sensitivity (e.g., CEM-estimated failure rates) into the Bayesian evidence model

### Gap 4: Missing connection between identifiability limits and evidence quantification
- Santos-Grueiro (14) shows behavioral evaluation cannot uniquely identify latent alignment
- This theoretical limit is not translated into quantitative uncertainty in current practice
- **Opportunity**: Use the normative indistinguishability framework to place principled bounds on achievable evidence

### Gap 5: RLHF failure modes conflate null results
- Abahana (23) shows 6 distinct failure modes that checkpoint-level analysis misses
- These would all appear as "null results" in naive analysis
- **Opportunity**: The Bayesian taxonomy should distinguish these failure modes

---

## Recommendations for Our Experiment

### Recommended datasets
1. **Activation steering benchmark data**: Collect/extract from ActAdd (2308.10248), representation engineering (2310.01405), extending activation steering (2403.05767) papers—these have quantitative effect sizes amenable to retrospective Bayesian analysis
2. **RLHF failure data from Abahana (2606.03238)**: 1,920 row-level transitions with proxy/judge scores—ideal for retrospective analysis
3. **PhilPapers sycophancy data**: Well-controlled forced-choice benchmark with reproducible evaluation methodology
4. **Simulated datasets**: Generate synthetic alignment experiment results under known true effect sizes to validate the Bayesian framework's ability to distinguish robustness from underpowering

### Recommended baselines
1. **Classical power analysis**: Retrospective power analysis for existing null results (lower bound comparison)
2. **Frequentist equivalence testing (TOST)**: Compare Bayesian framework with TOST for distinguishing robustness
3. **Naive BF01** (without prior sensitivity analysis): Compare with sensitivity-aware approach

### Recommended metrics
1. **BF01** (evidence for null): Primary output of the framework
2. **BF sensitivity range**: Width of BF curve across prior hyperparameter range—measure of evidence robustness
3. **P(experimental insufficiency | data)**: Key diagnostic probability under mixture prior model
4. **Correlation with classical power**: Validation metric against retrospective power analysis
5. **Calibration**: P(correctly classified as robust | truly robust) under simulation

### Methodological considerations
1. **Prior elicitation for alignment effects**: Need empirical priors on effect sizes from published steering/RLHF papers; Cauchy(0, r) with r calibrated from meta-analysis
2. **Peri-null vs. point-null**: Use peri-null hypotheses (practical equivalence regions) rather than point nulls—justified by Paper 6 and more appropriate for alignment
3. **Hierarchical model**: Pool effect sizes across alignment intervention types; estimate intervention-type-level parameters
4. **Intervention strength parameterization**: Explicitly model steering coefficient α as continuous parameter affecting expected effect size
5. **Measurement sensitivity model**: Include benchmark sensitivity (detection probability at each effect size) as model component
6. **Sequential extension**: Build in optional stopping rules (Paper 11) for prospective use of the framework
