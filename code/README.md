# Cloned Code Repositories

4 repositories cloned for the Bayesian Evidence Quantification research project.

---

## Repo 1: RewardBench (allenai/reward-bench)
- **URL**: https://github.com/allenai/reward-bench
- **Purpose**: Standardized reward model evaluation framework
- **Location**: `code/reward_bench/`
- **Key files**:
  - `rewardbench/` — core evaluation code
  - `scripts/` — evaluation scripts
  - `analysis/` — analysis notebooks
  - `paper-v2.pdf` — RewardBench paper
- **License**: Apache 2.0
- **Installation**:
  ```bash
  cd code/reward_bench
  pip install -e .
  ```
- **Notes**: Directly applicable for evaluating reward model quality as a measurement sensitivity proxy in the Bayesian framework. The structured benchmark categories (chat, chat-hard, safety, reasoning) map to different intervention domains.
- **How to use**:
  ```python
  from rewardbench import load_eval_dataset
  dataset = load_eval_dataset("reward-bench")
  # Compute scores for reward models
  ```

---

## Repo 2: CAA (nrimsky/CAA) — Contrastive Activation Addition for Sycophancy
- **URL**: https://github.com/nrimsky/CAA
- **Purpose**: Implementation of Contrastive Activation Addition (CAA) for steering LLMs via activation-space editing
- **Location**: `code/caa_sycophancy/`
- **Key files**:
  - `activation_steering_interp.ipynb` — main notebook for CAA experiments
  - `analysis/` — result analysis code
  - `README.md` — dataset sources and setup
- **Notes**: Core implementation of the standard alignment intervention methodology. The Contrastive Activation Addition (CAA) approach extracts mean-difference vectors between sycophantic/honest response activations and adds them at inference time. Provides:
  - Multiple behavioral evaluation datasets (coordination, corrigibility, hallucination, myopia, survival instinct, sycophancy)
  - Effect size data from steering experiments (coefficient α → behavioral change)
  - Contrast pair datasets from Anthropic model-written evals
- **Datasets used** (from Anthropic model-written-evals):
  - `coordinate-other-ais.jsonl`
  - `corrigible-neutral-HHH.jsonl`
  - `myopic-reward.jsonl`
  - `survival-instinct.jsonl`
  - Hallucination dataset (GPT-4 generated)
- **Relevance**: Primary source of effect size data for building empirical priors in the Bayesian framework. The relationship between steering coefficient α and behavioral change provides the intervention strength model.

---

## Repo 3: Alignment Benchmarks (centerforaisafety/wmdp)
- **URL**: https://github.com/centerforaisafety/wmdp
- **Purpose**: WMDP (Weapons of Mass Destruction Proxy) benchmark + RMU (Representation Misdirection for Unlearning) method
- **Location**: `code/alignment_benchmarks/`
- **Key files**:
  - `rmu/` — Representation Misdirection for Unlearning implementation
  - `data/` — benchmark data
  - `run_rmu_*.ipynb` — notebooks for different models (Mixtral, Yi, Zephyr)
- **Notes**: Provides examples of safety intervention evaluations with known null and non-null results. The unlearning experiments provide:
  - Pre/post intervention performance metrics
  - Alignment-capability tradeoff data ("alignment tax")
  - Multi-model evaluation across different architectures
- **Relevance**: Alignment interventions with structured evaluation across multiple models—ideal source of retrospective analysis data for the research hypothesis.

---

## Repo 4: OpenAI Evals
- **URL**: https://github.com/openai/evals
- **Purpose**: Framework for evaluating LLMs and alignment properties
- **Location**: `code/openai_evals/`
- **Notes**: Comprehensive evaluation framework with hundreds of evaluation templates. Provides standardized evaluation methodology for alignment experiments.
- **Relevance**: Evaluation infrastructure that can be adapted for collecting structured alignment evaluation data for the retrospective analysis.

---

## Notes on Missing Repositories

The following repositories were searched but not found:
- ActAdd official repo (Turner et al. 2308.10248) — code distributed via author sites
- Representation Engineering official repo (Zou et al. 2310.01405) — check https://github.com/andyzoujm/representation-engineering

### To clone representation-engineering:
```bash
cd code
git clone --depth=1 https://github.com/andyzoujm/representation-engineering.git representation_engineering
```

### For Bayesian analysis tools:
```python
# Key R packages (run in R environment)
install.packages("BayesFactor")  # JZS Bayes factors
install.packages("brms")          # Bayesian regression models
install.packages("bridgesampling") # Marginal likelihood estimation
install.packages("bayestestR")    # Bayesian effect sizes and CI

# Python equivalents
pip install pymc  # PyMC for MCMC
pip install arviz  # Bayesian analysis visualization
pip install bambi  # Bayesian model-building
```
