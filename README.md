# Bayesian Evidence Quantification for AI Alignment Null Results

A principled Bayesian framework for distinguishing genuine model robustness from experimental weakness in AI alignment research.

## Key Findings

- **77.8% of published alignment null results** have posterior P(experimental_weakness|data) > 0.3, far exceeding the 40% threshold predicted — most alignment "null results" are inconclusive, not evidence of robustness
- **Zero published null results** constitute strong evidence of genuine robustness (BF₀₁ > 10) in our curated 27-study dataset
- **Inverse jailbreak effect** (real LLM experiment with Llama-3.1-8B): Weak educational-framing jailbreaks achieve 20% success rate vs 0% for explicit DAN-style attacks — the strong-intervention null result is correctly flagged as *inconclusive*, not robust
- **Sequential Bayesian analysis** correctly identifies insufficient evidence in all 10-trial conditions, preventing premature robustness claims
- **85% of null results** fall in the "inconclusive" zone regardless of prior specification — this is structural, not a prior artifact

## How to Reproduce

### Environment Setup

```bash
uv pip install --python .venv/bin/python numpy pandas scipy matplotlib seaborn requests
export OPENROUTER_KEY="your-key-here"
```

### Run LLM Experiments (real API calls)

```bash
.venv/bin/python src/llm_experiments.py
```

### Run Full Analysis Pipeline

```bash
.venv/bin/python src/analysis.py
```

Results are saved to `results/` and figures to `results/figures/`.

## File Structure

```
src/
  bayesian_framework.py      # Core Bayesian model (BF01, posteriors, sensitivity)
  retrospective_dataset.py   # 33-study curated alignment null result dataset
  llm_experiments.py         # OpenRouter LLM experiments (jailbreak + sycophancy)
  analysis.py                # Full analysis and visualization pipeline
results/
  summary_results.json       # Key quantitative results
  figures/                   # 5 publication-quality figures
  retrospective_analysis/    # Per-study BF01, classifications, sensitivity
  llm_experiments/           # Raw LLM trial data and Bayesian analysis
planning.md                  # Pre-registered research plan
REPORT.md                    # Full research report with methodology and results
```

## Dependencies

Python 3.12 · numpy 2.5.0 · pandas 3.0.3 · scipy 1.18.0 · matplotlib 3.11.0 · seaborn 0.13.2 · requests

## Full Report

See [REPORT.md](REPORT.md) for complete methodology, results, analysis, and discussion.
