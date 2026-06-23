# Datasets for Bayesian Evidence Quantification in AI Alignment

This directory contains datasets relevant to the research project. Data files are NOT
committed to git due to size. Follow the download instructions below.

## Overview

These datasets support three aspects of the research:
1. **Effect size priors**: Quantitative data from alignment interventions to inform Bayesian priors
2. **Retrospective analysis**: Existing alignment evaluation results with null/non-null outcomes
3. **Validation**: Controlled evaluation frameworks for validating the Bayesian framework

---

## Dataset 1: Anthropic HH-RLHF

### Overview
- **Source**: HuggingFace — `Anthropic/hh-rlhf`
- **Size**: ~170K training examples, ~8.5K test examples
- **Format**: JSON (chosen/rejected response pairs)
- **Task**: Preference learning for helpfulness and harmlessness
- **Splits**: train, test
- **License**: MIT
- **Location**: `datasets/anthropic_hh_rlhf/`

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("Anthropic/hh-rlhf")
dataset.save_to_disk("datasets/anthropic_hh_rlhf/full")
```

### Loading

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/anthropic_hh_rlhf/full")
train = dataset["train"]
```

### Schema
- `chosen`: str — preferred response (human conversation + assistant response)
- `rejected`: str — rejected response

### Relevance
Core RLHF dataset for alignment. Used in Papers 23 (When RLHF Fails), 28, 29.
Provides empirical distribution of preference signal strength.

---

## Dataset 2: RewardBench

### Overview
- **Source**: HuggingFace — `allenai/reward-bench`
- **Size**: ~2.9K evaluation examples across multiple subsets
- **Format**: Parquet
- **Task**: Reward model evaluation — assessing whether reward models correctly distinguish preferred vs. rejected responses
- **License**: ODC-BY
- **Location**: `datasets/reward_bench/`

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("allenai/reward-bench")
dataset.save_to_disk("datasets/reward_bench/full")
```

### Loading

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/reward_bench/full")
```

### Schema
- `prompt`: str — user query
- `chosen`: str — preferred response
- `chosen_model`: str — model that generated chosen
- `rejected`: str — rejected response
- `rejected_model`: str — model that generated rejected
- `subset`: str — category (chat, chat-hard, safety, reasoning)
- `id`: str — example identifier

### Relevance
Standardized reward model evaluation benchmark (Paper 25: RewardBench2). Provides
quantitative scores for reward model quality across categories—relevant for effect size
priors in the Bayesian framework and for modeling measurement sensitivity.

---

## Dataset 3: UltraFeedback Binarized

### Overview
- **Source**: HuggingFace — `HuggingFaceH4/ultrafeedback_binarized`
- **Size**: ~60K training preference pairs
- **Format**: Parquet
- **Task**: Preference learning with GPT-4 feedback scores
- **License**: MIT
- **Location**: `datasets/ultrafeedback/`

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized")
dataset.save_to_disk("datasets/ultrafeedback/full")
```

### Schema
- `prompt`: str — user query
- `prompt_id`: str — prompt identifier
- `chosen`: list — preferred response with role/content
- `rejected`: list — rejected response with role/content
- `score_chosen`: float — GPT-4 quality score for chosen response
- `score_rejected`: float — GPT-4 quality score for rejected response

### Relevance
Provides continuous preference scores (not just binary labels), ideal for estimating
effect size distributions across alignment interventions. The score gap
(score_chosen - score_rejected) provides a quantitative measure of intervention
strength—directly usable as empirical prior for the Bayesian framework.

---

## Dataset 4: HelpSteer2

### Overview
- **Source**: HuggingFace — `nvidia/HelpSteer2`
- **Size**: ~21K training examples
- **Format**: Parquet
- **Task**: Multi-attribute response quality assessment
- **License**: CC-BY-4.0
- **Location**: `datasets/nvidia_helpsteer2/`

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("nvidia/HelpSteer2")
dataset.save_to_disk("datasets/nvidia_helpsteer2/full")
```

### Schema
- `prompt`: str — user query
- `response`: str — model response
- `helpfulness`: int (0-4) — helpfulness rating
- `correctness`: int (0-4) — factual correctness
- `coherence`: int (0-4) — coherence/fluency
- `complexity`: int (0-4) — complexity of reasoning
- `verbosity`: int (0-4) — appropriate verbosity

### Observed Statistics (sample n=50)
- helpfulness: min=0, max=4, mean=3.14
- correctness: min=0, max=4, mean=3.28
- coherence: min=1, max=4, mean=3.70

### Relevance
Provides multi-dimensional quality scores directly useful for:
1. Effect size priors: Distribution of quality differences across intervention conditions
2. Measurement sensitivity: Score distributions reveal floor/ceiling effects that create null results
3. Bayesian prior construction: Empirical distribution of response quality improvements

---

## Synthetic Dataset: RLHF Transition Data

For the retrospective analysis component, we will need to construct or obtain
detailed transition-level data from RLHF experiments. Key fields needed:

- `checkpoint_id`: Checkpoint identifier
- `step`: Training step
- `proxy_reward`: Learned reward model score
- `judge_score_1`: External LLM judge score
- `judge_score_2`: Second judge score (for evaluator gaming detection)
- `baseline_reward`: Pre-intervention baseline
- `intervention_type`: Type of alignment intervention
- `intervention_strength`: Quantitative measure (e.g., steering coefficient α)
- `is_null_result`: Binary label from original study

This structure matches the data schema in Paper 23 (When RLHF Fails: 1,920 row-level transitions).

**To obtain**: Contact authors of Paper 2606.03238 for data; or generate synthetic data
matching the statistical properties described in the paper.

---

## Notes on Synthetic Data Generation

For framework validation, synthetic datasets should be generated under:
1. **True null** (effect size δ = 0): Test false positive rate
2. **Small effect** (δ = 0.1-0.2): Underpowered scenario
3. **Medium effect** (δ = 0.3-0.5): Powered scenario  
4. **Strong effect** (δ > 0.5): Easy detection

Effect sizes should be calibrated to observed distributions in alignment literature.
Based on RewardBench scores and activation steering experiments:
- Null / noise range: |δ| < 0.1
- Weak intervention range: δ ∈ [0.1, 0.3]
- Effective intervention range: δ > 0.3
