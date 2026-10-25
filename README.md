# TailorMoE

**Code explanations, tailored to every developer.**

TailorMoE learns *how different developers like their code explained* from just a handful of pairwise choices, and then picks the explanation style that suits each person best. It combines a **Mixture of Sparse Bradley–Terry Experts** (trained with EM) with interpretable text features, so every personalisation decision can be traced back to human-readable reasons such as "this user dislikes jargon" or "this user wants short, code-focused answers".

> **Notebook:** `Personalised_Explanation_Preferences_MoE.ipynb`
> **Status:** Research prototype / proposal-stage experiment. All results below use **simulated developers** (see [Limitations](#limitations)).

---

## Table of Contents

1. [Motivation](#motivation)
2. [Key Ideas](#key-ideas)
3. [Pipeline Overview](#pipeline-overview)
4. [Models Used](#models-used)
5. [Interpretable Features](#interpretable-features)
6. [Simulated Developer Types](#simulated-developer-types)
7. [The Mixture-of-Experts Reward Model](#the-mixture-of-experts-reward-model)
8. [Results](#results)
9. [Getting Started](#getting-started)
10. [Configuration](#configuration)
11. [Using Real Preference Data](#using-real-preference-data)
12. [Output Files](#output-files)
13. [Limitations](#limitations)
14. [Future Work](#future-work)

---

## Motivation

Ask five developers to explain the same function and you will get five different styles. A beginner wants step-by-step walkthroughs with examples and plain language. A senior engineer wants two precise sentences that reference the code directly. A cautious reviewer wants claims that are tied to the code and not over-confident.

Most AI assistants use **one global notion of a "good" explanation**. TailorMoE asks a different question:

> *Can we quickly figure out which kind of reader we are talking to, and adapt the explanation, using only a few preference comparisons, while staying interpretable?*

---

## Key Ideas

- **Interpretable reward model.** Preferences are modelled as sparse weights over 9 readable text features, not a black-box embedding.
- **Mixture of experts.** Each expert captures one "taste profile". Users are softly assigned to experts, so the model discovers user groups on its own.
- **Few-shot routing.** For a new developer, a posterior over experts is computed from only *K* pairwise choices (K = 1 to 16).
- **Faithful explanations.** Explanations of the reward model's decisions are tested with a removal-based faithfulness (comprehensiveness) metric against baselines.
- **Personalised selection.** For each code snippet, the system picks the explanation variant that the routed reward model likes best.

---

## Pipeline Overview

```
 24 Python snippets  x  8 instruction variants
              │
              ▼
 Qwen2.5-Coder-1.5B-Instruct  ──►  192 explanations
              │
              ▼
 Feature extraction (9 interpretable features, incl. LM surprisal)
              │
              ▼
 Simulated developers (novice / expert / verifier)
 answer pairwise comparisons via Bradley–Terry
              │
              ▼
 EM-trained Mixture of Sparse (L1) Logistic Experts
              │
     ┌────────┼───────────────────────┐
     ▼        ▼                       ▼
 Expert     Few-shot adaptation    Faithfulness
 recovery   to new developers      of explanations
                     │
                     ▼
        Personalised explanation selection
```

### Step by step

| # | Stage | What happens |
|---|-------|--------------|
| 1 | **Setup** | Installs `transformers`, `accelerate`, `scikit-learn`, `scipy` if missing; fixes seeds (`SEED = 11`); picks GPU or CPU. |
| 2 | **Stimuli** | 24 short Python functions (e.g. `binary_search`, `fib`, `is_prime`, `merge_sorted`) are validated with `compile()`. |
| 3 | **Instruction design** | A 2 x 2 x 2 factorial: **detail** (brief / detailed) x **example** (none / one) x **confidence** (hedged / assertive) = 8 variants per snippet, giving **192 explanations**. |
| 4 | **Generation** | Greedy decoding with Qwen2.5-Coder-1.5B-Instruct, max 160 new tokens. |
| 5 | **Feature extraction** | Nine interpretable features per explanation, standardised (z-scored). |
| 6 | **Manipulation check** | Cohen's *d* shows whether each instruction actually changed the features it should. |
| 7 | **User simulation** | 120 training and 60 test developers, 40 pairwise comparisons each, generated from hidden type-specific weights. |
| 8 | **MoE training** | EM over sparse logistic experts; number of experts chosen on held-out log-likelihood. |
| 9 | **Evaluation** | Expert recovery, few-shot accuracy, faithfulness, and personalised-selection regret, all with bootstrap 95% CIs. |

---

## Models Used

| Role | Model |
|------|-------|
| Explanation generator | `Qwen/Qwen2.5-Coder-1.5B-Instruct` |
| Reader LM (readability via surprisal) | `Qwen/Qwen2.5-Coder-0.5B` |

---

## Interpretable Features

Each explanation is described by nine features:

| Feature | Meaning |
|---------|---------|
| `log_words` | Length (log scale) |
| `sentence_len` | Average words per sentence |
| `jargon` | Share of technical vocabulary |
| `hedges` | Hedging words per word ("likely", "may", ...) |
| `certainty` | Confidence markers per word ("definitely", "always", ...) |
| `examples` | Count of example markers and concrete calls |
| `steps` | Count of numbered or bulleted steps |
| `code_refs` | Share of code identifiers that the explanation mentions |
| `surprisal` | Mean bits/token under the reader LM (higher = harder to read) |

### Manipulation check (Cohen's *d*)

The instructions behaved as intended, with one honest exception to note:

| Feature | confidence | detail | example |
|---------|-----------:|-------:|--------:|
| log_words | -0.03 | **1.93** | 0.10 |
| sentence_len | 0.11 | **-1.67** | -0.11 |
| jargon | -0.03 | 0.51 | -0.02 |
| hedges | **-0.33** | -0.19 | -0.03 |
| certainty | -0.07 | 0.14 | 0.48 |
| examples | -0.08 | -0.64 | **1.07** |
| steps | 0.07 | **3.09** | -0.05 |
| code_refs | 0.08 | 1.18 | 0.24 |
| surprisal | -0.13 | -0.96 | -0.48 |

- The **detail** instruction strongly changes length and steps.
- The **example** instruction strongly increases example markers.
- The **confidence** instruction has only a small effect on `hedges`/`certainty`; the 1.5B model follows hedging instructions weakly. Hedge and certainty words are very rare in the generated text.

---

## Simulated Developer Types

Preferences are simulated with hidden weights over the standardised features (plus per-user Gaussian noise, SD 0.25) and choices are drawn from a **Bradley–Terry** model with sharpness `BETA = 2.0`.

| Type | What they prefer |
|------|------------------|
| **Novice** | Longer, example-rich, step-by-step, low-jargon, easy-to-read explanations |
| **Expert** | Short, technical explanations that point directly at the code |
| **Verifier** | Low over-confidence, some hedging, explanations tied to the code, with examples |

Training split: 46 experts, 42 novices, 32 verifiers (4,800 training comparisons). Test: 60 new developers (2,400 comparisons).

---

## The Mixture-of-Experts Reward Model

For a comparison between explanations *A* and *B* with feature difference `d = x_A - x_B`:

```
P(A preferred | user, expert k) = sigmoid( d · w_k )
P(A preferred | user)           = Σ_k r_k · sigmoid( d · w_k )
```

- **Experts** `w_k`: L1-regularised (sparse) logistic regressions, fit with `fit_intercept=False`.
- **Responsibilities** `r_k`: a user's posterior over experts.
- **Training:** EM with 30 iterations. The M-step fits one weighted sparse logistic model per expert; the E-step computes each user's posterior from the summed log-likelihood of all their comparisons.
- **Routing a new user:** `route()` combines the prior mixture weights with the likelihood of the user's *K* observed comparisons.
- **Model selection:** the number of experts (1 to 5) is chosen by held-out log-likelihood on validation users (route with the first half of their comparisons, score the second half).

---

## Results

All numbers come from the saved notebook run (Google Colab, CUDA GPU; generating the 192 explanations took about 1,096 s).

### 1. Number of experts

| Experts | Held-out log-likelihood |
|--------:|------------------------:|
| 1 | -377.1 |
| 2 | -291.9 |
| 3 | -265.4 |
| **4** | **-254.0** |
| 5 | -266.4 |

**4 experts were selected.** One of them received a negligible mixture weight (about 0.017), so effectively three main experts emerged, matching the three simulated types.

### 2. Recovery of hidden developer types

- **Adjusted Rand Index:** **0.952** (users to experts vs. true types)
- Cosine similarity between learned experts and true type weights:

| Learned expert | Matched type | Cosine |
|---------------:|--------------|-------:|
| expert_0 | verifier | 0.992 |
| expert_1 | expert | 0.996 |
| expert_3 | novice | 0.985 |

### 3. Few-shot adaptation to new developers

Accuracy at predicting a new developer's held-out choices after seeing *K* of their comparisons (95% bootstrap CI):

| K | Mixture of experts | Per-user model | One global model | Oracle (true weights) |
|--:|-------------------:|---------------:|-----------------:|----------------------:|
| 1 | **0.682** | 0.531 | 0.620 | 0.835 |
| 2 | **0.693** | 0.575 | 0.620 | 0.835 |
| 4 | **0.716** | 0.655 | 0.620 | 0.835 |
| 8 | **0.768** | 0.671 | 0.620 | 0.835 |
| 16 | **0.772** | 0.724 | 0.620 | 0.835 |

The mixture of experts is best at every K, and the gap over a per-user model is largest when data is scarce (K = 1 to 4). By type at K = 4, accuracy is 0.848 for novices, 0.694 for verifiers, and 0.646 for experts (global model: 0.660, 0.612, 0.599).

### 4. Faithfulness of explanations

Change in the predicted preference after removing the top-*m* features named by each explanation method (higher = more faithful):

| Method | m = 1 | m = 2 | m = 3 |
|--------|------:|------:|------:|
| **Contribution contrast** (per decision) | **0.255** | **0.218** | **0.227** |
| Routing weights only | 0.144 | 0.189 | 0.195 |
| Random (control) | 0.056 | 0.090 | 0.123 |

Per-decision contribution explanations are clearly more faithful than routing-only explanations or random ones.

### 5. Personalised explanation selection

Regret = how much worse the chosen explanation is than the best available one for that person (lower is better):

| Policy | Regret | 95% CI |
|--------|-------:|--------|
| **Personalised (MoE)** | **0.293** | 0.263 to 0.323 |
| Global model | 1.069 | 0.997 to 1.140 |
| Random | 1.766 | 1.713 to 1.819 |

Personalisation cuts regret by about 73% relative to a single global model. The biggest gain is for novices (0.133 vs. 1.432).

---

## Getting Started

### Requirements

- Python 3.9+
- A GPU is strongly recommended (CPU works but generation is slow)
- Internet access on first run to download the two Hugging Face models (about 3 GB and 1 GB)

Python packages (auto-installed by the first notebook cell if missing):

```
transformers  accelerate  scikit-learn  scipy
numpy  pandas  matplotlib  torch
```

### Run

1. Open `Personalised_Explanation_Preferences_MoE.ipynb` in Jupyter, VS Code, or Google Colab (a GPU runtime is recommended).
2. Run all cells from top to bottom.
3. Results and figures are written to `preference_work/outputs/`.

> Tip: in Colab, set an `HF_TOKEN` secret to avoid unauthenticated Hugging Face download warnings and get faster downloads.

---

## Configuration

Main settings live in the third code cell:

| Parameter | Default | Description |
|-----------|--------:|-------------|
| `GEN_LM` | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | Explanation generator |
| `READER_LM` | `Qwen/Qwen2.5-Coder-0.5B` | Reader model for surprisal |
| `MAX_NEW_TOKENS` | 160 | Max length of each explanation |
| `N_TRAIN_USERS` | 120 | Simulated training developers |
| `N_TEST_USERS` | 60 | Simulated test developers |
| `N_COMPARISONS` | 40 | Pairwise comparisons per developer |
| `N_EVAL_COMPARISONS` | 16 | Held-out comparisons per test developer |
| `K_SHOTS` | `[1, 2, 4, 8, 16]` | Few-shot budgets |
| `N_EXPERTS_GRID` | `[1..5]` | Candidate numbers of experts |
| `EM_ITERS` | 30 | EM iterations |
| `L1_C` | 0.5 | Inverse L1 strength (smaller = sparser) |
| `N_BOOT` | 1000 | Bootstrap repetitions |
| `SEED` | 11 | Global random seed |

---

## Using Real Preference Data

The notebook has a hook for real human data. Place a file at `preference_work/pairwise_choices.csv` with these columns:

| Column | Meaning |
|--------|---------|
| `user` | User ID |
| `a` | Row number (0-based) of explanation A in `explanations.csv` |
| `b` | Row number of explanation B in `explanations.csv` |
| `y` | `1` if A was preferred, `0` if B was preferred |

When the file exists, the notebook fits a new MoE on it and saves the per-user routing to `real_user_routing.csv` and prints the learned expert weights.

---

## Output Files

All saved to `preference_work/outputs/`:

| File | Content |
|------|---------|
| `explanations.csv` | The 192 generated explanations |
| `explanations_with_features.csv` | Explanations plus the 9 features |
| `manipulation_check.csv` | Cohen's *d* of instructions on features |
| `expert_number_selection.csv` | Held-out log-likelihood per number of experts |
| `expert_weights.csv` | Learned sparse expert weights |
| `fewshot_adaptation.csv` | Few-shot accuracy with confidence intervals |
| `faithfulness.csv` | Faithfulness / comprehensiveness results |
| `personalised_selection.csv` | Per-user, per-snippet regret |
| `summary.json` | Headline numbers |
| `fig1_manipulation_check.png` | Instruction-effect heatmap |
| `fig2_experts.png` | Learned experts vs. hidden types |
| `fig3_fewshot_adaptation.png` | Few-shot accuracy curves |
| `fig4_faithfulness.png` | Faithfulness curves |

---

## Limitations

- **Simulated users only.** The strong results (ARI 0.952, high cosine similarity) partly reflect that the simulated developers are generated from the same feature space and Bradley–Terry form the model assumes. Validation with real human preferences is essential before drawing conclusions.
- **Small, templated stimuli.** 24 short Python snippets and 192 explanations from one small generator model.
- **Hand-crafted features.** The 9 features are interpretable but incomplete. Jargon and hedge lists are manually defined, and hedging/confidence instructions were only weakly followed by the 1.5B model.
- **Selection is among pre-generated variants.** The system chooses from 8 existing explanations per snippet; it does not yet generate new text conditioned on the user profile.
- **Extra expert.** The model selected 4 experts for 3 true types; one expert is nearly unused.
- **Expert vs. novice demo.** In the printed `binary_search` demo, the expert and novice example users received the same variant, which suggests the 8-variant menu has limited diversity for that snippet.

---

## Future Work

- Collect real pairwise preferences from developers and fit the MoE on them.
- Add richer features (e.g. embeddings, readability scores) while keeping sparsity for interpretability.
- Condition the generator directly on the routed expert profile (prompting or fine-tuning) instead of selecting from a fixed menu.
- Evaluate with larger generator models and a broader set of code (multiple languages, longer functions).
- Study how explanation style (hedging, confidence) affects user trust and learning.

---

## License

Add your preferred license here (e.g. MIT).
