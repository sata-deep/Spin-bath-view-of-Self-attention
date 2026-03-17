# Spin-bath-view-of-Self-attention
This repository contains compact, reproducible experiments for testing the spin-bath / Hamiltonian view of self-attention developed in the accompanying work on GPT-2 as discussed in the paper: https://doi.org/10.1103/rkyb-d7d2

<img width="1853" height="979" alt="image" src="https://github.com/user-attachments/assets/5e9c9fb5-9443-44ed-beaf-941f21f0e2eb" />




This repository contains compact, reproducible experiments for testing the spin-bath / Hamiltonian view of self-attention developed in the accompanying work on GPT-2. The code extracts head-wise query, key, value, and output projections, builds the corresponding theoretical logit-gap prediction, and compares it with the model's observed next-token preference on a set of factual recall prompts.

The current scripts are designed to reproduce the main qualitative analyses discussed in the paper bundle included here, especially the comparison between:

$$
\Delta L_{\mathrm{theory}}
$$

and

$$
\Delta L_{\mathrm{actual}}.
$$

The repository is intentionally small: one script generates the head-level dataset, and the remaining scripts produce the figures.

## Scientific idea

For a prompt with token embeddings

$$
S = \{s_1, s_2, \dots, s_T\}, \qquad s_t \in \mathbb{R}^{d_{\mathrm{model}}},
$$

Each attention head defines projected query, key, and value states

$$
Q = S W_Q, \qquad K = S W_K, \qquad V = S W_V.
$$

The head-level interaction matrix is

$$
\Omega = \frac{QK^\top}{\sqrt{d_{\mathrm{head}}}},
$$

and for the final token position the corresponding attention weights are

$$
\alpha_i = \frac{\exp(\Omega_{T,i})}{\sum_j \exp(\Omega_{T,j})}.
$$

The head state aggregated at the final position is then

$$
N_0^{(\mathrm{head})} = \sum_{i=1}^{T} \alpha_i V_i.
$$

After projection back to model space,

$$
N_0 = N_0^{(\mathrm{head})} W_O^\top.
$$

For a "good" continuation token embedding $s_{\mathrm{good}}$ and a competing "bad" token embedding $s_{\mathrm{bad}}$, the theoretical head-level preference used throughout this repository is

$$ \Delta L_{\mathrm{theory}} = N_0 \cdot s_{\mathrm{good}}$$
-
$$
N_0 \cdot s_{\mathrm{bad}}.
$$

The measured model-level quantity is the actual difference in GPT-2 next-token logits,

$$
\Delta L_{\mathrm{actual}}
=\ell_{\mathrm{good}} - \ell_{\mathrm{bad}}.
$$

The central empirical question is whether heads whose internal geometry predicts a large positive or negative $\Delta L_{\mathrm{theory}}$ also show corresponding changes in $\Delta L_{\mathrm{actual}}$.

## Repository contents

- `data.py`: Loops over all GPT-2 layers and heads for a set of 20 factual prompts, computes $\Delta L_{\mathrm{theory}}$ and $\Delta L_{\mathrm{actual}}$, and writes `head_analysis_results.csv`.
- `Fig1.py`: Loads the CSV, isolates a target head, performs linear regression between theory and measured logits, and produces the correlation plot.
- `Fig2.py`: Builds a 2D decision landscape for a selected head and prompt by projecting token/value geometry onto a decision axis and an orthogonal axis.
- `Fig3.py`: Sweeps generation temperature and computes averaged order parameters such as next-token entropy and top-1 probability.
- `Fig4.py`: Causally tests one selected head using a forward pre-hook that zeros out the chosen head before the output projection, then measures the change in the logit gap.
- `head_analysis_results.csv`: Example head-level output so the figure scripts can be run without regenerating the dataset first.
- `2507.00683v5.pdf`: Local copy of the manuscript used for context.

## What each figure is testing

### 1. Head-wise theory vs. model output

`data.py` and `Fig1.py` test whether the Hamiltonian-inspired head-level score tracks GPT-2 behavior across prompts. For each prompt/head pair, the code stores:

$$
(\Delta L_{\mathrm{theory}}, \Delta L_{\mathrm{actual}}).
$$

`Fig1.py` then fits a line and reports the Pearson correlation and $r^2$ value for the selected head.

### 2. Decision landscape for a single head

`Fig2.py` constructs a geometric picture of the competition between the good and bad continuations. It uses a decision direction proportional to

$$
v_{\mathrm{diff}} \propto s_{\mathrm{good}}W_V - s_{\mathrm{bad}}W_V,
$$

and a second orthogonal direction to visualize where the head state $N_0^{(\mathrm{head})}$ sits relative to the two candidate continuations.

This makes the sign of the preference interpretable: positive alignment with the decision axis favors the good token, while negative alignment favors the bad token.

### 3. Temperature as a control parameter

`Fig3.py` treats sampling temperature as a global control parameter and tracks two averaged observables over prompts:

$$
p_i(T) = \frac{\exp(\ell_i/T)}{\sum_j \exp(\ell_j/T)},
$$

$$
H(T) = -\sum_i p_i(T)\log_2 p_i(T),
$$

and

$$
P_{\mathrm{top1}}(T) = \max_i p_i(T).
$$

These quantities play the role of order/disorder diagnostics: higher temperature typically raises entropy and lowers the dominance of the top prediction.

### 4. Causal ablation

`Fig4.py` asks whether a chosen head is merely correlated with the effect or actually contributes to it. The code zeros the selected head's slice in the concatenated multi-head output just before the `c_proj` operation, then recomputes

$$
\Delta L_{\mathrm{actual}}^{(\mathrm{ablated})}.
$$

Comparing baseline and ablated runs reveals whether removing that head weakens or strengthens the factual preference.

## Quickstart

### 1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you need a CUDA-enabled PyTorch build, install the appropriate PyTorch wheel first and then install the remaining requirements.

### 2. Generate the head-level dataset

```bash
python data.py
```

This downloads GPT-2 weights on the first run if they are not already cached locally, and writes:

```text
head_analysis_results.csv
```

### 3. Recreate the figures

```bash
python Fig1.py
python Fig2.py
python Fig3.py
python Fig4.py
```

These scripts save:

```text
Fig1.png
Fig2.png
Fig3.png
Fig4.png
```

to the repository root.

## Practical notes

- The scripts use Hugging Face GPT-2 weights and tokenizer assets. The first run therefore requires internet access unless the cache is already populated.
- `data.py` uses 20 factual prompts with manually specified good/bad continuations. Extending that list is the simplest way to broaden the empirical test bed.
- The default "antagonistic" head used in the plotting scripts is `Layer 3, Head 5`, as set near the top of `Fig1.py`, `Fig2.py`, and `Fig4.py`.
- Running on CPU is possible, but the all-head sweep in `data.py` is much faster on GPU.
- The code operates directly on GPT-2 embeddings and projection matrices, so it is best interpreted as a mechanistic probe rather than a training or fine-tuning pipeline.

## Citation

If this code, analysis, or any derived results are useful in your work, please cite:

**Bhattacharjee & Lee, Phys. Rev. E 113, 015308 (2026).**

If relevant for your workflow, you may also cite the repository manuscript included here:

- *Testing the spin-bath view of self-attention: A Hamiltonian analysis of GPT-2 Transformer*, `arXiv:2507.00683`.

Please keep the citation request with any reuse of the codebase, derivative scripts, or reproduced figures.
