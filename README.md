# Bounded Decision-Making via Diffusion Transformers (DiT)



**Probabilistic Modeling for Tumor Diagnosis with Provable Hallucination Bounds**
**Period:** Feb 2026 – May 2026
**Tech:** Python, PyTorch, Julia, Triton, DiT

## Overview

This repository implements a **bounded decision-making** framework using **Diffusion Transformers (DiT)** to **strictly constrain generative hallucinations** in medical image restoration and downstream tumor diagnosis.
Unlike conventional “better upsampling” narratives, this work introduces a **boundedness theorem** that formalizes *when* and *how much* a generative restoration model may deviate from diagnostically faithful structures.

**Core idea:**

> Use probabilistic structuring + diffusion priors + explicit bounds to ensure **diagnostic fidelity** under restoration, reducing unsafe “plausible-but-wrong” generations.

---

## Objectives

* Design a **DiT-based diffusion** pipeline that supports **probabilistic inference** under medical constraints.
* Formulate a **boundedness theorem** to prevent hallucinations beyond permissible diagnostic error margins.
* Implement an **HPC-optimized pipeline** using **Julia** (orchestration/compute) + **Triton** (custom GPU kernels).
* Achieve **model compression** of **[N]%** while maintaining accuracy loss within **[N]%**, validating reliability as a probabilistic diagnostic tool.

---

## Key Contributions

1. **Boundedness theorem for hallucination control**

   * Establishes a constraint mechanism for generative restoration that preserves medically meaningful structures.
   * Goes beyond pixel-level loss or naive “upsampling hypothesis”.

2. **Diffusion Transformer (DiT) backbone for 3D medical imaging**

   * Patch-token attention backbone with diffusion timestep conditioning.
   * Supports restoration → inference workflows.

3. **Julia + Triton HPC integration**

   * Julia coordinates probabilistic inference / experiment orchestration.
   * Triton accelerates bottleneck kernels (attention / patchify / likelihood evaluation / sampling routines).

4. **Compression without diagnostic collapse**

   * Compression method: **[e.g., structured pruning / low-rank / quantization / distillation]**
   * Outcome: **[N]%** compression with **≤ [N]%** accuracy drop (task metrics below).

---

## Repository Structure

```text
.
├── README.md
├── configs/
│   ├── train_dit.yaml
│   ├── infer_bounded.yaml
│   └── compression.yaml
├── src/
│   ├── models/
│   │   ├── dit3d.py                # DiT backbone (3D patch tokens, AdaLN-Zero)
│   │   ├── diffusion.py            # q(x_t|x_0), sampling, eps/v parameterization
│   │   └── bounded_head.py         # bound-enforcing modules (projection/penalty)
│   ├── bounds/
│   │   ├── theorem.md              # theorem statement + assumptions
│   │   ├── bound_estimator.py      # compute/estimate bound terms from data/model
│   │   └── constraint.py           # enforce constraints in training/inference
│   ├── ops/
│   │   ├── blur.py        
│   │   └── sr.py
│   ├── eval/
│   │   ├── h5.py  
│   │   ├── ema.py            
│   │   └── mc_dropout.py
│   ├── pnp/
│   │   ├── cg.py  
│   │   ├── grad.py    
│   │   ├── pnp_train.py           
│   │   └── pnpstarter.py
│   ├── data/
│   │   ├── h5.py  
│   │   ├── monai.py
│   │   ├── pcam_starter.py            
│   │   └── split_utils.py
│   ├── train/
│   │   ├── train.py
│   │   ├── train_bounded.py        # adds bounded objective
│   │   └── metrics.py
│   ├── infer/
│   │   ├── restore.py              # restoration sampling
│   │   └── diagnose.py             # diagnosis head / evaluation
│   └── utils/
│       ├── make_dit_builder.py
│       ├── utils.py
│       └── ckpt.py
├── triton_kernels/
│   ├── attn_fused.py               # optional fused attention kernels
│   ├── patchify3d.py               # patchify/unpatchify acceleration
│   └── bound_ops.py                # fast bound-related ops
├── julia/
│   ├── Project.toml
│   ├── src/
│   │   ├── orchestrate.jl          # experiment orchestration
│   │   └── inference.jl            # probabilistic inference drivers
│   └── scripts/
│       ├── run_train.jl
│       └── run_infer.jl
└── experiments/
    ├── results/
    └── notebooks/
```

---

## Methodology

### 1) Generative restoration with diffusion DiT

We model a diffusion process over medical volumes and train a DiT network to predict **noise / velocity / x0** given `(x_t, t)`.

* **Backbone:** DiT (Transformer over 3D patch tokens)
* **Objective:** denoising loss (e.g., MSE on eps or v)
* **Sampling:** DDPM/DDIM/DPM-Solver (configurable)

### 2) Bounded decision-making

We enforce constraints that **bound clinically relevant deviations**. The bound can be expressed as:

* A hard constraint: projection onto feasible diagnostic manifold
* A soft constraint: penalty term added to diffusion loss
* A post-hoc constraint: rejection / correction during sampling

**Goal:** ensure restoration does not introduce structures that would flip/alter diagnosis outside defined tolerance.

### 3) Compression with fidelity guarantees

We compress the diffusion backbone while controlling diagnosis degradation:

* Technique: **[pruning/quantization/distillation/low-rank]**
* Validation: segmentation/diagnosis metrics remain within **[N]%** drop.

---

## Boundedness Theorem (Draft Slot)

> **Theorem (Bounded Diagnostic Deviation).**
> Under assumptions **A1–A[k]** and constraint operator **C(·)**, the restored sample ( \hat{x}*0 ) produced by the bounded diffusion process satisfies
> [
> d*{\text{diag}}(\hat{x}*0, x_0) \le \epsilon(\theta, \mathcal{D}, t),
> ]
> where ( d*{\text{diag}} ) is a diagnosis-aware discrepancy measure and ( \epsilon ) depends on model parameters ( \theta ), data distribution ( \mathcal{D} ), and diffusion time ( t ).

* Full statement & assumptions: `src/bounds/theorem.md`
* How bounds are estimated: `src/bounds/bound_estimator.py`
* How constraints are enforced: `src/bounds/constraint.py`

---

## Installation

### Python (PyTorch)

```bash
conda create -n bounded-dit python=3.10 -y
conda activate bounded-dit
pip install -r requirements.txt
```

### Triton (GPU kernels)

```bash
pip install triton
```

### Julia

```bash
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
```

---

## Quickstart

### Train DiT baseline

```bash
python -m src.train.train_dit --config configs/train_dit.yaml
```

### Train bounded diffusion (with constraint objective)

```bash
python -m src.train.train_bounded --config configs/infer_bounded.yaml
```

### Restore + Diagnose

```bash
python -m src.infer.restore  --config configs/infer_bounded.yaml
python -m src.infer.diagnose --config configs/infer_bounded.yaml
```

### Julia orchestration (optional)

```bash
julia --project=julia julia/scripts/run_train.jl
julia --project=julia julia/scripts/run_infer.jl
```

---

## Metrics & Evaluation

### Restoration metrics

* PSNR / SSIM (optional, not sufficient alone)
* Per-structure similarity (lesion-aware metrics)
* Uncertainty calibration (e.g., NLL, ECE if applicable)

### Diagnostic fidelity metrics

* Tumor diagnosis accuracy / AUROC
* Lesion segmentation Dice (if segmentation proxy is used)
* Bound violation rate: `% samples where d_diag > ε`

---

## Results (To be filled)

* Compression: **[N]%** parameter reduction
* Accuracy loss: **≤ [N]%**
* Bound violation rate: **[N]% → [N]%** (before/after bounding)

---

## Safety Notes (Medical Use)

This repository is **research-only**. Outputs from generative models can be misleading even when visually plausible.
Always evaluate with diagnosis-aware metrics and boundedness checks before any clinical interpretation.

---

## Roadmap (Feb–May 2026)

* [ ] finalize theorem assumptions + proof sketch
* [ ] implement bound estimator for tumor-specific discrepancy
* [ ] integrate Triton kernels for attention/patchify/hot ops
* [ ] run ablations: DiT depth/patch size/sampler/bound strength
* [ ] compression + robustness validation
* [ ] finalize reproducibility package (configs, seeds, checkpoints)

---

## Citation

If you build on this work, cite:

* **[Author]**, *Bounded Decision-Making via Diffusion Transformers*, 2026. *(preprint planned)*

