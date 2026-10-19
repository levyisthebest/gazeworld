# GazeWorld

**A World Model of Radiologist Reading for Medical Image Representation Learning**

Yiwei Li\*, Zihao Wu\*, Huaqin Zhao, Yifan Zhou, Chao Cao, Dajiang Zhu, Tianming Liu, Lin Zhao†
<br>*\*Equal contribution. †Corresponding author.*

📄 Paper: [arXiv:2605.23992](https://arxiv.org/abs/2605.23992)

---

GazeWorld treats a chest radiograph as a **perceptual world** and the radiologist's
fixation sequence as a **trajectory** through it. An autoregressive predictor forecasts the
latent representation of the next expert-fixated patch from all previously visited ones
(JEPA-style, in representation space), while a **spatial-completion** branch predicts the
representations of all unvisited patches. At inference, the predictor generates patch
representations **from the image alone, with no gaze data**, so the learned encoder transfers
to standard diagnostic tasks. Frozen GazeWorld features reach state-of-the-art diagnostic and
zero-shot accuracy on CheXpert / RSNA / SIIM-ACR, and a generic decoder on the same frozen
features beats the purpose-built LogitGaze-Med on GazeSearch scanpath prediction.

```
L_pre = L_AR  +  λ · L_SC          (λ = 1.0)
  L_AR = mean_i  SmoothL1( LN(ẑ_{s_i}), sg( z̄_{s_i} ) )     # next-fixation prediction
  L_SC = mean_p  SmoothL1( LN(r̂_p),    sg( LN(z̄_p) ) )      # spatial completion, p ∉ visited
```
Targets come from an EMA momentum copy of the online encoder (stop-gradient); the online
encoder, fixation embedder, AR predictor, prediction head and SC decoder are trained jointly.

## Repository layout

This repo contains the two components evaluated in the paper, sharing one world-model package.

```
gazeworld/
├── models/                         # GazeWorld world-model (representation learning; Tables 1 & 2)
│   ├── fixation_ar_jepa.py             # base blocks: RMSNorm, RoPE, SwiGLU, FixationEmbedder, causal Transformer
│   ├── fixation_ar_jepa_medclip.py     # MedCLIP-Swin visual encoder (224², 49 patches, d=768)
│   ├── fixation_ar_jepa_v4.py          # full model w/ AR + spatial-completion (ablation modes a/b/c/d)
│   └── fixation_ar_jepa_v4b.py         # clean paper model  (build_v4b)
├── datasets/
│   ├── fixation_dataset.py             # MIMIC-EYE gaze sequences (pretraining)
│   ├── cached_fixation_dataset.py      # cached fixation features (.npz)
│   ├── chexpert_dataset.py             # CheXpert (14-class / 5×200)
│   └── transforms.py
├── scripts/
│   ├── preprocess_fixation_cache.py    # build the MIMIC-EYE fixation cache
│   ├── train_v4b_stage1.py             # Stage-1: world-model pretraining on MIMIC-EYE
│   ├── train_v4b_stage2.py             # Stage-2: CheXpert fine-tune
│   ├── train_v4b_rsna_siim_sft.py      # SFT on RSNA / SIIM-ACR
│   └── benchmark_v4b.py                # frozen linear probe @ 1% / 10% / 100% (Table 1)
└── scanpath/                       # GazeSearch scanpath decoder (probing; Table 3)
    ├── sgj_models/
    │   ├── scanpath_predictor_v5.py        # "Ours (CheX)"    — CheXNet (torchxrayvision) backbone
    │   ├── scanpath_predictor_v5res.py     # "Ours (Res)"     — ResNet-50 backbone
    │   └── scanpath_predictor_v5medclip.py # "Ours (MedCLIP)" — MedCLIP-Swin backbone
    ├── data/gazesearch_dataset.py      # GazeSearch (ChestSearch protocol)
    ├── evaluation/                     # ScanMatch, MultiMatch, scanpath metrics
    └── scripts/train_scanpath_v5*.py, eval_scanpath_v5*.py
```

The scanpath decoder reuses `models.fixation_ar_jepa` (causal Transformer blocks) from the
world-model package; the imports resolve relative to the repo root, so no path editing is needed.

## Installation

```bash
conda create -n gazeworld python=3.9 -y && conda activate gazeworld
pip install -r requirements.txt
```

## Data & paths

Datasets and pretrained backbone weights are **not** included. Download them, then edit the
path constants near the top of the relevant scripts (they are currently absolute paths from the
authors' machine):

| Resource | Used by | Get it from |
|---|---|---|
| **MIMIC-EYE** (radiologist gaze) | `scripts/preprocess_fixation_cache.py`, `train_v4b_stage1.py` | PhysioNet (Hsieh et al., 2023) |
| **CheXpert** | `train_v4b_stage2.py`, `benchmark_v4b.py` | Stanford CheXpert |
| **RSNA Pneumonia** | `train_v4b_rsna_siim_sft.py`, `benchmark_v4b.py` | RSNA / Kaggle |
| **SIIM-ACR Pneumothorax** | `train_v4b_rsna_siim_sft.py`, `benchmark_v4b.py` | Kaggle |
| **GazeSearch (ChestSearch)** | `scanpath/data/gazesearch_dataset.py` | Pham et al., 2025 |
| **MedCLIP-Swin weights** | MedCLIP encoder | `medclip` package / MedCLIP release |

Dataset statistics (from the paper): MIMIC-EYE = 3,032 CXRs, split 2,729 / 151 / 152, ~47
fixations/image; GazeSearch = 3,870 train / 488 test, 13 pathology categories, ≤7 fixations.

## Reproducing the results

### 1. World-model pretraining (Stage-1)
```bash
python scripts/preprocess_fixation_cache.py          # build the MIMIC-EYE fixation cache (once)
python scripts/train_v4b_stage1.py --out_dir outputs/gazeworld
# MedCLIP-Swin encoder, 8-layer causal Transformer (d=768, 12 heads), AdamW lr 3e-4 wd 0.04,
# bs 32, 15 epochs, AMP, λ=1.0, EMA momentum 0.998→1.0 (cosine).
```

### 2. Diagnostic transfer (Tables 1 & 2)
```bash
# Frozen linear probe at 1% / 10% / 100% labels on CheXpert / RSNA / SIIM-ACR:
python scripts/benchmark_v4b.py --stage1_ckpt outputs/gazeworld/stage1_ckpt.pth
# Supervised fine-tune:
python scripts/train_v4b_stage2.py        --stage1_ckpt outputs/gazeworld/stage1_ckpt.pth   # CheXpert
python scripts/train_v4b_rsna_siim_sft.py --stage1_ckpt outputs/gazeworld/stage1_ckpt.pth   # RSNA / SIIM
```

### 3. Scanpath probing on GazeSearch (Table 3)
The visual encoder is frozen; only a lightweight scanpath decoder is trained on top of its
patch features (448², 14×14 grid), so differences reflect scanpath information already in the
representation. Three backbone variants:
```bash
cd scanpath
python scripts/train_scanpath_v5.py        && python scripts/eval_scanpath_v5.py        # Ours (CheX)
python scripts/train_scanpath_v5res.py     && python scripts/eval_scanpath_v5res.py     # Ours (Res)
python scripts/train_scanpath_v5medclip.py && python scripts/eval_scanpath_v5medclip.py # Ours (MedCLIP)
# Decoder: 6-layer causal Transformer (d_s=512, 8 heads), heads = spatial(196-way)/duration/termination,
# loss = CE + 0.1·L1(dwell) + 0.1·BCE(term), 7 fixations. AdamW lr 5e-4, bs 16, 30 epochs.
```

## Results (from the paper)

### Table 1 — Supervised classification AUROC (%), frozen backbone, 1% / 10% / 100% labels
| Method | CheXpert | RSNA | SIIM-ACR |
|---|---|---|---|
| BioViL | 54.19 / 60.37 / 66.55 | 70.74 / 72.84 / 78.48 | 62.98 / 68.39 / 78.20 |
| MedKLIP | – / – / 79.43 | 78.62 / 82.76 / 84.67 | 74.50 / 78.27 / 84.67 |
| MGCA | 67.12 / 76.10 / 81.57 | 80.41 / 85.67 / 88.24 | 86.14 / 88.13 / 91.84 |
| GLoRIA | 67.40 / 77.37 / 81.74 | 81.90 / 81.24 / 88.57 | – / – / – |
| PRIOR | 65.51 / 70.70 / 79.11 | 79.74 / 79.72 / 83.95 | 81.96 / 82.76 / 85.33 |
| CheXWorld | 62.81 / 80.03 / 85.39 | 78.30 / 84.12 / 88.66 | 62.31 / 81.54 / 91.34 |
| MedCLIP | 74.65 / 82.88 / 85.58 | 82.31 / 84.16 / 87.86 | 86.09 / 87.41 / 91.46 |
| EGMA | 71.21 / 79.79 / 85.45 | 82.49 / 84.73 / 88.09 | 87.17 / 87.91 / 88.98 |
| **GazeWorld (ours)** | **78.37 / 83.41 / 87.61** | **83.27 / 85.75 / 90.15** | **87.85 / 90.48 / 94.27** |

### Table 2 — Zero-shot classification, AUROC / Acc / F1 (%)
| Method | CheXpert 5×200 | RSNA | SIIM-ACR |
|---|---|---|---|
| CLIP | 15.50 / 19.60 / 10.99 | 21.67 / 23.46 / 19.87 | 42.13 / 46.11 / 40.13 |
| GLoRIA | 49.18 / 51.10 / 47.25 | 60.77 / 61.53 / 60.09 | – / – / – |
| MGCA | 42.53 / 43.39 / 39.67 | 59.19 / 60.82 / 59.57 | 36.60 / 38.45 / 34.95 |
| MedCLIP | 56.63 / 57.40 / 55.72 | 58.08 / 62.17 / 40.06 | 56.17 / 56.89 / 55.44 |
| EGMA | 56.09 / 57.15 / 55.03 | 57.46 / 66.85 / 43.08 | 59.79 / 63.24 / 61.04 |
| **GazeWorld (ours)** | **58.13 / 59.42 / 56.85** | **62.84 / 70.18 / 47.31** | **64.29 / 66.59 / 63.26** |

### Table 3 — GazeSearch scanpath probing (SM/STDE/Vec/Dir/Pos ↑, SED ↓)
| Method | SM ↑ | SED ↓ | STDE ↑ | Vec ↑ | Dir ↑ | Pos ↑ |
|---|---|---|---|---|---|---|
| GazeFormer | 0.280 | 5.11 | 0.801 | 0.902 | 0.644 | 0.803 |
| HAT | 0.296 | 5.07 | 0.800 | 0.909 | 0.649 | 0.825 |
| GazeSearch | 0.351 | 4.89 | 0.809 | 0.917 | 0.679 | 0.829 |
| LogitGaze | 0.328 | 5.07 | 0.810 | 0.882 | 0.643 | 0.809 |
| LogitGaze-Med (Res) | 0.416 | 4.68 | 0.852 | 0.935 | 0.650 | 0.823 |
| LogitGaze-Med (CheX) | 0.419 | 4.68 | 0.855 | 0.938 | 0.651 | 0.823 |
| **Ours (MedCLIP)** | 0.479 | 3.74 | 0.909 | 0.941 | 0.649 | 0.837 |
| **Ours (Res)** | 0.484 | 3.66 | **0.912** | **0.966** | 0.678 | **0.917** |
| **Ours (CheX)** | **0.489** | **3.63** | 0.911 | **0.966** | **0.686** | 0.915 |

## Citation

```bibtex
@article{li2026gazeworld,
  title   = {A World Model of Radiologist Reading for Medical Image Representation Learning},
  author  = {Li, Yiwei and Wu, Zihao and Zhao, Huaqin and Zhou, Yifan and Cao, Chao
             and Zhu, Dajiang and Liu, Tianming and Zhao, Lin},
  journal = {arXiv preprint arXiv:2605.23992},
  year    = {2026}
}
```
