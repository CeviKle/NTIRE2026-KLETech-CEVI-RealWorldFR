# [NTIRE 2026 Challenge on Real-World Face Restoration](https://cvlai.net/ntire/2026/) @ [CVPR 2026](https://cvpr.thecvf.com/)

[![ntire](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Fraw.githubusercontent.com%2Fjkwang28%2FNTIRE2026_RealWorld_Face_Restoration%2Fmain%2Ffigs%2Fdiamond_badge.json)](https://www.cvlai.net/ntire/2026/)
[![page](https://img.shields.io/badge/Project-Page-blue?logo=github&logoSvg)](https://ntire-face.github.io/2026/)
[![visitors](https://visitor-badge.laobi.icu/badge?page_id=jkwang28.NTIRE2026_RealWorld_Face_Restoration&right_color=violet)](https://github.com/jkwang28/NTIRE2026_RealWorld_Face_Restoration)
[![GitHub Stars](https://img.shields.io/github/stars/jkwang28/NTIRE2026_RealWorld_Face_Restoration?style=social)](https://github.com/jkwang28/NTIRE2026_RealWorld_Face_Restoration)

## About the Challenge

This challenge focuses on restoring real-world degraded face images. The task is to recover high-quality face images with rich high-frequency details from low-quality inputs. At the same time, the output should preserve facial identity to a reasonable degree. There are no restrictions on computational resources such as model size or FLOPs. The main goal is to **achieve the best possible image quality and identity consistency**. 
Participants are ranked based on visual quality while ensuring identity similarity above a threshold; final scores combine several no-reference IQA metrics and FID. 

## Challenge results
**Test Set** – 450 low-quality (LQ) images drawn from five real-world subsets (WIDER-Test, WebPhoto-Test, CelebChild-Test, LFW-Test, and CelebA) are provided for evaluation.  

**Identity Validation** – Cosine similarity is measured with a pretrained **AdaFace** model. Thresholds: 0.30 (WIDER & WebPhoto), 0.60 (LFW & CelebChild), 0.50 (CelebA). A submission fails if more than ten faces fall below the dataset-specific threshold.  

**Metrics** – Valid submissions are scored with six no-reference metrics: **CLIPIQA, MANIQA, MUSIQ, Q-Align, NIQE,** and **FID** (against FFHQ).  

**Overall Score**

$$
\text{Score} = \text{CLIPIQA} + \text{MANIQA} + \frac{\text{MUSIQ}}{100} + \max\left(0, \frac{10 - \text{NIQE}}{10}\right) + \frac{\text{QALIGN}}{5} + \max\left(0, \frac{100-\text{FID}}{100}\right).
$$

**Ranking rule** – Teams are first screened by the identity filter; qualifying entries are ranked descending by the overall score. Minor deviations between Codalab and reproduced scores are tolerated after code verification.  

Overview
We propose SA-FGRC, a lightweight plug-in module inserted between Stage 2 and Stage 3 of the AllForFace pipeline. It addresses a fundamental limitation of diffusion-based face restoration: uniform texture enhancement across all facial regions despite their fundamentally different texture statistics.
Pipeline:
LQ → Stage 1 (StyleGANv2) → Stage 2 (DiffBIR) → [SA-FGRC] → Stage 3 (DINOv2-VAE) → HQ
Key idea: Decompose Stage 2 output into wavelet frequency bands, apply region-specific CNN corrections guided by BiSeNet face parsing, and reconstruct — leaving the low-frequency (identity) band completely untouched.

Results
MetricBaseline (AllForFace)Ours (SA-FGRC)DiffCLIPIQA ↑0.95220.9536+0.0014MUSIQ ↑77.5377.30-0.23NIQE ↓4.013.57-0.44MANIQA ↑0.72850.6922-0.0363Identity sim—0.9832excellent

Method
SA-FGRC Module

BiSeNet face parsing — segments face into 5 regions: skin, hair, eyes, mouth, background
Haar wavelet decomposition — splits Stage 2 output into LL (low-freq) + LH, HL, HH (high-freq)
Region-specific CNNs — 5 lightweight CNNs (23K params each) apply targeted corrections per region
LL band untouched — identity and structure are preserved by design
Inverse wavelet — reconstruct corrected image → feed into Stage 3

Learned region correction strengths
RegionInitialFinalHair0.10.216Skin0.10.169Eyes0.10.157Mouth0.10.083Background0.10.028
The network correctly learned that hair needs the strongest correction (diffusion over-generates strand artifacts) and background/mouth need minimal correction.

Setup
Requirements
bashpip install -r requirements.txt
Download weights
Download all model weights from Google Drive:
Download weights
Place all files in ./model_zoo/team06_KLETechCEVI/:
FileSizeDescriptionsemantic_best.pth214 MBSA-FGRC module weightsfidelity_model.pth318 MBStage 1 fidelity modelnaturalness_model.pt43 MBStage 3 naturalness modelv2.pth1.35 GBControlNet weightsv2-1_512-ema-pruned.ckpt4.86 GBStable Diffusion weightsram_plus_swin_large_14m.pth2.8 GBRAM captionerbert-base-uncased/—Text encoder79999_iter.pth51 MBBiSeNet face parser

Inference
bashexport BASICSR_JIT=True

CUDA_VISIBLE_DEVICES=0 python test.py \
    --test_dir /path/to/test/images \
    --save_dir ./results \
    --model_id 6
The test images must be organized as:
test_dir/
├── CelebA/
├── Wider-Test/
├── LFW-Test/
├── WebPhoto-Test/
└── CelebChild-Test/

Repository Structure
NTIRE2026-KLETech-CEVI-RealWorldFR/
├── models/
│   ├── team00_CodeFormer/          # Baseline
│   ├── team01_AllForFace/          # AllForFace base pipeline
│   └── team06_KLETechCEVI/         # Our SA-FGRC module
│       ├── main.py                 # Entry point
│       ├── allforface_freq.py      # Full pipeline with SA-FGRC
│       ├── semantic_wavelet_refiner.py  # SA-FGRC implementation
│       ├── model.py                # BiSeNet model definition
│       └── resnet.py               # BiSeNet backbone
├── model_zoo/
│   └── team06_KLETechCEVI/
│       └── team06_KLETechCEVI.txt  # Weight download links
├── test.py
├── eval.py
└── requirements.txt

Team
KLETech-CEVI · KLE Technological University, Hubballi, India

Nikhil Akalwadi
Sujith Roy V
Claudia Jesuraj
Vikas B
Spoorthi LC
Ramesh Ashok Tabib
Uma Mudenagudi


Citation
If you find this work useful, please cite:
bibtex@inproceedings{ntiface26face,
  title={NTIRE 2026 challenge on real-world face restoration: Methods and results},
  author={Wang, Jingkai and others},
  booktitle={CVPRW},
  year={2026}
}

Acknowledgements
This work builds on AllForFace, DiffBIR, and BiSeNet.[IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch), [VQFR](https://github.com/TencentARC/VQFR), [AdaFace](https://github.com/mk-minchul/AdaFace). 
