# SolarGuard AI: Deep Learning-Based Solar Panel Fault Classification

CAI 421 Deep Learning, Group 3. Controlled four-way comparison of a CBAM-style attention block and Focal Loss on the 12-class InfraredSolarModules dataset.
## Configurations

1. CNN + Cross-Entropy (baseline) - implemented
2. CNN + attention + Cross-Entropy - pending
3. CNN + Focal Loss (gamma = 2) - pending
4. CNN + attention + Focal Loss - pending
## Data

The notebook downloads the InfraredSolarModules dataset automatically from https://github.com/RaptorMaps/InfraredSolarModules (MIT license). No manual download is needed.
## Setup

Recommended: Google Colab. Upload the notebook and run it.
Local use: `pip install -r requirements.txt` (Python 3.10 or later).
## Reproduce

Open `SolarGuard_Baseline_L 2.ipynb` in Google Colab.
Runtime -> Run all.
The random seed is fixed (SEED = 42). Train/validation/test splits (70/15/15, stratified) are saved as CSV files for reuse by all configurations.
Outputs: test metrics (JSON), confusion matrix (PNG), training history (CSV), and model weights (.pth).
## Notes

No test data is used for tuning. Exact duplicate images and conflicting labels are removed before splitting. Perturbation tests are an image-space sensitivity check, not evidence of field robustness.
