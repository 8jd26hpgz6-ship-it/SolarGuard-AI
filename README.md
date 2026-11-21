# SolarGuard AI — Final Code Submission Package

This is the source-code and experiment-evidence component of the CAI 421 final project submission. It is organized for inspection and reproduction.

## Folder map

- `code/solarguard_train.py` — CNN, optional CBAM, Cross-Entropy/Focal Loss, training and evaluation.
- `code/summarize_results.py` — generates the summary tables from `all_results.csv`.
- `code/requirements.txt` — Python dependencies.
- `code/AI_TOOL_DISCLOSURE.txt` — disclosure for the course appendix.
- `code/screenshots/` — source-code screenshots and the supplied Colab execution evidence.
- `colab/SolarGuard_Run_All_Experiments.ipynb` — Colab workflow used for the final run.
- `results/outputs/` — exported result data, including per-seed/per-condition classification reports, confusion matrices and training histories.
- `results/run_log.txt` — full run log.
- `results/RUN_MANIFEST.md` — run summary, results and interpretation notes.

## Reproduce the run in Google Colab

1. Open `colab/SolarGuard_Run_All_Experiments.ipynb` in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU** (or another CUDA GPU if T4 is unavailable).
3. Run all notebook cells. At the upload prompt, upload this ZIP file: `SolarGuard_AI_Final_Code_Package.zip`.
4. The notebook downloads the public Raptor Maps dataset, runs the smoke test and GPU check, then runs all four configurations with seeds 42, 43 and 44.
5. At the end, the notebook creates and downloads `results_bundle.zip`.

The command used for the full experiment is:

```bash
python -u solarguard_train.py \
  --data-root /content/solar_dataset/InfraredSolarModules \
  --output-dir outputs \
  --config all \
  --seeds 42 43 44 \
  --max-epochs 50
```

Then create the paper-ready summaries with:

```bash
python summarize_results.py --output-dir outputs
```

## Final run evidence

The supplied Colab screenshots show the successful smoke test, CUDA availability with an NVIDIA Tesla T4, the training/evaluation run, and summary output. The exported results contain **144 evaluation rows**: 4 configurations × 3 seeds × 12 test conditions. The exact aggregate metrics are in `results/outputs/summary_tables.md` and `results/verified_results_summary.csv`.

The best mean clean-test Macro-F1 in this run was obtained by **CNN + CBAM + Cross-Entropy** (0.5512 ± 0.0233), with accuracy 0.7540 ± 0.0083. This is a descriptive result over three seeds, not a claim of statistical significance. See `results/RUN_MANIFEST.md` for the complete result summary and limitations.

## Important implementation notes

- The source dataset is downloaded by the notebook and is not included in this archive.
- Model weight files (`*.pt`) are not included in the exported result bundle; training histories and evaluation reports are included.
- Brightness perturbations are additive offsets to normalized pixel values, not percentages.
- The reported image perturbations are controlled image-space sensitivity tests, not evidence of real-world field robustness.
- The paper uses the final three-seed run results consistently. The earlier Checkpoint 2 one-run baseline numbers are historical and are not mixed with the final aggregate table.

## Local smoke test

Install dependencies from `code/requirements.txt`, then run:

```bash
cd code
python solarguard_train.py --smoke-test
```

Expected message:

```text
Smoke test passed: baseline/CBAM shapes and Focal Loss are valid.
```
