## Table A. Clean test set (mean ± SD over seeds)

| Configuration | Seeds | Accuracy | Macro-F1 | Macro-recall | Best val Macro-F1 |
|---|---|---|---|---|---|
| CNN_CE | 3 | 0.7432 ± 0.0151 | 0.5209 ± 0.0529 | 0.4964 ± 0.0551 | 0.5266 ± 0.0492 |
| CNN_CBAM_CE | 3 | 0.7540 ± 0.0083 | 0.5512 ± 0.0233 | 0.5176 ± 0.0245 | 0.5586 ± 0.0200 |
| CNN_Focal | 3 | 0.7314 ± 0.0088 | 0.4957 ± 0.0411 | 0.4628 ± 0.0449 | 0.5203 ± 0.0199 |
| CNN_CBAM_Focal | 3 | 0.7421 ± 0.0081 | 0.5188 ± 0.0217 | 0.4820 ± 0.0193 | 0.5222 ± 0.0126 |

## Table B. Minority-class recall, clean test set (mean ± SD)

| Configuration | Soiling | Diode-Multi | Hot-Spot-Multi |
|---|---|---|---|
| CNN_CE | 0.0222 ± 0.0385 | 0.5185 ± 0.1335 | 0.3604 ± 0.1219 |
| CNN_CBAM_CE | 0.0444 ± 0.0385 | 0.4444 ± 0.0741 | 0.3694 ± 0.0949 |
| CNN_Focal | 0.0333 ± 0.0333 | 0.3333 ± 0.2429 | 0.2883 ± 0.1092 |
| CNN_CBAM_Focal | 0.0556 ± 0.0192 | 0.3457 ± 0.0214 | 0.3784 ± 0.0270 |

## Table C. Macro-F1 under perturbation (mean over seeds)

| Condition | CNN_CE | CNN_CBAM_CE | CNN_Focal | CNN_CBAM_Focal |
|---|---|---|---|---|
| clean_0 | 0.5209 | 0.5512 | 0.4957 | 0.5188 |
| brightness_-0.2 | 0.4149 | 0.4314 | 0.4029 | 0.3983 |
| brightness_-0.4 | 0.2704 | 0.2942 | 0.2766 | 0.2649 |
| brightness_0.2 | 0.3577 | 0.3685 | 0.3499 | 0.3541 |
| brightness_0.4 | 0.1244 | 0.1273 | 0.1130 | 0.1382 |
| contrast_0.6 | 0.3564 | 0.3994 | 0.3408 | 0.3637 |
| contrast_0.8 | 0.4755 | 0.4918 | 0.4557 | 0.4777 |
| contrast_1.2 | 0.5252 | 0.5443 | 0.4921 | 0.5189 |
| contrast_1.4 | 0.5050 | 0.5295 | 0.4797 | 0.4979 |
| gaussian_noise_0.05 | 0.1747 | 0.1464 | 0.1498 | 0.1499 |
| gaussian_noise_0.1 | 0.0682 | 0.0833 | 0.0829 | 0.0856 |
| gaussian_noise_0.2 | 0.0402 | 0.0621 | 0.0540 | 0.0526 |
