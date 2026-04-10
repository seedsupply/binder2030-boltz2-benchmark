# Binder2030 Boltz-2 Benchmark

This repository contains analysis code and Boltz-2 prediction 
results for the following manuscript:

**"Binder2030: A Standardized Quantitative Binding Dataset 
for Benchmarking AI-Based Affinity Prediction Across 
Membrane Protein Targets"**

Naoki Tarui, Thuy Duong Nguyen, Masaharu Nakayama  
SEEDSUPPLY INC.  
Contact: naoki.tarui@seedsupply.co.jp  
Website: www.seedsupply.co.jp

---

## Repository Contents

### data/
| File | Description |
|------|-------------|
| gpcr_correlation_results.csv | Per-target Pearson r, GPCR (n=100) |
| slc_correlation_results_pub.csv | Per-target Pearson r, SLC (n=82) |
| fpocket_gpcr_results.csv | fpocket drug_score, GPCR (n=100) |
| fpocket_slc_results.csv | fpocket drug_score, SLC (n=82) |
| integrated_analysis.csv | GPCR+SLC integrated analysis (n=182) |

### scripts/
| File | Description |
|------|-------------|
| 01_slc_correlation.py | SLC per-target correlation analysis |
| 02_fpocket_gpcr.py | GPCR fpocket analysis |
| 03_fpocket_slc.py | SLC fpocket analysis |
| 04_integrated_analysis.py | GPCR+SLC integrated analysis |
| 05_gpcr_improved_model.py | GPCR improved prediction model |
| 06_residual_analysis.py | Residual variance analysis |
| 07_slc_std_aff.py | SLC std_aff_value analysis |
| 08_prepare_slc_inputs.py | Boltz-2 input YAML preparation (requires proprietary data) |
| 09_run_fpocket_gpcr.py | fpocket execution for GPCR |
| 10_run_fpocket_slc.py | fpocket execution for SLC |

---

## Key Findings

### GPCR (100 targets, 1,270 pairs)
- Mean Pearson r = 0.214
- CWxP motif in TM6 as structural predictor (P = 0.036*)
- cwxp_type refinement improves prediction (P = 0.010**)
- Best combined model R² = 21.4%

### SLC (82 targets, 1,391 pairs)
- Mean Pearson r = 0.121
- Clear Orthosteric Pocket as structural predictor (P = 0.0002***)
- Compound logP and MW diversity as additional predictors
- Best combined model R² = 28.0%

### Integrated Analysis (182 targets)
- Common principle: consistency of ligand-induced structural change
- Structural predictor P = 0.0001***
- fpocket drug_score uninformative in both classes (P = 0.916)
- std_aff_value as post-hoc prediction reliability indicator

---

## Data Availability

The **Binder2030 experimental Kd values** are proprietary to 
SEEDSUPPLY INC. and are not included in this repository.

### What is available here
- Boltz-2 predicted affinity values (affinity_pred_value)
- Per-target Pearson correlation statistics
- fpocket structural analysis results (drug_score, pocket_volume)
- All Python analysis scripts

### What requires a Data Use Agreement
- Experimental Kd values (pKd) measured by ASMS
- Available for non-commercial research upon request
- Contact: naoki.tarui@seedsupply.co.jp

### Not available
- Compound structures (SMILES)
- Compound identifiers linked to experimental Kd data

---

## Requirements

```
Python 3.10
pandas >= 1.5
scipy >= 1.9
scikit-learn >= 1.1
rdkit >= 2022.09
numpy >= 1.23
fpocket (https://github.com/Discngine/fpocket)
```

---

## Citation

If you use this code or data, please cite:

> Tarui N, Nguyen TD, Nakayama M. Binder2030: A Standardized 
> Quantitative Binding Dataset for Benchmarking AI-Based 
> Affinity Prediction Across Membrane Protein Targets. 
> *Journal of Chemical Information and Modeling*, 2026. 
> [DOI to be added upon publication]

---

## License

Code: MIT License  
Data: See Data Availability section above.

© 2026 SEEDSUPPLY INC. All rights reserved.
