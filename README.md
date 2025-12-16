# dMMR-Immune-Subtyping

**Immune Subtyping and 20-Gene Classifier for dMMR/MSI-H Tumors**

## Overview

This repository contains analysis code and data for the manuscript:

> **"Finding the True Responders: Stratifying dMMR/MSI-H Tu-mors for ICI Response"**

## 20-Gene Signature

| Gene | Gene | Gene | Gene |
|------|------|------|------|
| CTSS | STAT1 | CD74 | HLA-A |
| HLA-E | HLA-C | HLA-B | CFB |
| HLA-DRA | HLA-DRB1 | HLA-DQA1 | TAP1 |
| SOD2 | IDO1 | IFITM1 | SERPING1 |
| C1S | B2M | RNF213 | IFI30 |

## Results Summary

### Classifier Performance

| Cohort | N | AUC (95% CI) | Accuracy |
|--------|---|--------------|----------|
| Internal (TCGA, 5-fold CV) | 259 | 0.948 (0.920–0.973) | 88.8% |
| External (GSE39582) | 77 | 0.935 (0.880–0.979) | 85.7% |

### Clinical Validation (IMvigor210, TMB ≥ 10)

| Group | ORR | p-value |
|-------|-----|---------|
| HotHigh | 55.6% (20/36) | 0.034 |
| Others | 32.8% (20/61) | - |

## Setup

### 1. Unzip TCGA data

```bash
unzip TCGA_expression_with_labels.zip
```

### 2. Install Python dependencies

```bash
pip install pandas numpy scikit-learn scipy lifelines matplotlib seaborn
```

### 3. Install R packages

```r
install.packages(c("dplyr", "readr"))
BiocManager::install(c("fgsea", "msigdbr"))
```

## Files

### Data Files

| File | Description |
|------|-------------|
| `TCGA_expression_with_labels.zip` | TCGA dMMR/MSI-H cohort (n=259) |
| `GEO_expression_with_labels.csv` | External validation cohort GSE39582 (n=77) |
| `IMvigor210_expression_20genes.csv` | IMvigor210 expression data |
| `IMvigor210_clinical.csv` | IMvigor210 clinical data |
| `DEG_HotHigh_vs_rest.csv` | DEG results for signature selection |

### Scripts

| Script | Description |
|--------|-------------|
| `signature_selection.R` | 20-gene signature selection pipeline |
| `classifier_validation.py` | 5-fold CV and external validation |
| `IMvigor210_prediction.py` | Apply classifier to IMvigor210 |
| `IMvigor210_survival_analysis.py` | Multivariate Cox regression |

## Usage

### 1. Signature Selection (R)

```r
source("signature_selection.R")
```

### 2. Classifier Validation (Python)

```bash
python classifier_validation.py
```

### 3. IMvigor210 Analysis (Python)

```bash
python IMvigor210_prediction.py
python IMvigor210_survival_analysis.py
```

## Data Sources

| Dataset | Source | URL |
|---------|--------|-----|
| TCGA | UCSC Xena | https://xenabrowser.net/datapages/ |
| GSE39582 | GEO | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE39582 |
| IMvigor210 | R package | http://research-pub.gene.com/IMvigor210CoreBiologies/ |

## License

MIT License

## Contact

For questions, please open an issue on GitHub.
