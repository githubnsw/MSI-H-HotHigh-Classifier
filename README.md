# dMMR-immune-subtyping

Immune subtyping and 20-gene signature for dMMR/MSI-H tumors

## Overview

This repository contains the code for the manuscript:
**"Immune landscape-based subtyping and a 20-gene signature 
for predicting immunotherapy response in dMMR/MSI-H tumors"**

## Data Sources

| Dataset | Source | Accession |
|---------|--------|-----------|
| TCGA (UCEC, COAD, READ, STAD) | UCSC Xena | Pan-Cancer Atlas |
| External validation | GEO | GSE39582 |
| Clinical validation | IMvigor210 | R package |

## Requirements

- R >= 4.0
- Python >= 3.8

### R packages
```r
install.packages(c("dplyr", "ggplot2", "survival", "survminer"))
BiocManager::install(c("limma", "clusterProfiler", "fgsea", "MCPcounter"))
```

### Python packages
```bash
pip install pandas numpy scikit-learn lifelines scipy matplotlib seaborn
```

## Usage

### 1. Immune subtyping (R)
```r
source("scripts/R/01_data_preprocessing.R")
source("scripts/R/02_MCP_counter_scoring.R")
# ... 순서대로 실행
```

### 2. Classifier validation (Python)
```bash
python scripts/Python/10_classifier_validation.py
```

### 3. IMvigor210 clinical validation (Python)
```bash
python scripts/Python/11_IMvigor210_analysis.py
```

## 20-Gene Signature

| Gene | Description |
|------|-------------|
| CTSS | Cathepsin S |
| STAT1 | Signal transducer and activator of transcription 1 |
| CD74 | MHC class II invariant chain |
| HLA-A | MHC class I |
| ... | ... |

## Results Summary

| Cohort | N | AUC (95% CI) |
|--------|---|--------------|
| Internal (TCGA) | 259 | 0.948 (0.920-0.973) |
| External (GSE39582) | 77 | 0.935 (0.880-0.979) |

| TMB-high (IMvigor210) | HotHigh | Others | p-value |
|-----------------------|---------|--------|---------|
| ORR | 55.6% | 32.8% | 0.034 |

## Citation

If you use this code, please cite:
> [논문 citation 정보]

## License

MIT License

## Contact

[이메일 주소]
