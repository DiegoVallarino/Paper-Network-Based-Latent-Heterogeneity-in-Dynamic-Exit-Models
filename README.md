# Replication package

This repository contains the replication package for:

**Diego Vallarino**  
**Network-Based Latent Heterogeneity in Dynamic Exit Models: Evidence from U.S. Portland Cement**  
April 2026

## What is included

The package includes the manuscript PDF, the main Python replication pipeline, the exported figures, and the exported tables currently available in the working directory you shared.

The package is organized so it can be uploaded directly to GitHub.

## Important note on full reproducibility

The main script depends on the raw input file:

`data/raw/data_ryan_java.csv`

That file was **not included** among the materials provided here. Because of that, this package is fully suitable for:

1. archiving the current paper, code, tables, and figures;
2. sharing the replication structure on GitHub;
3. allowing a full rerun as soon as you add the raw file in the indicated folder.

In other words, this is a **GitHub-ready reproduction package with one missing raw input file**.

## Folder structure

```text
replication-package/
├── README.md
├── requirements.txt
├── .gitignore
├── CITATION.cff
├── run_replication_windows.bat
├── run_replication_unix.sh
├── paper/
│   └── Network_Based_Latent_Heterogeneity.pdf
├── scripts/
│   ├── replication_pipeline.py
│   └── verify_package.py
├── data/
│   ├── raw/
│   │   └── PLACE_DATA_HERE.txt
│   └── processed/
├── results/
│   ├── figures/
│   └── tables/
└── docs/
    ├── GITHUB_STEP_BY_STEP.md
    ├── REPRODUCIBILITY_NOTES.md
    └── PACKAGE_CONTENTS.md
```

## Quick start

### Option A. Archive only
If your immediate goal is to publish the paper, code, tables, and figures on GitHub, you can upload this package as-is.

### Option B. Full rerun
1. Put `data_ryan_java.csv` inside `data/raw/`
2. Create a virtual environment
3. Install the dependencies
4. Run the pipeline

Windows:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts\replication_pipeline.py
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/replication_pipeline.py
```

## Verify the package

You can run:

```bash
python scripts/verify_package.py
```

This checks that the expected paper, code, figures, and tables are present.

## Main outputs already included

### Figures
- Figure 1: Embedding characterization
- Figure 2: Coefficient comparison
- Figure 3: Hazard profiles
- Figure 4: Bootstrap distributions

### Tables
- Table 1: Summary statistics
- Table 2: Annual exit rates
- Table 3: Market heterogeneity
- Table 4: Baseline hazard
- Table 5: Embedding-augmented hazard
- Table 6: Model comparison
- Table 7: Fixed-effects benchmark
- Table 8: Robustness
- Bootstrap and appendix correction tables

## Suggested GitHub repository name

`network-latent-heterogeneity-replication`

## Citation

See `CITATION.cff`.
