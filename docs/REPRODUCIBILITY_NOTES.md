# Reproducibility notes

## Current status
This package is close to full reproducibility, but one raw input file is still missing:

`data/raw/data_ryan_java.csv`

## What works now
- The repository can be uploaded directly to GitHub.
- The paper, script, figures, and CSV outputs are organized and documented.
- Another researcher can inspect all current outputs.
- The pipeline can be executed immediately after the missing raw file is added.

## What to do next
1. Add `data_ryan_java.csv` to `data/raw/`
2. Create a clean virtual environment
3. Install `requirements.txt`
4. Run `python scripts/replication_pipeline.py`
5. Compare the regenerated outputs with the archived outputs in `results/`

## Recommended final check before public release
- Open the paper PDF and verify that every figure and table cited in the manuscript exists in the repository
- Confirm that no confidential or proprietary files are present
- Replace the placeholder GitHub URL inside `CITATION.cff`
- Decide whether you want to add a license file
