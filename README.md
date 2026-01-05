# python-gppi

A python-based gPPI (generalized psychophysiological interaction) first-level analysis
implemented in Python using Nilearn and NiBabel.

## Overview 

This repository implements a first-level gPPI pipeline that:

- Defines seed regions of interest and extracts confound-cleaned seed time series.
- Constructs PPI interaction regressors (phys × psych) using ridge regression deconvultion.
- Fits a voxel-wise GLM to each run and saves beta (effect-size) maps for psychological, physiological, and PPI (interaction) regressors.

## Analysis details 

**Main regressors:**
1. Psychological — event regressors (convolved with HRF)
2. Physiological — seed BOLD time series, cleaned from motion and physiological compounds and high pass filtered
3. Interaction — deconvolved seed neural activity × mean-centred psychological, re-convolved with HRF
4. Confounds — motion and physiological confounds from fMRIPrep

**Implementation notes:**
- The pipeline uses `nilearn.glm.FirstLevelModel` for model fitting and `make_first_level_design_matrix` to assemble regressors.
- A deconvolution helper ([from IHB-IBR](https://github.com/IHB-IBR-department/BOLD_deconvolution)) is used to obtain neuronal estimates from seed time series before forming PPI interaction terms.
- Default constants (such as `TR`, `SEEDS`, and participant lists) are set in `python_gppi/main.py` and must be adapted for your dataset.

## Quickstart — run the analysis 

Install dependencies (examples):

```bash
# with uv (recommended in this repo)
uv run python_gppi/main.py
```
Alternatively, use your standard pip/conda workflow.

Before running, edit `python_gppi/main.py` to adjust file paths and constants
(e.g., `FMRIPREP_ROOT`, `ROI_DIR`, `TESTS`, `PARTICIPANTS`). The script expects
fMRIPrep outputs and per-run `events_files` CSVs as described in the code.

## References & Resources 

- Dartbrains connectivity: https://dartbrains.org/content/Connectivity.html
- Mumford gPPI videos: https://www.youtube.com/watch?v=M8APlF6oBgA
- SPM manual: https://www.fil.ion.ucl.ac.uk/spm/doc/spm12_manual.pdf
- BOLD deconvolution library used for deconvolution step:
  https://github.com/IHB-IBR-department/BOLD_deconvolution


