# coding: utf-8
# Author: Salma Elnagar

"""gPPI first-level analysis.

See `README.md` for the high-level description, analysis steps, references,
and usage instructions.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.glm.first_level import make_first_level_design_matrix, FirstLevelModel
from nilearn.glm.first_level.hemodynamic_models import spm_hrf
from nilearn.maskers import NiftiMasker
from nilearn.masking import intersect_masks
from nilearn.image import resample_to_img
# Deconvolution helper (make sure the repo is cloned)
sys.path.append(
    "path_to_BoldDeconvolution_repo_from_IHB-IBR"
)
from bold_deconvolution import ridge_regress_deconvolution

# Global variables
TR = 1.5
MICROTIME_RESOLUTION = 16
FMRIPREP_ROOT = Path('/Path_to_fMRI_prep_Data/')
N_RUNS = 5
TESTS = ['test1', 'test2']
GLM_TYPES = ['Expectancy', 'Memory', 'ExpectMem']
PARTICIPANTS = range(1,66) 
SEEDS = {'Seed_name': 'Seed_file_name.nii'}

# Functions
def frame_times_grid(n_scans, tr):
    """Return explicit frame (volume) acquisition times."""
    return np.arange(n_scans) * tr

def load_events_for_run(main_dir, participant, run):
    """Load per-run events CSV.

    Returns (events_df, path) where events_df contains 'onset', 'duration', 'trial_type' as is expected for nilearn events.
    """
    evt_path = (
        main_dir
        / f"events_files/sbj{participant:01d}_run{run:01d}_onsets.csv"
    )
    if not evt_path.exists():
        raise FileNotFoundError(
            f"Events file not found for sbj{participant}: {evt_path}"
        )

    events_df = pd.read_csv(evt_path)
    required_cols = {"onset", "duration", "trial_type"}
    if not required_cols.issubset(set(events_df.columns)):
        raise ValueError(
            f"Events CSV missing required columns {required_cols}. Found: {list(events_df.columns)}"
        )
    return events_df, evt_path


def load_fmriprep_confounds(fmriprep_root, participant, run, include_fd=True):
    """Load selected motion and physiological confounds from fMRIPrep TSV file.
    Optionally add frame displacement.

    Returns (confounds_df, confound_names).
    """
    tsv_path = (
        fmriprep_root
        / f"sub-{participant:02d}"
        / "ses-02"
        / "func"
        / f"sub-{participant:02d}_ses-02_task-search2_run-{run:02d}_desc-confounds_timeseries.tsv"
    )
    if not tsv_path.exists():
        raise FileNotFoundError(
            f"Confounds file not found for sbj{participant}: {tsv_path}"
        )

    confounds_df = pd.read_csv(tsv_path, sep="\t")

    motion_names = [
        "trans_x",
        "trans_x_derivative1",
        "trans_x_derivative1_power2",
        "trans_x_power2",
        "trans_y",
        "trans_y_derivative1",
        "trans_y_derivative1_power2",
        "trans_y_power2",
        "trans_z",
        "trans_z_derivative1",
        "trans_z_derivative1_power2",
        "trans_z_power2",
        "rot_x",
        "rot_x_derivative1",
        "rot_x_derivative1_power2",
        "rot_x_power2",
        "rot_y",
        "rot_y_derivative1",
        "rot_y_derivative1_power2",
        "rot_y_power2",
        "rot_z",
        "rot_z_derivative1",
        "rot_z_derivative1_power2",
        "rot_z_power2",
    ]

    physio_names = [
        "global_signal",
        "global_signal_derivative1",
        "csf",
        "csf_derivative1",
        "white_matter",
        "white_matter_derivative1",
    ]

    motion_cols = confounds_df.loc[:, motion_names]
    physio_cols = confounds_df.loc[:, physio_names]

    confounds = pd.concat([motion_cols, physio_cols], axis=1)
    if include_fd and "framewise_displacement" in confounds_df.columns:
        confounds["framewise_displacement"] = confounds_df["framewise_displacement"]

    return confounds.fillna(0), list(confounds.columns)


def extract_seed_timeseries(func_img, fmriprep_root, participant, run, seed_path, tr):
    """Extract mean ROI time series and optionally clean with confounds and high pass filter.
    Average across ROI, but alternatively you can get the eigenvariate like SPM.
    Returns a 1D array (n_scans,).
    """
    add_regs, _ = load_fmriprep_confounds(fmriprep_root, participant, run)

    seed_img_resampled = resample_to_img(
        seed_path,
        func_img,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )
    masker = NiftiMasker(mask_img=seed_img_resampled, high_pass=1/128, t_r=tr)

    seed_2d = masker.fit_transform(func_img, confounds=add_regs)

    return seed_2d.mean(axis=1)


def build_psych_term(condition, n_scans, events_df):
    """Build psychological events regressor in microtime grid.

    Returns (psych_in_microtime, timeseries_in_microtime).
    """
    dt = TR / MICROTIME_RESOLUTION
    timeseries_in_microtime = np.arange(0.0, n_scans * TR, dt, dtype=float)
    psych_in_microtime = np.zeros_like(timeseries_in_microtime, dtype=float)

    rows = events_df[events_df["trial_type"] == condition]
    for _, row in rows.iterrows():
        t0 = float(row["onset"])
        t1 = t0 + float(row["duration"])
        in_evt = (timeseries_in_microtime >= t0) & (timeseries_in_microtime < t1)
        if not in_evt.any():
            in_evt[np.argmin(np.abs(timeseries_in_microtime - t0))] = True
        psych_in_microtime[in_evt] = 1.0

    psych_in_microtime -= psych_in_microtime.mean()
    return psych_in_microtime, timeseries_in_microtime


def run_gppi_for_participant(test, glm_type, seed_name, seed_path, participant):
    """Process each participant for a single (test, GLM, seed).

    This function performs the per-run assembly and multi-run GLM fit, then
    writes contrast images to disk.
    """
    participant_2d = f"{participant:02d}"
    main_dir = Path(f"/main_dir_for_results/")
    sbj_out = main_dir / "gPPI" / glm_type / "FirstLevel" / seed_name / participant_2d
    sbj_out.mkdir(parents=True, exist_ok=True)

    print(f"Running {test}/{glm_type}/{seed_name} for P{participant_2d}")

    imgs = []
    dms_all = []
    all_conditions_for_ppi = set()
    mask_imgs = []

    def _func_file_paths(participant_id, run_id):
        """Return the bold and brain-mask paths for a participant/run as Path objects."""
        func_dir = FMRIPREP_ROOT / f"sub-{participant_id}" / "ses-02" / "func"
        prefix = f"sub-{participant_id}_ses-02_task-search2_run-{run_id}_space-MNI152NLin6Asym_desc-"
        bold = func_dir / f"{prefix}preproc_bold.nii.gz"
        mask = func_dir / f"{prefix}brain_mask.nii.gz"
        return bold, mask

    for run in range(1, N_RUNS + 1):
        run_2d = f"{run:02d}"
        bold_path, brain_mask = _func_file_paths(participant_2d, run_2d)

        if not bold_path.exists():
            print(f"[SKIP] Missing BOLD: {bold_path}")
            continue
        if not brain_mask.exists():
            print(f"[SKIP] Missing mask: {brain_mask}")
            continue

        func_img = nib.load(bold_path)
        n_scans = func_img.shape[-1]
        frame_times = frame_times_grid(n_scans) + TR * 0.5

        events_df, _ = load_events_for_run(main_dir, participant, run)

        events_df = events_df[~events_df["trial_type"].isin(["fixation"])]
        conditions = [str(c) for c in events_df["trial_type"].unique().tolist()]

        confounds, confounds_names = load_fmriprep_confounds(
            FMRIPREP_ROOT, participant, run, include_fd=True
        )

        seed_ts = extract_seed_timeseries(
            func_img, FMRIPREP_ROOT, participant, run, seed_path, TR
        )
        neuronal = ridge_regress_deconvolution(seed_ts, TR, 0.005, MICROTIME_RESOLUTION)
        neuronal = (neuronal - neuronal.mean()) / neuronal.std(ddof=0)

        dm = make_first_level_design_matrix(
            frame_times,
            events=events_df,
            hrf_model="spm",
            drift_model="cosine",
            high_pass=1 / 128,
            add_regs=confounds,
            add_reg_names=confounds_names,
        )

        dm["phys_seed"] = seed_ts - seed_ts.mean()

        conditions = [c for c in conditions if (c in dm.columns and c != "motor")]
        ppi_cols = []
        for cond in conditions:
            psych_in_microtime, _ = build_psych_term(cond, n_scans, events_df)
            ppi = neuronal * psych_in_microtime
            hrf = spm_hrf(TR, oversampling=MICROTIME_RESOLUTION)
            ppi_conv = np.convolve(ppi, hrf)[
                MICROTIME_RESOLUTION // 2 : n_scans * MICROTIME_RESOLUTION
                + MICROTIME_RESOLUTION // 2 : MICROTIME_RESOLUTION
            ]
            ppi_col = f"ppi_{cond}"
            dm[ppi_col] = ppi_conv
            ppi_cols.append(ppi_col)

        if "constant" not in dm.columns:
            dm["constant"] = 1.0

        for ppi_column in ppi_cols:
            dm[ppi_column] -= dm[ppi_column].mean() 
            ppi_stack = np.column_stack([dm[ppi_column].to_numpy() for ppi_column in ppi_cols])
            ppi_scale = ppi_stack.std(ddof=0)
            if ppi_scale > 0:
                for ppi_column in ppi_cols:
                    dm[ppi_column] /= ppi_scale
        
        for c in conditions:
            dm[c] = (dm[c] - dm[c].mean()) 

        dm["phys_seed"] = (dm["phys_seed"] - dm["phys_seed"].mean()) / dm[
            "phys_seed"
        ].std(ddof=0)

        ordered = (
            ["constant", "phys_seed"]
            + conditions
            + ppi_cols
            + [
                c
                for c in dm.columns
                if c not in (["constant", "phys_seed"] + conditions + ppi_cols)
            ]
        )
        X = dm[ordered].copy()

        imgs.append(func_img)
        dms_all.append(X)
        all_conditions_for_ppi.update(conditions)
        mask_imgs.append(nib.load(brain_mask))

    if len(imgs) == 0:
        print(f"[SKIP] No valid runs for participant {participant_2d}")
        return

    across_runs_brain_mask = intersect_masks(mask_imgs, threshold=0.2, connected=False)

    glm = FirstLevelModel(
        t_r=TR,
        noise_model="ar1",
        drift_model=None,
        high_pass=None,
        hrf_model=None,
        mask_img=across_runs_brain_mask,
        minimize_memory=False,
        signal_scaling=0,
        standardize=False,
        smoothing_fwhm=7.5,
    )
    glm = glm.fit(imgs, design_matrices=dms_all)

    dm_list = glm.design_matrices_

    def _save_contrast(vecs, filename_prefix: str):
        res_local = glm.compute_contrast(vecs, stat_type="t", output_type="all")
        for out_name, img in res_local.items():
            img.to_filename(str(sbj_out / f"{filename_prefix}_{out_name}.nii.gz"))

    # Save physiological main effect
    phys_vecs = [np.zeros(dm.shape[1]) for dm in dm_list]
    for v, dm in zip(phys_vecs, dm_list):
        if "phys_seed" in dm.columns:
            v[dm.columns.get_loc("phys_seed")] = 1.0
    _save_contrast(phys_vecs, "all_runs_phys_seed")

    # Psych and PPI main effects
    conditions_sorted = sorted(all_conditions_for_ppi)
    for cond in conditions_sorted:
        psych_vecs = [np.zeros(dm.shape[1]) for dm in dm_list]
        for v, dm in zip(psych_vecs, dm_list):
            if cond in dm.columns:
                v[dm.columns.get_loc(cond)] = 1.0
        _save_contrast(psych_vecs, f"all_runs_psych_{cond}")

    for cond in conditions_sorted:
        ppi_name = f"ppi_{cond}"
        ppi_vecs = [np.zeros(dm.shape[1]) for dm in dm_list]
        for v, dm in zip(ppi_vecs, dm_list):
            if ppi_name in dm.columns:
                v[dm.columns.get_loc(ppi_name)] = 1.0
        _save_contrast(ppi_vecs, f"all_runs_{ppi_name}")

    # Pairwise contrasts depending on GLM type
    def _get_pairs(gtype: str):
        if gtype == "Expectancy":
            return [
                ("congruent", "incongruent"),
                ("congruent", "unrelated"),
                ("incongruent", "unrelated"),
            ]
        if gtype == "Memory":
            return [("remember", "forget")]
        if gtype == "ExpectMem":
            return [
                ("congrem", "congforg"),
                ("incongrem", "incongforg"),
                ("unrrem", "unrforg"),
            ]
        return []

    pairs = _get_pairs(glm_type)

    for a, b in pairs:
        diff_vecs = []
        for dm in dm_list:
            v = np.zeros(dm.shape[1])
            if a in dm.columns:
                v[dm.columns.get_loc(a)] += 1.0
            if b in dm.columns:
                v[dm.columns.get_loc(b)] -= 1.0
            diff_vecs.append(v)
        res = glm.compute_contrast(diff_vecs, stat_type="t", output_type="all")
        for out_name, img in res.items():
            img.to_filename(str(sbj_out / f"all_runs_psych_{a}-{b}_{out_name}.nii.gz"))

    for a, b in pairs:
        name_a = f"ppi_{a}"
        name_b = f"ppi_{b}"
        diff_vecs = []
        for dm in dm_list:
            v = np.zeros(dm.shape[1])
            if name_a in dm.columns:
                v[dm.columns.get_loc(name_a)] += 1.0
            if name_b in dm.columns:
                v[dm.columns.get_loc(name_b)] -= 1.0
            diff_vecs.append(v)
        res = glm.compute_contrast(diff_vecs, stat_type="t", output_type="all")
        for out_name, img in res.items():
            img.to_filename(str(sbj_out / f"all_runs_ppi_{a}-{b}_{out_name}.nii.gz"))

def main() -> None:
    """Top-level script entry point.

    Runs gPPI analysis for every combination of tests, GLM types, seeds and participants.
    """
    for test in TESTS:
        for glm_type in GLM_TYPES:
            for seed_name, seed_path in SEEDS.items():
                for participant in PARTICIPANTS:
                    run_gppi_for_participant(
                        test, glm_type, seed_name, seed_path, participant
                    )

if __name__ == "__main__":
    main()
