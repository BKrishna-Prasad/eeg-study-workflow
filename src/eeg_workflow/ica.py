"""ICA fitting, inspection materials and reproducible exclusion decisions."""

from datetime import datetime, timezone
from pathlib import Path
import warnings

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne.preprocessing import ICA

from .io import read_json, sha256, write_json


def fit_ica(raw, settings):
    """Fit extended Infomax to a high-passed copy, excluding marked channels."""
    fit = raw.copy().filter(settings.get("fit_l_freq", 1.0), None, picks="eeg")
    rank = mne.compute_rank(fit, rank=None)["eeg"]
    count = settings["n_components"]
    if count is None:
        count = rank
    if isinstance(count, bool) or not isinstance(count, int) or not 2 <= count <= rank:
        raise ValueError(f"n_components must be an integer between 2 and EEG rank {rank}")
    model = ICA(n_components=count, max_iter="auto", random_state=settings.get("seed", 97),
                method="infomax", fit_params={"extended": True})
    model.fit(fit, picks="eeg", reject_by_annotation=True)
    return model, fit


def iclabel_suggestions(fit, model, threshold=0.8):
    """Return advisory labels; preserve and expose classifier assumptions."""
    from mne_icalabel import label_components
    warnings.warn("ICLabel recommends common average reference and 1–100 Hz fitting data. "
                  "Check the configured reference and filters; classifier scores require inspection.", UserWarning)
    result = label_components(fit, model, method="iclabel")
    return pd.DataFrame({"component": range(model.n_components_), "label": result["labels"],
                         "score": np.asarray(result["y_pred_proba"], dtype=float),
                         "suggested": [label in ("eye blink", "muscle artifact", "heart beat") and score >= threshold
                                       for label, score in zip(result["labels"], result["y_pred_proba"])],
                         "method": "ICLabel advisory"})


def save_inspection(fit, model, output, label):
    """Save all component maps, source traces and spectra for offline review."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    figures = model.plot_components(show=False, title=f"{label}: ICA topographies")
    if not isinstance(figures, list):
        figures = [figures]
    for page, fig in enumerate(figures, 1):
        fig.savefig(output / f"components_{page}.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    sources = model.get_sources(fit)
    values = sources.get_data()
    limit = min(len(fit.times), int(15 * fit.info["sfreq"]))
    fig, axes = plt.subplots(model.n_components_, 1, figsize=(11, 1.4 * model.n_components_), sharex=True, squeeze=False)
    for i, ax in enumerate(axes[:, 0]):
        ax.plot(fit.times[:limit], values[i, :limit], lw=0.6); ax.set_ylabel(f"IC {i}")
    axes[-1, 0].set_xlabel("Time from prepared recording start (s)")
    fig.suptitle(f"{label}: ICA source preview (first 15 s)"); fig.tight_layout()
    fig.savefig(output / "sources.png", dpi=120); plt.close(fig)
    from scipy.signal import welch
    freqs, psd = welch(values, fs=fit.info["sfreq"], nperseg=min(int(2 * fit.info["sfreq"]), values.shape[1]))
    fig, axes = plt.subplots(model.n_components_, 1, figsize=(10, 1.4 * model.n_components_), sharex=True, squeeze=False)
    for i, ax in enumerate(axes[:, 0]):
        ax.semilogy(freqs, psd[i]); ax.set_ylabel(f"IC {i}")
        ax.set_xlim(0, min(fit.info["lowpass"], fit.info["sfreq"] / 2))
    axes[-1, 0].set_xlabel("Frequency (Hz); ICA source scale is arbitrary")
    fig.suptitle(f"{label}: ICA source spectra"); fig.tight_layout()
    fig.savefig(output / "source_psd.png", dpi=120); plt.close(fig)


def bound_files(run):
    run = Path(run)
    return {name: sha256(run / name) for name in ("prepared-raw.fif", "model-ica.fif", "config.json", "manifest.json")}


def save_review(run, exclusions, reviewer, reason, demonstration_only=False):
    """Persist a final decision tied to this data/model/configuration only."""
    run = Path(run)
    if (run / "ica_review.json").exists():
        raise FileExistsError("A review already exists; use a new run directory for a revised decision")
    manifest = read_json(run / "manifest.json")
    n_components = manifest["n_components"]
    exclusions = list(exclusions)
    if any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < n_components for i in exclusions):
        raise ValueError("Rejected components must be valid zero-based integer indices")
    if len(set(exclusions)) != len(exclusions):
        raise ValueError("Rejected component indices must not repeat")
    if len(exclusions) == n_components:
        raise ValueError("Rejecting every fitted component requires a separate protocol review")
    if not reviewer.strip() or not reason.strip():
        raise ValueError("Reviewer and reason are required, including for an empty exclusion list")
    if demonstration_only and not manifest["demonstration_only"]:
        raise ValueError("A demonstration decision cannot be used for research data")
    decision = {"schema_version": 1, "reviewed": True,
                "decision_kind": "scripted_synthetic_example" if demonstration_only else "human_review",
                "reviewer": reviewer, "reason": reason,
                "excluded_components": sorted(exclusions), "n_components": n_components,
                "demonstration_only": manifest["demonstration_only"], "bound_files": bound_files(run),
                "reviewed_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(run / "ica_review.json", decision)
    return decision


def apply_review(run):
    """Reject missing, stale or edited review decisions before applying ICA."""
    run = Path(run)
    decision = read_json(run / "ica_review.json")
    manifest = read_json(run / "manifest.json")
    if decision.get("reviewed") is not True or decision.get("bound_files") != bound_files(run):
        raise ValueError("ICA decision is unreviewed or bound to different data/model/configuration")
    if decision.get("demonstration_only") != manifest["demonstration_only"]:
        raise ValueError("ICA decision/data demonstration status differs")
    if decision.get("decision_kind") != "human_review" and not manifest["demonstration_only"]:
        raise ValueError("Research data require a human review")
    model = mne.preprocessing.read_ica(run / "model-ica.fif")
    exclusions = decision["excluded_components"]
    if (decision.get("n_components") != model.n_components_ or
        any(type(i) is not int or not 0 <= i < model.n_components_ for i in exclusions) or
        len(set(exclusions)) != len(exclusions) or len(exclusions) >= model.n_components_ or
        not decision.get("reviewer", "").strip() or not decision.get("reason", "").strip()):
        raise ValueError("Invalid persisted ICA review")
    raw = mne.io.read_raw_fif(run / "prepared-raw.fif", preload=True)
    model.exclude = exclusions
    return raw, model.apply(raw.copy()), decision
