"""Saved subject stages with an explicit ICA review gate."""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from .epochs import autoreject_epochs, make_epochs, validate_epoching
from .ica import apply_review, bound_files, fit_ica, iclabel_suggestions, save_inspection
from .io import read_json, sha256, software_versions, write_json
from .preprocessing import interpolate_and_reference, prepare_raw
from .spectral import band_features, morlet_power, roi_curves, to_db


def validate_config(config):
    """Reject unknown top-level sections and missing workflow parameters."""
    required = {"preprocessing", "ica", "epoching", "analysis", "autoreject"}
    if not required.issubset(config) or set(config) - required - {"xdf"}:
        raise ValueError("Config requires preprocessing, ica, epoching, analysis and autoreject; optional xdf")
    for section, keys in {"preprocessing": {"l_freq", "h_freq"}, "ica": {"n_components"},
                          "epoching": {"mode"},
                          "analysis": {"spectral_conditions", "feature_conditions", "spectral_rois",
                                       "feature_rois", "bands", "morlet_conditions", "morlet_channels",
                                       "morlet_freqs", "morlet_cycles_factor"}}.items():
        if not isinstance(config[section], dict) or not keys.issubset(config[section]):
            raise ValueError(f"Missing parameters in {section}")
    if type(config["autoreject"]) is not bool:
        raise ValueError("autoreject must be a JSON boolean")
    validate_epoching(config["epoching"])
    allowed = {"preprocessing": {"l_freq", "h_freq", "notch_hz", "montage", "rename_channels",
                                  "channel_types", "drop_channels", "crop_s", "bad_channels",
                                  "bad_channel_method", "seed", "interpolate_bads", "reference", "reference_stage"},
               "ica": {"n_components", "fit_l_freq", "seed"},
               "analysis": {"spectral_conditions", "feature_conditions", "spectral_rois", "feature_rois",
                            "bands", "morlet_conditions", "morlet_channels", "morlet_freqs",
                            "morlet_cycles_factor", "fmin", "fmax"}}
    for section, keys in allowed.items():
        if set(config[section]) - keys:
            raise ValueError(f"Unknown {section} keys: {set(config[section]) - keys}")
    if config["preprocessing"].get("reference_stage", "after_ica") not in ("before_ica", "after_ica"):
        raise ValueError("reference_stage must be before_ica or after_ica")
    if type(config["preprocessing"].get("interpolate_bads", True)) is not bool:
        raise ValueError("interpolate_bads must be a JSON boolean")
    for field in ("spectral_conditions", "feature_conditions", "spectral_rois", "feature_rois", "bands"):
        if not config["analysis"][field]:
            raise ValueError(f"Configure at least one {field}")
    if not 0 <= config["analysis"].get("fmin", 1) < config["analysis"].get("fmax", 40):
        raise ValueError("Analysis requires 0 <= fmin < fmax")
    return config


def prepare_run(raw, config, output, subject_id, demonstration_only=False, input_path=None,
                suggest_iclabel=False, synthetic_suggestions=False):
    """Save a fitted model and inspection figures without applying exclusions."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", subject_id):
        raise ValueError("Use a pseudonymous subject ID containing letters, digits, underscores or hyphens")
    validate_config(config)
    if (raw.info.get("description") or "").startswith("DEMONSTRATION ONLY") and not demonstration_only:
        raise ValueError("This input is labelled synthetic; use demonstration_only=True")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new, empty output directory for each run")
    output.mkdir(parents=True, exist_ok=True)
    prepared, qc, bads = prepare_raw(raw, config["preprocessing"])
    if demonstration_only:
        prepared.info["description"] = "DEMONSTRATION ONLY: public workflow example; no research finding"
    model, fit = fit_ica(prepared, config["ica"])
    prepared.save(output / "prepared-raw.fif", fmt="double", overwrite=False)
    model.save(output / "model-ica.fif", overwrite=False)
    write_json(output / "config.json", config)
    manifest = {"schema_version": 1, "subject_id": subject_id, "demonstration_only": demonstration_only,
                "output_label": "DEMONSTRATION ONLY" if demonstration_only else f"Subject {subject_id}",
                "input_basename": Path(input_path).name if input_path else "in-memory Raw object",
                "input_sha256": sha256(input_path) if input_path else None,
                "n_channels": len(prepared.ch_names), "n_samples": prepared.n_times,
                "first_samp": int(prepared.first_samp), "sfreq": prepared.info["sfreq"],
                "n_components": model.n_components_, "bad_channel_report": bads,
                "versions": software_versions()}
    write_json(output / "manifest.json", manifest)
    qc["demonstration_only"] = demonstration_only; qc.to_csv(output / "channel_qc.csv", index=False)
    label = manifest["output_label"]
    save_inspection(fit, model, output / "inspection", label)
    if synthetic_suggestions:
        if not demonstration_only:
            raise ValueError("Synthetic trace suggestions are only available for the demonstration")
        from .demo import synthetic_blink_suggestions
        suggestions = synthetic_blink_suggestions(fit, model)
    elif suggest_iclabel:
        suggestions = iclabel_suggestions(fit, model)
    else:
        suggestions = pd.DataFrame(columns=["component", "label", "score", "suggested", "method"])
    suggestions["demonstration_only"] = demonstration_only
    suggestions.to_csv(output / "ica_suggestions.csv", index=False)
    return suggestions


def finish_run(run):
    """Apply a reviewed exclusion list, interpolate/reference, then segment."""
    run = Path(run)
    if (run / "cleaned-raw.fif").exists():
        raise FileExistsError("This run is already finished; use a new run for revised decisions")
    config, manifest = read_json(run / "config.json"), read_json(run / "manifest.json")
    before, after_ica, decision = apply_review(run)
    cleaned = interpolate_and_reference(after_ica, config["preprocessing"])
    epochs, segmentation = make_epochs(cleaned, config["epoching"])
    if config["autoreject"]:
        epochs, rejection, report = autoreject_epochs(epochs)
        np.savez_compressed(run / "autoreject_log.npz", labels=rejection.labels, bad_epochs=rejection.bad_epochs,
                            demonstration_only=manifest["demonstration_only"])
        fig = rejection.plot("horizontal", show=False)
        fig.suptitle(manifest["output_label"] + ": AutoReject log")
        fig.savefig(run / "autoreject.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    else:
        report = {"enabled": False, "note": "AutoReject omitted from this configured run"}
    if not len(epochs):
        raise ValueError("No epochs remain after artifact rejection")
    cleaned.save(run / "cleaned-raw.fif", fmt="double", overwrite=False)
    epochs.save(run / "clean-epo.fif", fmt="double", overwrite=False)
    write_json(run / "processing_log.json", {"demonstration_only": manifest["demonstration_only"],
                                              "ica_review_sha256": sha256(run / "ica_review.json"),
                                              "excluded_components": decision["excluded_components"],
                                              "segmentation": segmentation, "autoreject": report,
                                              "reference": config["preprocessing"].get("reference", "average"),
                                              "reference_stage": config["preprocessing"].get("reference_stage", "after_ica"),
                                              "cleaned_sha256": sha256(run / "cleaned-raw.fif"),
                                              "epochs_sha256": sha256(run / "clean-epo.fif")})
    # Compare ICA removal alone so the reference change does not mask it.
    channel = next(ch for ch in before.ch_names if ch in after_ica.ch_names and ch == "Fp1") if "Fp1" in before.ch_names else before.ch_names[0]
    limit = min(before.n_times, int(15 * before.info["sfreq"]))
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(before.times[:limit], before.get_data(picks=[channel])[0, :limit] * 1e6, label="Before ICA", alpha=0.8)
    ax.plot(before.times[:limit], after_ica.get_data(picks=[channel])[0, :limit] * 1e6, label="After reviewed ICA", lw=1)
    ax.set(xlabel="Time from prepared recording start (s)", ylabel=f"{channel} (µV)",
           title=manifest["output_label"] + ": ICA removal before final interpolation/reference")
    ax.legend(); fig.tight_layout(); fig.savefig(run / "ica_comparison.png", dpi=140); plt.close(fig)
    return epochs


def analyse_run(run):
    """Export subject features, linear PSD/TFR arrays and labelled figures."""
    run = Path(run)
    manifest, config = read_json(run / "manifest.json"), read_json(run / "config.json")
    log = read_json(run / "processing_log.json")
    if log["epochs_sha256"] != sha256(run / "clean-epo.fif"):
        raise ValueError("Epoch file changed after processing")
    if (log["ica_review_sha256"] != sha256(run / "ica_review.json") or
        read_json(run / "ica_review.json")["bound_files"] != bound_files(run)):
        raise ValueError("Review, model, data or configuration changed after processing")
    epochs = mne.read_epochs(run / "clean-epo.fif", preload=True)
    settings = config["analysis"]; demo = manifest["demonstration_only"]; subject = manifest["subject_id"]
    bounds = {"fmin": settings.get("fmin", 1.0), "fmax": settings.get("fmax", 40.0)}
    curves = roi_curves(epochs, subject, settings["spectral_conditions"], settings["spectral_rois"], demo, **bounds)
    features = band_features(epochs, subject, settings["feature_conditions"], settings["feature_rois"], settings["bands"], demo, **bounds)
    curves.to_csv(run / "psd_curves.csv", index=False); features.to_csv(run / "features.csv", index=False)
    rois = list(settings["spectral_rois"])
    fig, axes = plt.subplots(1, len(rois), figsize=(5 * len(rois), 4), squeeze=False)
    for roi, ax in zip(rois, axes[0]):
        for condition, table in curves[curves.roi == roi].groupby("condition", sort=False):
            ax.plot(table.frequency_hz, table.psd_db, label=condition)
        ax.set(title=roi, xlabel="Frequency (Hz)", ylabel="PSD (dB re 1 V²/Hz)"); ax.legend()
    fig.suptitle(manifest["output_label"] + ": Welch PSD"); fig.tight_layout()
    fig.savefig(run / "psd.png", dpi=140); plt.close(fig)
    tfr_values = []
    for condition in settings["morlet_conditions"]:
        tfr = morlet_power(epochs, condition, settings["morlet_channels"],
                           settings["morlet_freqs"], settings["morlet_cycles_factor"])
        slug = re.sub(r"[^A-Za-z0-9_-]", "_", condition)
        np.savez_compressed(run / f"tfr_{slug}.npz", power_v2=tfr.data, freqs_hz=tfr.freqs,
                            times_s=tfr.times, channels=np.array(tfr.ch_names),
                            demonstration_only=demo, condition=condition)
        tfr_values.append((condition, tfr, to_db(tfr.data.mean(axis=0))))
    if tfr_values:
        lower = min(v.min() for _, _, v in tfr_values); upper = max(v.max() for _, _, v in tfr_values)
        fig, axes = plt.subplots(1, len(tfr_values), figsize=(6 * len(tfr_values), 4), squeeze=False)
        for ax, (condition, tfr, values) in zip(axes[0], tfr_values):
            im = ax.pcolormesh(tfr.times, tfr.freqs, values, shading="auto", cmap="viridis", vmin=lower, vmax=upper)
            ax.set(title=condition, xlabel="Epoch time (s)", ylabel="Frequency (Hz)")
            fig.colorbar(im, ax=ax, label="Power (dB re 1 V²)")
        fig.suptitle(manifest["output_label"] + ": Morlet power, no baseline correction")
        fig.tight_layout(); fig.savefig(run / "tfr.png", dpi=140); plt.close(fig)
    write_json(run / "subject_summary.json", {"subject_id": subject, "demonstration_only": demo,
               "label": manifest["output_label"], "n_epochs": len(epochs),
               "counts_by_event_tag": {key: len(epochs[key]) for key in epochs.event_id},
               "n_feature_rows": len(features), "metrics": features.metric.unique().tolist(),
               "analysis_config_sha256": sha256(run / "config.json"),
               "processing_log_sha256": sha256(run / "processing_log.json"),
               "ica_review_sha256": sha256(run / "ica_review.json"), "versions": software_versions()})
    return features
