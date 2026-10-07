"""Commands for subject preparation, ICA review, analysis and group summaries."""

import argparse
import re
from pathlib import Path

import mne
import pandas as pd
import matplotlib.pyplot as plt

from .demo import demo_config, synthetic_raw
from .group_analysis import grand_average, hemispheric_comparison, load_features, paired_comparison
from .ica import save_review
from .io import read_json, read_raw, sha256, software_versions, write_json
from .workflow import analyse_run, finish_run, prepare_run


def run_demo(output, seed=97, subject_id="demo-001", config=None):
    """Run a scripted synthetic example, clearly distinguished from human review."""
    config = demo_config() if config is None else config
    config["ica"]["seed"] = seed
    config["preprocessing"]["seed"] = seed
    raw = synthetic_raw(seed)
    suggestions = prepare_run(raw, config, output, subject_id, demonstration_only=True, synthetic_suggestions=True)
    rejected = suggestions.loc[suggestions.suggested, "component"].astype(int).tolist()
    save_review(output, rejected, "scripted synthetic demonstration",
                "DEMONSTRATION ONLY: selected by correlation with the generated blink trace; no human review or study finding.",
                demonstration_only=True)
    finish_run(output); analyse_run(output)
    return Path(output)


def run_group(paths, output, condition_a, condition_b, curve_paths=None, metrics=None, hemispheres=False):
    """Export paired statistics, participant trajectories and optional grand PSD."""
    data = load_features(paths)
    metrics = ([m for m in data.metric.unique() if m.endswith("_ln_mean_psd") or m == "FAA"]
               if metrics is None else list(metrics))
    if not metrics:
        raise ValueError("No log mean PSD or FAA metrics found")
    if not set(metrics).issubset(data.metric):
        raise ValueError("Requested metrics are absent from the feature tables")
    result = paired_comparison(data, condition_a, condition_b, metrics)
    hemisphere = hemispheric_comparison(data, [condition_a, condition_b]) if hemispheres else None
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new, empty group output directory")
    output.mkdir(parents=True, exist_ok=True)
    data.to_csv(output / "participant_features.csv", index=False)
    result.to_csv(output / "paired_tests.csv", index=False)
    if hemisphere is not None and not hemisphere.empty:
        hemisphere.to_csv(output / "hemispheric_tests.csv", index=False)
    label = "DEMONSTRATION ONLY" if bool(data.demonstration_only.iloc[0]) else "EXPLORATORY RESEARCH ANALYSIS"
    for metric in metrics:
        subset = data[data.metric == metric]
        if subset.empty:
            continue
        wide = subset.pivot(index="subject_id", columns="condition", values="value").reindex(columns=[condition_a, condition_b]).dropna()
        if wide.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        for _, row in wide.iterrows():
            ax.plot([0, 1], row, "o-", alpha=0.45, lw=1)
        ax.plot([0, 1], wide.mean(), "D-", color="black", lw=2, label="Participant mean")
        ax.set_xticks([0, 1], [condition_a, condition_b])
        ax.set(title=f"{label}: {metric} (n={len(wide)})", ylabel=subset.units.iloc[0]); ax.legend()
        slug = re.sub(r"[^A-Za-z0-9_-]", "_", metric)
        fig.tight_layout(); fig.savefig(output / f"{slug}.png", dpi=140); plt.close(fig)
    if curve_paths:
        curves = pd.concat([pd.read_csv(p, dtype={"subject_id": str, "condition": str, "roi": str})
                            for p in curve_paths], ignore_index=True)
        if set(curves.subject_id) != set(data.subject_id) or set(curves.demonstration_only) != set(data.demonstration_only):
            raise ValueError("Curve and feature subject IDs or demonstration status differ")
        summary = grand_average(curves)
        summary.to_csv(output / "grand_average_psd.csv", index=False)
        rois = sorted(summary.roi.unique())
        fig, axes = plt.subplots(1, len(rois), figsize=(5 * len(rois), 4), squeeze=False)
        for roi, ax in zip(rois, axes[0]):
            for condition, table in summary[summary.roi == roi].groupby("condition", sort=False):
                x, y, error = table.frequency_hz.to_numpy(), table.mean_subject_db.to_numpy(), table.sem_subject_db.to_numpy()
                ax.plot(x, y, label=condition); ax.fill_between(x, y-error, y+error, alpha=0.15)
            ax.set(title=roi, xlabel="Frequency (Hz)", ylabel="Subject mean ± SEM (dB re 1 V²/Hz)"); ax.legend()
        fig.suptitle(label + ": grand average PSD"); fig.tight_layout()
        fig.savefig(output / "grand_average_psd.png", dpi=140); plt.close(fig)
    write_json(output / "group_summary.json", {"label": label, "demonstration_only": bool(data.demonstration_only.iloc[0]),
               "n_subjects": int(data.subject_id.nunique()), "conditions": [condition_a, condition_b],
               "inference": "Exploratory paired t-tests and two-sided Wilcoxon; unadjusted p-values; participants are the unit of analysis",
               "source_files": [{"basename": Path(p).name, "sha256": sha256(p)} for p in paths],
               "curve_files": [{"basename": Path(p).name, "sha256": sha256(p)} for p in (curve_paths or [])],
               "versions": software_versions()})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="EEG preprocessing and subject-to-group analysis with MNE")
    subs = parser.add_subparsers(dest="command", required=True)
    demo = subs.add_parser("demo", help="DEMONSTRATION ONLY: scripted end-to-end synthetic example")
    demo.add_argument("--output", required=True); demo.add_argument("--seed", type=int, default=97)
    demo.add_argument("--subject", default="demo-001"); demo.add_argument("--config")
    prepare = subs.add_parser("prepare", help="Fit ICA and save inspection materials; stop for review")
    prepare.add_argument("--input", required=True); prepare.add_argument("--config", required=True)
    prepare.add_argument("--output", required=True); prepare.add_argument("--subject", required=True)
    prepare.add_argument("--suggest-iclabel", action="store_true")
    prepare.add_argument("--demonstration-only", action="store_true", help="Label a synthetic input and every output")
    review = subs.add_parser("review", help="Record final zero-based exclusions after inspection")
    review.add_argument("--run", required=True); review.add_argument("--exclude", required=True, help="Comma-separated indices, or none")
    review.add_argument("--reviewer", required=True); review.add_argument("--reason", required=True)
    for name in ("finish", "analyse"):
        command = subs.add_parser(name); command.add_argument("--run", required=True)
    group = subs.add_parser("group", help="Aggregate subject features and run paired condition comparisons")
    group.add_argument("--features", nargs="+", required=True); group.add_argument("--curves", nargs="+")
    group.add_argument("--metrics", nargs="+", help="Selected metric names; defaults to all log mean PSD features and FAA")
    group.add_argument("--hemispheres", action="store_true", help="Also compare Left_Frontal and Right_Frontal log PSD within each condition")
    group.add_argument("--output", required=True); group.add_argument("--condition-a", required=True); group.add_argument("--condition-b", required=True)
    args = parser.parse_args(argv)
    mne.set_log_level("WARNING")
    if args.command == "demo":
        run_demo(args.output, args.seed, args.subject, read_json(args.config) if args.config else None)
    elif args.command == "prepare":
        config = read_json(args.config)
        raw = read_raw(args.input, config.get("xdf"))
        prepare_run(raw, config, args.output, args.subject, input_path=args.input,
                    demonstration_only=args.demonstration_only, suggest_iclabel=args.suggest_iclabel)
    elif args.command == "review":
        indices = [] if args.exclude.lower() == "none" else [int(i.strip()) for i in args.exclude.split(",")]
        save_review(args.run, indices, args.reviewer, args.reason)
    elif args.command == "finish":
        finish_run(args.run)
    elif args.command == "analyse":
        analyse_run(args.run)
    elif args.command == "group":
        run_group(args.features, args.output, args.condition_a, args.condition_b, args.curves, args.metrics, args.hemispheres)
    print("Completed", args.command)
