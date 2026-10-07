"""Event-locked epochs and fixed windows within experimental blocks."""

import mne
import numpy as np
import pandas as pd


def validate_epoching(settings):
    """Validate a mode and its configuration before fitting ICA."""
    mode = settings.get("mode")
    allowed = {"events": {"mode", "annotation_map", "tmin", "tmax", "baseline"},
               "annotations": {"mode", "annotation_map", "length_s"},
               "intervals": {"mode", "intervals", "length_s"}}
    if mode not in allowed or set(settings) - allowed.get(mode, set()):
        raise ValueError("Epoching mode must be events, annotations or intervals, with only its documented keys")
    if mode in ("events", "annotations"):
        mapping = settings.get("annotation_map", {})
        if not isinstance(mapping, dict) or not mapping or any(
                not isinstance(k, str) or not k or not isinstance(v, str) or not v or
                not all(v.split("/")) for k, v in mapping.items()):
            raise ValueError("annotation_map must map recorded descriptions to non-empty condition labels")
        if any(k.lower().startswith(("bad", "edge")) for k in mapping):
            raise ValueError("BAD and EDGE annotations are reserved for rejection")
    if mode == "events":
        if not np.isfinite([settings.get("tmin", np.nan), settings.get("tmax", np.nan)]).all() or settings["tmin"] >= settings["tmax"]:
            raise ValueError("Event epochs require finite tmin < tmax")
        baseline = settings.get("baseline")
        if baseline is not None:
            if not isinstance(baseline, (list, tuple)) or len(baseline) != 2:
                raise ValueError("Baseline must be null or a two-item interval inside the epoch")
            start = settings["tmin"] if baseline[0] is None else baseline[0]
            stop = settings["tmax"] if baseline[1] is None else baseline[1]
            if not np.isfinite([start, stop]).all() or not settings["tmin"] <= start <= stop <= settings["tmax"]:
                raise ValueError("Baseline must lie inside the epoch")
    else:
        length = settings.get("length_s", 0)
        if not np.isfinite(length) or length <= 0:
            raise ValueError("Block windows require a positive finite length_s")
        if mode == "intervals" and not settings.get("intervals"):
            raise ValueError("Configure at least one labelled interval")
    return settings


def make_epochs(raw, settings):
    """Segment events or blocks; reject windows intersecting BAD annotations."""
    validate_epoching(settings)
    sfreq = raw.info["sfreq"]
    mode = settings["mode"]
    if mode == "events":
        labels = list(dict.fromkeys(settings["annotation_map"].values()))
        ids = {label: i + 1 for i, label in enumerate(labels)}
        descriptions = {key: ids[value] for key, value in settings["annotation_map"].items()}
        events, _ = mne.events_from_annotations(raw, event_id=descriptions)
        if not len(events):
            raise ValueError("No events match annotation_map")
        if len(set(events[:, 0])) != len(events):
            raise ValueError("Mapped events share sample onsets; resolve duplicate triggers")
        mapping = {label: code for label, code in ids.items() if code in events[:, 2]}
        epochs = mne.Epochs(raw, events, event_id=mapping, tmin=settings["tmin"],
                            tmax=settings["tmax"], baseline=settings.get("baseline"),
                            picks="eeg", preload=True, reject_by_annotation=True)
        tails = []
        generated = len(events)
    else:
        step = round(settings["length_s"] * sfreq)
        if step < 1 or not np.isclose(step, settings["length_s"] * sfreq):
            raise ValueError("Window duration must be a whole number of samples")
        if mode == "annotations":
            # MNE event samples include first_samp, including after a crop.
            events, descriptions = mne.events_from_annotations(raw)
            intervals = []
            for marker, condition in settings["annotation_map"].items():
                durations = raw.annotations.duration[raw.annotations.description == marker]
                onsets = events[events[:, 2] == descriptions.get(marker, -1), 0]
                if len(onsets) != len(durations):
                    raise ValueError(f"Cannot align annotation durations for {marker!r}")
                for onset, duration in zip(onsets, durations):
                    if duration <= 0:
                        raise ValueError(f"{marker!r} has no block duration; use events mode for point triggers")
                    intervals.append((condition, int(onset), int(onset + round(duration * sfreq))))
        else:
            intervals = []
            for row in settings["intervals"]:
                condition = row["condition"]
                start, stop = row["start_s"], row["stop_s"]
                if (not isinstance(condition, str) or not condition or not all(condition.split("/")) or
                        not np.isfinite([start, stop]).all() or not 0 <= start < stop):
                    raise ValueError("Intervals require a condition and finite 0 <= start_s < stop_s")
                intervals.append((condition, raw.first_samp + round(start * sfreq),
                                  raw.first_samp + round(stop * sfreq)))
        intervals.sort(key=lambda row: row[1])
        if not intervals:
            raise ValueError("No blocks match the configuration")
        for i, (_, start, stop) in enumerate(intervals):
            if not raw.first_samp <= start < stop <= raw.last_samp + 1:
                raise ValueError("Block lies outside the prepared recording")
            if i and start < intervals[i - 1][2]:
                raise ValueError("Configured blocks overlap; resolve their boundaries")
        labels = list(dict.fromkeys(condition for condition, _, _ in intervals))
        ids = {label: i + 1 for i, label in enumerate(labels)}
        rows, metadata, tails = [], [], []
        for block, (condition, start, stop) in enumerate(intervals):
            count = (stop - start) // step
            tails.append({"block": block, "condition": condition,
                          "discarded_tail_s": (stop - start - count * step) / sfreq})
            for onset in range(start, stop - step + 1, step):
                rows.append([onset, 0, ids[condition]])
                metadata.append({"block": block, "condition": condition, "start_sample": onset,
                                 "stop_sample_exclusive": onset + step})
        if not rows:
            raise ValueError("No complete windows fit in the configured blocks")
        rows = np.asarray(rows, dtype=int)
        mapping = {label: code for label, code in ids.items() if code in rows[:, 2]}
        epochs = mne.Epochs(raw, rows, event_id=mapping, tmin=0, tmax=(step - 1) / sfreq,
                            baseline=None, picks="eeg", preload=True,
                            metadata=pd.DataFrame(metadata), reject_by_annotation=True)
        generated = len(rows)
    if not len(epochs):
        raise ValueError("No epochs remain after boundary and annotation rejection")
    return epochs, {"mode": mode, "generated": generated, "kept_after_annotations": len(epochs),
                    "partial_windows": tails, "drop_log": [list(v) for v in epochs.drop_log]}


def autoreject_epochs(epochs, seed=42):
    """Estimate epoch rejection and interpolation with optional AutoReject."""
    from autoreject import AutoReject
    cleaned, log = AutoReject(random_state=seed).fit_transform(epochs, return_log=True)
    report = {"before": len(epochs), "kept": len(cleaned), "dropped": len(epochs) - len(cleaned),
              "interpolated_epochs_by_channel": dict(zip(epochs.ch_names,
                                                         np.sum(log.labels == 2, axis=0).astype(int).tolist()))}
    return cleaned, log, report
