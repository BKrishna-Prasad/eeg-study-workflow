"""Welch PSD, Morlet power and configurable channel-region features."""

import numpy as np
import pandas as pd


def select_condition(epochs, condition):
    """Use MNE hierarchical tag matching; require surviving epochs."""
    required = set(condition.split("/"))
    if not condition or not all(required):
        raise ValueError("Conditions require non-empty hierarchical tags")
    keys = [key for key in epochs.event_id if required.issubset(set(key.split("/")))]
    if not keys:
        raise ValueError(f"Missing condition {condition!r}")
    selected = epochs[keys]
    if not len(selected):
        raise ValueError(f"No surviving epochs for {condition!r}")
    return selected


def roi_indices(names, channels):
    if not channels or len(set(channels)) != len(channels) or not set(channels).issubset(names):
        raise ValueError(f"ROI requires every channel exactly once: {channels}")
    return [names.index(ch) for ch in channels]


def to_db(power):
    """Express positive power relative to one unit of its linear scale."""
    power = np.asarray(power)
    if not np.isfinite(power).all() or np.any(power <= 0):
        raise ValueError("Log power requires finite positive values")
    return 10 * np.log10(power)


def welch_psd(epochs, condition, fmin=1.0, fmax=40.0):
    """Use one full-epoch Hamming segment, zero overlap and no zero padding."""
    selected = select_condition(epochs, condition)
    n_samples = len(selected.times)
    spectrum = selected.compute_psd(method="welch", fmin=fmin, fmax=fmax,
                                    n_fft=n_samples, n_per_seg=n_samples, n_overlap=0,
                                    window="hamming", remove_dc=True, picks="eeg", n_jobs=1)
    return spectrum


def roi_curves(epochs, subject_id, conditions, rois, demonstration_only=False, fmin=1.0, fmax=40.0):
    rows = []
    for condition in conditions:
        psd = welch_psd(epochs, condition, fmin, fmax)
        mean = psd.get_data().mean(axis=0)
        for roi, channels in rois.items():
            power = mean[roi_indices(psd.ch_names, channels)].mean(axis=0)
            for freq, linear, db in zip(psd.freqs, power, to_db(power)):
                rows.append({"subject_id": subject_id, "condition": condition, "roi": roi,
                             "frequency_hz": float(freq), "psd_v2_hz": float(linear), "psd_db": float(db),
                             "n_epochs": len(select_condition(epochs, condition)),
                             "channels": ",".join(channels), "demonstration_only": demonstration_only})
    return pd.DataFrame(rows)


def band_features(epochs, subject_id, conditions, rois, bands, demonstration_only=False, fmin=1.0, fmax=40.0):
    """Return mean PSD, ln(mean PSD), mean dB and optional ln(right/left) FAA."""
    rows = []
    for condition in conditions:
        psd = welch_psd(epochs, condition, fmin, fmax)
        mean = psd.get_data().mean(axis=0)
        logs = {}
        for roi, channels in rois.items():
            curve = mean[roi_indices(psd.ch_names, channels)].mean(axis=0)
            to_db(curve)
            for band, (lower, upper) in bands.items():
                if not psd.freqs[0] <= lower < upper <= psd.freqs[-1]:
                    raise ValueError("Band bounds must lie within the computed PSD range")
                mask = (psd.freqs >= lower) & (psd.freqs <= upper)
                if not mask.any():
                    raise ValueError(f"No frequency bins for {band}")
                power = float(curve[mask].mean())
                ln_power = float(np.log(power)); logs[(roi, band)] = ln_power
                for suffix, value, units in (("mean_psd", power, "V^2/Hz"),
                                             ("ln_mean_psd", ln_power, "ln(PSD / (1 V^2/Hz))"),
                                             ("mean_db", float(to_db(curve[mask]).mean()), "dB re 1 V^2/Hz")):
                    rows.append({"subject_id": subject_id, "condition": condition,
                                 "metric": f"{roi}_{band}_{suffix}", "value": value, "units": units,
                                 "channels": ",".join(channels), "band_hz": f"{lower:g}-{upper:g} inclusive",
                                 "n_epochs": len(select_condition(epochs, condition)),
                                 "demonstration_only": demonstration_only})
        if ("Left_Frontal", "Alpha") in logs and ("Right_Frontal", "Alpha") in logs:
            rows.append({"subject_id": subject_id, "condition": condition, "metric": "FAA",
                         "value": logs[("Right_Frontal", "Alpha")] - logs[("Left_Frontal", "Alpha")],
                         "units": "ln(Right_Frontal alpha PSD / Left_Frontal alpha PSD)",
                         "channels": "L:" + ",".join(rois["Left_Frontal"]) + ";R:" + ",".join(rois["Right_Frontal"]),
                         "band_hz": f"{bands['Alpha'][0]:g}-{bands['Alpha'][1]:g} inclusive",
                         "n_epochs": len(select_condition(epochs, condition)), "demonstration_only": demonstration_only})
    return pd.DataFrame(rows)


def morlet_power(epochs, condition, channels, freqs=None, cycles_factor=0.5):
    """Return averaged linear Morlet power with no baseline normalization."""
    selected = select_condition(epochs, condition)
    roi_indices(selected.ch_names, channels)
    freqs = np.arange(1.0, 41.0) if freqs is None else np.asarray(freqs, dtype=float)
    if (freqs.ndim != 1 or len(freqs) == 0 or np.any(np.diff(freqs) <= 0) or
        np.any(freqs <= 0) or np.any(freqs >= selected.info["sfreq"] / 2) or cycles_factor <= 0):
        raise ValueError("Require increasing positive frequencies below Nyquist and positive cycles_factor")
    return selected.compute_tfr(method="morlet", freqs=freqs, n_cycles=freqs * cycles_factor,
                                picks=channels, average=True, return_itc=False,
                                use_fft=True, zero_mean=True, n_jobs=1)
