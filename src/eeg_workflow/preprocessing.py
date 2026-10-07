"""Filtering, channel checks and post-ICA interpolation/reference."""

import mne
import numpy as np
import pandas as pd


def channel_qc(raw):
    """Return channel dispersion for inspection, without automatic decisions."""
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    values = raw.get_data(picks=picks)
    names = [raw.ch_names[i] for i in picks]
    if not np.isfinite(values).all():
        raise ValueError("Non-finite EEG samples: inspect the recording before filtering")
    return pd.DataFrame({"channel": names, "std_uV": values.std(axis=1) * 1e6,
                         "peak_to_peak_uV": np.ptp(values, axis=1) * 1e6})


def apply_reference(raw, reference):
    """Apply average or named-channel reference, or preserve the input reference."""
    if reference is None:
        return raw
    eeg_names = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, exclude=[])]
    if reference != "average" and (not isinstance(reference, list) or not reference or
                                   len(set(reference)) != len(reference) or
                                   not set(reference).issubset(eeg_names)):
        raise ValueError("reference must be average, a non-empty list of channel names, or null")
    if isinstance(reference, list) and set(reference).intersection(raw.info["bads"]):
        raise ValueError("Reference channels are marked bad; inspect them before referencing")
    raw.set_eeg_reference(reference, projection=False)
    return raw


def prepare_raw(raw, settings):
    """Configure channels, inspect dispersion, filter and mark bad channels."""
    raw = raw.copy().load_data()
    if settings.get("rename_channels"):
        raw.rename_channels(settings["rename_channels"])
    if settings.get("channel_types"):
        raw.set_channel_types(settings["channel_types"])
    drop = settings.get("drop_channels", [])
    if not set(drop).issubset(raw.ch_names):
        raise ValueError("drop_channels contains unknown channel names")
    raw.drop_channels(drop)
    eeg_names = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, exclude=[])]
    if len(eeg_names) < 4:
        raise ValueError("At least four typed EEG channels are required")
    if settings.get("crop_s") is not None:
        start, stop = settings["crop_s"]
        raw.crop(start, stop)
    if settings.get("montage") is not None:
        raw.set_montage(settings["montage"], match_case=False)
    positions = np.array([raw.info["chs"][i]["loc"][:3] for i in mne.pick_types(raw.info, eeg=True, exclude=[])])
    if not np.isfinite(positions).all() or np.any(np.linalg.norm(positions, axis=1) == 0):
        raise ValueError("EEG positions are required for ICA topographies; set a montage or provide a Raw object with positions")
    qc = channel_qc(raw)
    low, high = settings["l_freq"], settings["h_freq"]
    if not 0 < low < high < raw.info["sfreq"] / 2:
        raise ValueError("Require 0 < l_freq < h_freq < Nyquist")
    raw.filter(low, high, picks="eeg", fir_design="firwin")
    notch = settings.get("notch_hz")
    if notch is not None:
        if not 0 < notch < raw.info["sfreq"] / 2:
            raise ValueError("Notch must be below Nyquist")
        raw.notch_filter([notch], picks="eeg", fir_design="firwin")
    bads = settings.get("bad_channels", [])
    if not set(bads).issubset(eeg_names):
        raise ValueError("Every bad channel must name an EEG channel")
    raw.info["bads"] = sorted(set(raw.info["bads"] + bads))
    mode = settings.get("bad_channel_method", "manual")
    report = {"method": mode, "manual_bads": list(bads), "detector": {}}
    if mode == "pyprep":
        from pyprep.find_noisy_channels import NoisyChannels
        detector = NoisyChannels(raw, do_detrend=True, random_state=settings.get("seed", 97))
        detector.find_all_bads(ransac=True)
        report["detector"] = {k: [str(c) for c in v] for k, v in detector.get_bads(as_dict=True).items()}
        raw.info["bads"] = sorted(set(raw.info["bads"] + [str(c) for c in detector.get_bads()]))
    elif mode != "manual":
        raise ValueError("bad_channel_method must be manual or pyprep")
    report["final_bads"] = list(raw.info["bads"])
    if settings.get("reference_stage", "after_ica") == "before_ica":
        apply_reference(raw, settings.get("reference", "average"))
    return raw, qc, report


def interpolate_and_reference(raw, settings):
    """Optionally interpolate bad channels and apply the configured reference."""
    cleaned = raw.copy()
    if cleaned.info["bads"] and settings.get("interpolate_bads", True):
        cleaned.interpolate_bads(reset_bads=True)
    # Reapply after interpolation so the final reference includes restored channels.
    apply_reference(cleaned, settings.get("reference", "average"))
    return cleaned
