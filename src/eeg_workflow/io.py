"""Recording readers and run metadata utilities."""

import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import mne
import numpy as np


def write_json(path, value):
    """Write portable JSON; disallow non-finite numeric values."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def numpy_scalar(item):
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Not JSON serializable: {type(item).__name__}")
    text = json.dumps(value, indent=2, allow_nan=False, default=numpy_scalar) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def software_versions():
    result = {}
    for name in ("mne", "numpy", "scipy", "pandas", "matplotlib", "scikit-learn",
                 "pyprep", "autoreject", "mne-icalabel", "pyxdf"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def read_raw(path, xdf=None):
    """Read a supported MNE recording or an explicitly configured XDF stream."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".vhdr":
        return mne.io.read_raw_brainvision(path, preload=True)
    if suffix == ".fif" or path.name.lower().endswith(".fif.gz"):
        return mne.io.read_raw_fif(path, preload=True)
    readers = {".edf": mne.io.read_raw_edf, ".bdf": mne.io.read_raw_bdf,
               ".set": mne.io.read_raw_eeglab}
    if suffix in readers:
        return readers[suffix](path, preload=True)
    if suffix != ".xdf":
        raise ValueError("Supported inputs: .vhdr, .fif, .fif.gz, .edf, .bdf, .set, .xdf")
    if not xdf:
        raise ValueError("XDF requires stream names, channel_map and eeg_unit in config")
    import pyxdf

    streams, _ = pyxdf.load_xdf(path)
    def select(name):
        found = [s for s in streams if s["info"]["name"][0] == name]
        if len(found) != 1:
            raise ValueError(f"Expected exactly one XDF stream named {name!r}")
        return found[0]
    eeg = select(xdf["eeg_stream"])
    markers = select(xdf["marker_stream"])
    timestamps = np.asarray(eeg["time_stamps"], dtype=float)
    sfreq = float(eeg["info"]["nominal_srate"][0])
    intervals = np.diff(timestamps)
    if sfreq <= 0 or len(intervals) == 0 or np.any(intervals <= 0):
        raise ValueError("XDF EEG requires a positive rate and increasing timestamps")
    # Nominal-rate conversion cannot preserve gaps or strongly irregular sampling.
    if np.max(np.abs(intervals * sfreq - 1)) > xdf.get("max_interval_error", 0.2):
        raise ValueError("XDF sampling is irregular or has gaps; review timestamps before conversion")
    labels = [c["label"][0] for c in eeg["info"]["desc"][0]["channels"][0]["channel"]]
    mapping = xdf["channel_map"]
    if not mapping or not set(mapping).issubset(labels):
        raise ValueError("XDF channel_map must explicitly identify recorded EEG channels")
    scale = {"V": 1.0, "uV": 1e-6}.get(xdf["eeg_unit"])
    if scale is None:
        raise ValueError("Verify the XDF EEG units and specify V or uV")
    data = np.asarray(eeg["time_series"], dtype=float)
    if data.shape != (len(timestamps), len(labels)):
        raise ValueError("XDF samples and channel metadata disagree")
    picks = [labels.index(ch) for ch in mapping]
    raw = mne.io.RawArray(data[:, picks].T * scale,
                         mne.create_info(list(mapping.values()), sfreq, "eeg"))
    onsets = np.asarray(markers["time_stamps"]) - timestamps[0]
    if len(onsets) != len(markers["time_series"]):
        raise ValueError("XDF marker timestamps and values disagree")
    if np.any((onsets < 0) | (onsets > raw.times[-1])):
        raise ValueError("XDF contains markers outside the EEG recording; inspect before conversion")
    descriptions = [xdf.get("marker_prefix", "") + str(row[0])
                    for row in markers["time_series"]]
    raw.set_annotations(mne.Annotations(onsets, np.zeros(len(onsets)), descriptions))
    return raw
