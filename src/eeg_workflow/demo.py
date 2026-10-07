"""Deterministic synthetic EEG for a lightweight workflow demonstration."""

from importlib.resources import files
import json

import mne
import numpy as np


def demo_config():
    return json.loads(files("eeg_workflow").joinpath("defaults.json").read_text(encoding="utf-8"))


def synthetic_raw(seed=97):
    """Generate rhythms, drift, blink-like pulses, block annotations and triggers."""
    rng = np.random.default_rng(seed)
    sfreq = 128.0
    times = np.arange(int(96 * sfreq)) / sfreq
    names = ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz", "FC1", "FC2",
             "C3", "C4", "Cz", "P3", "P4", "Pz", "Oz"]
    blink = sum(np.exp(-0.5 * ((times - centre) / 0.12) ** 2)
                for centre in np.arange(6, 94, 4.5))
    oscillations = np.array([np.sin(2 * np.pi * f * times + phase)
                             for f, phase in ((6, 0.2), (10, 0.7), (22, 1.2))])
    mixing = rng.normal(size=(len(names), 3))
    data = 5e-6 * mixing @ oscillations + rng.normal(0, 1e-6, (len(names), len(times)))
    front_weights = np.array([1.0, 0.9, 0.45, 0.4, 0.3, 0.3, 0.45, 0.25, 0.25,
                              0.1, 0.1, 0.1, 0.02, 0.02, 0.02, 0.01])
    data += front_weights[:, None] * blink * 100e-6
    data += rng.uniform(1e-6, 4e-6, (len(names), 1)) * np.sin(2 * np.pi * 0.25 * times)
    data += rng.uniform(0.2e-6, 1e-6, (len(names), 1)) * np.sin(2 * np.pi * 50 * times)
    # A noisy channel exercises the optional marked-channel/interpolation path.
    data[names.index("P3")] += rng.normal(0, 5e-6, len(times))
    raw = mne.io.RawArray(np.vstack([data, blink * 100e-6]),
                         mne.create_info(names + ["EOG_demo"], sfreq, ["eeg"] * len(names) + ["eog"]))
    raw.set_montage("standard_1020")
    raw.info["description"] = "DEMONSTRATION ONLY: generated rhythms/artifacts/events; no participant recording"
    raw.set_annotations(mne.Annotations([4, 40, 4, 12, 20, 40, 48, 56], [24, 24] + [0] * 6,
                                        ["Block/A", "Block/B"] + ["Trial/A"] * 3 + ["Trial/B"] * 3))
    return raw


def synthetic_blink_suggestions(fit, model):
    """Compare components with a known generated trace; this is not ICLabel."""
    import pandas as pd
    reference = fit.get_data(picks=["EOG_demo"])[0]
    sources = model.get_sources(fit).get_data()
    correlations = np.array([abs(np.corrcoef(source, reference)[0, 1]) for source in sources])
    best = int(correlations.argmax())
    return pd.DataFrame({"component": range(model.n_components_), "label": "known synthetic blink trace",
                         "score": correlations,
                         "suggested": [i == best and correlations[i] >= 0.5 for i in range(model.n_components_)],
                         "method": "DEMONSTRATION ONLY: absolute correlation with generated EOG_demo"})
