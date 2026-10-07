"""Verify signal transformations, experimental windows and review gates."""

import json
from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import mne
import numpy as np

from eeg_workflow.cli import run_demo
from eeg_workflow.demo import demo_config, synthetic_raw
from eeg_workflow.epochs import make_epochs
from eeg_workflow.ica import apply_review, save_review
from eeg_workflow.io import read_json, read_raw
from eeg_workflow.preprocessing import prepare_raw
from eeg_workflow.spectral import morlet_power, roi_indices, select_condition, welch_psd
from eeg_workflow.workflow import analyse_run

mne.set_log_level("ERROR")


class SignalTests(unittest.TestCase):
    def test_synthetic_determinism_dimensions_events(self):
        a, b = synthetic_raw(97), synthetic_raw(97)
        np.testing.assert_array_equal(a.get_data(), b.get_data())
        self.assertEqual(a.get_data().shape, (17, 96 * 128))
        self.assertEqual(len(a.annotations), 8)
        self.assertFalse(np.array_equal(a.get_data(), synthetic_raw(98).get_data()))

    def test_filter_shape_and_known_frequency_attenuation(self):
        raw = synthetic_raw()
        times = raw.times
        raw._data[:16] = 1e-5 * (np.sin(2*np.pi*10*times) + np.sin(2*np.pi*50*times))
        config = demo_config()
        config["preprocessing"]["reference"] = None
        prepared, qc, report = prepare_raw(raw, config["preprocessing"])
        self.assertEqual(prepared.get_data().shape, raw.get_data().shape)
        middle = (times > 10) & (times < 85)
        values = prepared.get_data(picks=["F3"])[0, middle]
        ten = abs(np.dot(values, np.sin(2*np.pi*10*times[middle])))
        fifty = abs(np.dot(values, np.sin(2*np.pi*50*times[middle])))
        self.assertLess(fifty / ten, 0.01)
        self.assertEqual(report["final_bads"], ["P3"])
        self.assertEqual(len(qc), 16)

    def test_crop_coordinates_boundaries_and_bad_annotation(self):
        raw = synthetic_raw().crop(2, 70)
        config = demo_config()
        epochs, log = make_epochs(raw, config["epoching"])
        self.assertEqual(len(epochs), 24)
        self.assertEqual(len(epochs.times), 256)
        self.assertEqual(epochs.events[0, 0], 4 * 128)
        self.assertTrue((epochs.metadata.stop_sample_exclusive - epochs.metadata.start_sample == 256).all())
        raw.annotations.append(5.0, 1.0, "BAD_demo")
        rejected, _ = make_epochs(raw, config["epoching"])
        self.assertLess(len(rejected), len(epochs))
        self.assertEqual(log["generated"], 24)

    def test_block_overlap_and_point_markers_fail(self):
        raw = synthetic_raw()
        raw.set_annotations(mne.Annotations([4, 20], [24, 24], ["Block/A", "Block/B"]))
        with self.assertRaisesRegex(ValueError, "overlap"):
            make_epochs(raw, demo_config()["epoching"])
        with self.assertRaisesRegex(ValueError, "no block duration"):
            make_epochs(synthetic_raw(), {"mode": "annotations", "length_s": 2,
                                         "annotation_map": {"Trial/A": "A"}})

    def test_event_locked_baseline_and_condition_mapping(self):
        settings = {"mode": "events", "annotation_map": {"Trial/A": "A", "Trial/B": "B"},
                    "tmin": -0.5, "tmax": 1.5, "baseline": [-0.5, 0]}
        epochs, report = make_epochs(synthetic_raw().crop(2, 70), settings)
        self.assertEqual(len(epochs), 6)
        self.assertEqual(len(epochs["A"]), 3)
        self.assertEqual(epochs.events[0, 0], 4 * 128)
        self.assertEqual(len(epochs.times), 257)
        baseline = epochs.get_data()[:, :, epochs.times <= 0]
        np.testing.assert_allclose(baseline.mean(axis=2), 0, atol=1e-18)
        self.assertEqual(report["generated"], 6)

    def test_explicit_intervals_match_annotations_and_log_tails(self):
        settings = {"mode": "intervals", "length_s": 2, "intervals": [
                    {"condition": "A", "start_s": 4, "stop_s": 28},
                    {"condition": "B", "start_s": 40, "stop_s": 64.5}]}
        epochs, report = make_epochs(synthetic_raw(), settings)
        reference, _ = make_epochs(synthetic_raw(), demo_config()["epoching"])
        np.testing.assert_array_equal(epochs.events, reference.events)
        self.assertEqual(report["partial_windows"][1]["discarded_tail_s"], 0.5)
        settings["intervals"][0]["stop_s"] = 100
        with self.assertRaisesRegex(ValueError, "outside"):
            make_epochs(synthetic_raw(), settings)

    def test_welch_and_morlet_known_rhythm_and_dimensions(self):
        raw = synthetic_raw(); raw._data[:16] = 1e-5 * np.sin(2*np.pi*10*raw.times)
        epochs, _ = make_epochs(raw, demo_config()["epoching"])
        psd = welch_psd(epochs, "A")
        self.assertEqual(psd.get_data().shape, (12, 16, 79))
        self.assertEqual(psd.freqs[np.argmax(psd.get_data().mean(axis=(0, 1)))], 10)
        tfr = morlet_power(epochs, "B", ["F3"], [6, 10, 20])
        self.assertEqual(tfr.data.shape, (1, 3, 256))
        self.assertEqual(tfr.freqs[tfr.data[:, :, 64:192].mean(axis=(0, 2)).argmax()], 10)
        with self.assertRaises(ValueError):
            roi_indices(psd.ch_names, ["F3", "missing"])
        with self.assertRaises(ValueError):
            select_condition(epochs, "Missing")

    def test_xdf_explicit_units_channel_mapping_and_timestamp_checks(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        eeg = {"info": {"name": ["EEG"], "nominal_srate": ["128"],
                        "desc": [{"channels": [{"channel": [{"label": ["a"]}, {"label": ["b"]}]}]}]},
               "time_series": np.ones((256, 2)), "time_stamps": 100 + np.arange(256) / 128}
        marker = {"info": {"name": ["Markers"]}, "time_series": [["S1"]], "time_stamps": np.array([100.5])}
        settings = {"eeg_stream": "EEG", "marker_stream": "Markers", "channel_map": {"a": "F3", "b": "F4"}, "eeg_unit": "uV"}
        module = SimpleNamespace(load_xdf=lambda _: ([eeg, marker], {}))
        with patch.dict("sys.modules", {"pyxdf": module}):
            raw = read_raw("fixture.xdf", settings)
            np.testing.assert_allclose(raw.get_data(), 1e-6)
            self.assertEqual(raw.ch_names, ["F3", "F4"])
            self.assertAlmostEqual(raw.annotations.onset[0], 0.5)
            eeg["time_stamps"][100:] += 0.5
            with self.assertRaisesRegex(ValueError, "irregular"):
                read_raw("fixture.xdf", settings)


class SavedWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.run_dir = Path(cls.temporary.name) / "demo"
        run_demo(cls.run_dir)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_end_to_end_outputs_reference_and_subject_features(self):
        run = self.run_dir
        required = ["prepared-raw.fif", "model-ica.fif", "ica_review.json", "cleaned-raw.fif",
                    "clean-epo.fif", "features.csv", "psd_curves.csv", "subject_summary.json",
                    "tfr_A.npz", "tfr_B.npz", "inspection/components_1.png"]
        self.assertTrue(all((run / name).is_file() for name in required))
        summary = read_json(run / "subject_summary.json")
        self.assertTrue(summary["demonstration_only"])
        self.assertEqual(summary["n_epochs"], 24)
        raw = mne.io.read_raw_fif(run / "cleaned-raw.fif", preload=True)
        np.testing.assert_allclose(raw.get_data(picks="eeg").mean(axis=0), 0, atol=1e-18)
        self.assertEqual(raw.info["bads"], [])
        model = mne.preprocessing.read_ica(run / "model-ica.fif")
        self.assertNotIn("P3", model.ch_names)
        import pandas as pd
        table = pd.read_csv(run / "features.csv")
        for condition, subset in table.groupby("condition"):
            values = subset.set_index("metric").value
            self.assertAlmostEqual(values["FAA"], values["Right_Frontal_Alpha_ln_mean_psd"] - values["Left_Frontal_Alpha_ln_mean_psd"])
            self.assertAlmostEqual(values["Frontal_Theta_ln_mean_psd"], np.log(values["Frontal_Theta_mean_psd"]))

    def test_review_persistence_binding_and_invalid_indices(self):
        run = self.run_dir
        _, _, review = apply_review(run)
        self.assertEqual(review["decision_kind"], "scripted_synthetic_example")
        original = (run / "config.json").read_bytes()
        try:
            (run / "config.json").write_bytes(original + b" ")
            with self.assertRaisesRegex(ValueError, "bound to different"):
                apply_review(run)
            with self.assertRaisesRegex(ValueError, "changed after processing"):
                analyse_run(run)
        finally:
            (run / "config.json").write_bytes(original)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "manifest.json").write_text(json.dumps({"n_components": 6, "demonstration_only": False}))
            for exclusions in ([6], [-1], [True], [1, 1], list(range(6))):
                with self.assertRaises(ValueError):
                    save_review(folder, exclusions, "reviewer", "reason")
            with self.assertRaises(ValueError):
                save_review(folder, [], "", "reason")
        with self.assertRaises(FileExistsError):
            save_review(run, [], "reviewer", "changed decision")

    def test_finish_requires_a_review(self):
        review_path = self.run_dir / "ica_review.json"
        contents = review_path.read_bytes()
        try:
            review_path.unlink()
            with self.assertRaises(FileNotFoundError):
                apply_review(self.run_dir)
        finally:
            review_path.write_bytes(contents)


if __name__ == "__main__":
    unittest.main()
