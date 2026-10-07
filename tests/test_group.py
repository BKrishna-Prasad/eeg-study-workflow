"""Verify participant pairing, definitions and group curve consistency."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from eeg_workflow.group_analysis import grand_average, load_features, paired_comparison


def features():
    rows = []
    for subject, a, b in (("sub-01", 1, 1.5), ("sub-02", 3, 2), ("sub-03", 4, 5)):
        for condition, value in (("A", a), ("B", b)):
            rows.append({"subject_id": subject, "condition": condition, "metric": "FAA", "value": value,
                         "units": "ln(R/L)", "channels": "L:F3;R:F4", "band_hz": "8-13 inclusive",
                         "demonstration_only": True})
    return pd.DataFrame(rows)


class GroupTests(unittest.TestCase):
    def test_numeric_subject_ids_keep_leading_zeroes(self):
        data = features()
        data["subject_id"] = data.subject_id.map({"sub-01": "001", "sub-02": "002", "sub-03": "010"})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "features.csv"
            data.to_csv(path, index=False)
            output = load_features([path])
        self.assertEqual(set(output.subject_id), {"001", "002", "010"})

    def test_pairs_are_matched_by_subject_and_missing_ids_reported(self):
        data = features().sample(frac=1, random_state=9)
        output = paired_comparison(data, "A", "B").iloc[0]
        self.assertEqual(output.n_pairs, 3)
        self.assertAlmostEqual(output.mean_difference, -1/6)
        data = data[~((data.subject_id == "sub-02") & (data.condition == "B"))]
        output = paired_comparison(data, "A", "B").iloc[0]
        self.assertEqual(output.n_pairs, 2); self.assertEqual(output.excluded_subject_ids, "sub-02")

    def test_duplicates_units_and_mixed_data_are_rejected(self):
        for alteration in ("duplicate", "units", "demo", "nonfinite"):
            data = features()
            if alteration == "duplicate": data = pd.concat([data, data.iloc[[0]]])
            elif alteration == "units": data.loc[0, "units"] = "different"
            elif alteration == "demo": data.loc[0, "demonstration_only"] = False
            else: data.loc[0, "value"] = np.inf
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "features.csv"; data.to_csv(path, index=False)
                with self.assertRaises(ValueError): load_features([path])

    def test_degenerate_pairs_are_flagged(self):
        data = features()
        for subject in data.subject_id.unique():
            data.loc[(data.subject_id == subject) & (data.condition == "B"), "value"] = data.loc[(data.subject_id == subject) & (data.condition == "A"), "value"].iloc[0]
        output = paired_comparison(data, "A", "B").iloc[0]
        self.assertEqual(output.status, "all_differences_zero")
        self.assertTrue(np.isnan(output.p_t_unadjusted))

    def test_grand_average_weights_subjects_and_rejects_frequency_mismatch(self):
        data = pd.DataFrame([{"subject_id": subject, "condition": "A", "roi": "Frontal", "frequency_hz": freq,
                              "psd_db": value, "channels": "F3,F4", "demonstration_only": True}
                             for subject, value in (("s1", 0.0), ("s2", 2.0)) for freq in (8.0, 9.0)])
        output = grand_average(data)
        np.testing.assert_allclose(output.mean_subject_db, 1)
        np.testing.assert_allclose(output.sem_subject_db, 1)
        data.loc[0, "frequency_hz"] = 8.5
        with self.assertRaisesRegex(ValueError, "frequency grids"):
            grand_average(data)


if __name__ == "__main__":
    unittest.main()
