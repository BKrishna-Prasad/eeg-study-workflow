# Configuration and recording setup

I keep the recording assumptions in a JSON file so a subject run can be repeated with the same settings. Start from `configs/study.json`; edit the file before preparing a recording. Unknown configuration keys are rejected to catch misspelled options.

## Recording inputs

BrainVision headers need their companion files in the expected locations. FIF can preserve annotations and electrode positions prepared in MNE. EDF/BDF and EEGLAB files are loaded through their MNE readers; check channel types and annotation descriptions after importing.

Inspect the metadata before choosing conditions:

```python
from eeg_workflow.io import read_raw

raw = read_raw("data/sub-001.vhdr")
print(raw.ch_names)
print(raw.get_channel_types())
print(raw.info["sfreq"])
print(raw.annotations)
```

MNE stores EEG in volts. Format readers handle recorded scaling; an array supplied to `mne.io.RawArray` must already use the correct units. Verify scaling when converting your own files.

## Preprocessing settings

| Setting | Use |
| --- | --- |
| `l_freq`, `h_freq` | Positive FIR band-pass limits below Nyquist. Choose them for the question and sampling rate. |
| `notch_hz` | A single line-noise frequency below Nyquist, or `null` to skip. Avoid an unnecessary notch outside the analysis passband. |
| `rename_channels` | Map recorded names to the names used by the montage and ROIs. Renaming precedes type assignment and dropping. |
| `channel_types` | Set auxiliary channels explicitly, for example `{"VEOG": "eog", "ECG": "ecg"}`. |
| `drop_channels` | Explicit channel names to remove. Unknown names fail rather than being silently ignored. |
| `montage` | An MNE built-in montage name, or `null` to preserve positions already present in `Raw`. |
| `crop_s` | `null` or `[start, stop]` in seconds from the loaded recording's start. |
| `bad_channels` | EEG channels identified during inspection. They are excluded from ICA fitting. |
| `bad_channel_method` | `manual` uses explicit bads; `pyprep` also runs PyPREP's full noisy-channel checks. |
| `interpolate_bads` | Whether to interpolate marked EEG channels after ICA removal. If false, they stay marked and are excluded from analysis. |
| `reference` | `"average"`, a list such as `["M1", "M2"]`, or `null` to preserve the existing reference. Named references must be present and usable. |
| `reference_stage` | `before_ica` or `after_ica`. The configured reference is applied after interpolation in either case. |
| `seed` | Random seed for optional bad-channel detection. |

The workflow requires at least four typed EEG channels and usable EEG positions for topographic review. For a custom cap, set positions on an MNE `Raw` object and use `montage: null`; do not assign an unrelated montage just to make plotting succeed.

`channel_qc.csv` reports each EEG channel's standard deviation and peak-to-peak amplitude before filtering. It is a review aid rather than a pass/fail threshold. Inspect the continuous signal and add `BAD_*` annotations for unusable intervals before `prepare_run`. If PyPREP marks channels you disagree with, inspect its report, then create a fresh configuration/run with the manually reviewed list and `bad_channel_method: manual`.

## ICA settings

`n_components` is an integer from 2 to the usable EEG rank, or `null` to use that rank. Marking channels or referencing can reduce rank. `fit_l_freq` controls an additional high-pass on the fitting copy; the fitted decomposition is applied to the saved prepared recording. `seed` makes fitting repeatable in the same software environment. The decomposition uses extended Infomax.

Install the `artifacts` extra before using ICLabel, PyPREP or AutoReject. ICLabel is requested with `prepare --suggest-iclabel`; `autoreject: true` enables epoch rejection/interpolation after segmentation. Their outputs still require inspection. These optional stages are off in the synthetic demo.

## Epoching

I configure epoching separately from signal preprocessing so the same processing functions can be used with different recording layouts. Recorded marker names can be flat or hierarchical; the values in `annotation_map` give them analysis labels. For example, `{"S1": "A", "S2": "B"}` works without any phase hierarchy.

| Recording layout | Epoching mode | Required information |
| --- | --- | --- |
| Point triggers for trials or stimuli | `events` | Trigger descriptions and an event-relative time window |
| Annotations identifying blocks or phases | `annotations` | Annotation descriptions with positive durations |
| Block boundaries in a timing log or defined separately | `intervals` | Labelled start/stop times in the prepared recording |
| Continuous recording without event markers | `intervals` | A single interval, or several justified intervals, covering the portion to analyse |

### Event-locked windows

Use point triggers with exact descriptions from `raw.annotations`:

```json
{
  "mode": "events",
  "annotation_map": {"Stimulus/S 1": "A", "Stimulus/S 2": "B"},
  "tmin": -0.5,
  "tmax": 1.5,
  "baseline": null
}
```

`tmin`/`tmax` are relative to the trigger. MNE includes both endpoint samples. For amplitude baseline subtraction, use a two-item interval such as `[-0.5, 0]`; this is separate from TFR baseline normalization. Events outside usable boundaries or overlapping `BAD` annotations are dropped and logged. Duplicate mapped event onsets fail for review.

### Duration-bearing annotations

Use annotations with a positive duration to identify blocks or phases:

```json
{
  "mode": "annotations",
  "annotation_map": {"Block/A": "A", "Block/B": "B"},
  "length_s": 2.0
}
```

The workflow divides each block into complete non-overlapping windows. Remainders shorter than a window are discarded and logged. Annotation names can map to hierarchical condition labels such as `A/early` and `A/late`.

Point markers have no block duration. When start/end triggers delimit a phase, convert their known boundaries into duration-bearing annotations or explicit intervals before using block mode. Repeated identical markers need protocol-specific interpretation; the workflow does not assume that a particular number or order defines a phase.

### Explicit intervals

For a recording whose block times are available separately:

```json
{
  "mode": "intervals",
  "length_s": 2.0,
  "intervals": [
    {"condition": "A", "start_s": 10.0, "stop_s": 40.0},
    {"condition": "B", "start_s": 50.0, "stop_s": 80.0}
  ]
}
```

Interval times are relative to the **prepared recording**, after any crop. Blocks must lie within that recording and must not overlap. Block stop times are exclusive; each two-second window has exactly `2 × sampling_rate` samples. The same convention is used for duration-bearing annotations.

### Recordings without markers

Event markers are not required for interval mode. For example, to analyse the first 120 seconds of a continuous recording that contains at least that duration:

```json
{
  "mode": "intervals",
  "length_s": 2.0,
  "intervals": [
    {"condition": "recording", "start_s": 0.0, "stop_s": 120.0}
  ]
}
```

Set `spectral_conditions` and `feature_conditions` to `["recording"]`. Use `morlet_conditions: []` to skip time-frequency analysis, or `["recording"]` to compute it. Adjust the channel, ROI, band and reference settings for the actual recording. The label `recording` identifies the selected data without assigning an experimental condition.

This configuration produces subject-level spectra and features from complete windows. A paired group comparison requires two meaningful conditions measured in the same subjects. With one condition, use the subject outputs and `grand_average` function for descriptive group PSD summaries; do not invent a second condition for a statistical test.

## Analysis settings

`spectral_conditions`, `feature_conditions` and `morlet_conditions` select condition labels. Flat names such as `A`, `B` or `recording` work directly. With optional slash-separated labels, MNE hierarchical matching means `A` includes `A/early` and `A/late`; select a full label for a phase-specific comparison. Conditions with no surviving epochs fail explicitly.

`spectral_rois` and `feature_rois` map ROI names to channel lists. Every selected channel must exist among the usable EEG channels; the workflow does not silently reduce an ROI. Names may be arbitrary. `Left_Frontal`, `Right_Frontal` and the `Alpha` band enable the optional FAA feature and hemispheric comparisons.

`fmin`/`fmax` set the Welch output range. `bands` supplies inclusive frequency bounds within that range. The PSD grid depends on epoch duration, so the chosen bounds must fit its available bins. Adjacent inclusive bands share a boundary bin.

`morlet_channels`, `morlet_freqs` and `morlet_cycles_factor` control Morlet analysis: `n_cycles = frequency × factor`. Choose a factor and epoch length that support the lowest frequency. An empty `morlet_conditions` list skips this stage. Saved arrays contain averaged linear power; the figure uses dB and no baseline correction.

## Custom inputs through Python

The saved workflow accepts an MNE `Raw` object, so additional readers, custom electrode positions or externally recorded events can be prepared in Python:

```python
import mne
from eeg_workflow.io import read_json
from eeg_workflow.workflow import prepare_run

raw = mne.io.read_raw_fif("data/sub-001-raw.fif", preload=True)
# Set channel types, custom positions and annotations here if needed.
config = read_json("configs/study.json")
prepare_run(raw, config, "derivatives/sub-001", "sub-001")
```

For stim-channel triggers, use `mne.find_events` and `mne.annotations_from_events` to attach descriptions before preparation. Inspect the resulting onsets and labels; the workflow cannot infer experimental meaning from numeric triggers.

## XDF input

Install the `xdf` extra and add an `xdf` section:

```json
{
  "eeg_stream": "EEG",
  "marker_stream": "Markers",
  "channel_map": {"recorded_1": "F3", "recorded_2": "F4"},
  "eeg_unit": "uV",
  "marker_prefix": "",
  "max_interval_error": 0.2
}
```

Map every EEG channel you intend to use; the two entries above only illustrate the syntax. Stream names must identify exactly one EEG and marker stream. The importer checks increasing timestamps and sampling intervals before converting to a regular `RawArray`; recordings with gaps or excessive irregularity fail for separate handling. The interval threshold is a conversion check, not a timing-accuracy guarantee. XDF point markers can be used with event mode or independently defined intervals.

## Group outputs

Group inputs must use unique subject/condition/metric rows, consistent units, ROI channels and band definitions. PSD curves must have matching frequency grids. Synthetic and real-data tables cannot be mixed.

`--metrics` selects the features for paired tests and trajectory plots. Omitting it selects all natural-log mean PSD features and FAA. `--hemispheres` additionally compares the named left/right frontal ROIs within each selected condition. Condition differences are reported as A minus B; hemispheric differences are right minus left. P-values are unadjusted, so define comparisons and any multiplicity correction in your study's analysis plan.
