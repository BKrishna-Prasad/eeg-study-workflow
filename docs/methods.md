# Methods and interpretation

I keep preprocessing choices and feature definitions explicit so the exported tables can be interpreted alongside the recording and analysis settings.

## Channels, filtering and reference

EEG values use volts internally. QC reports channel dispersion in microvolts before filtering. FIR band-pass and optional notch filters use MNE's `firwin` implementation. Marked bad channels are excluded from the fitted decomposition and, if requested, interpolated after ICA removal. The reference can be applied before or after fitting; it is reapplied after interpolation to include restored channels.

Reference choice, electrode geometry and interpolation affect downstream features. Inspect the channel report and the continuous recording; a configuration file cannot establish recording quality.

## ICA review

The decomposition uses extended Infomax on a high-passed copy of the prepared recording. I save component topographies, source previews and source spectra before applying exclusions. The final decision includes zero-based component indices, reviewer, rationale, timestamp and hashes binding it to the fitted recording, model and settings.

Use the saved plots to begin inspection, then review the full source activity and component properties when needed:

```python
import mne

raw = mne.io.read_raw_fif("derivatives/sub-001/prepared-raw.fif", preload=True)
ica = mne.preprocessing.read_ica("derivatives/sub-001/model-ica.fif")
ica.plot_components()
ica.plot_sources(raw)
ica.plot_properties(raw, picks=[0, 1])
```

ICLabel suggestions are optional and never substitute for the recorded decision. Its documented input assumptions include common average reference, extended Infomax and a 1–100 Hz passband. A lower sampling rate or narrower passband may prevent matching those conditions, so interpret its scores cautiously and keep its warnings visible. [ICLabel documentation](https://mne.tools/mne-icalabel/stable/generated/api/mne_icalabel.iclabel.iclabel_label_components.html).

PyPREP can mark noisy channels and AutoReject can repair/reject epochs. Review their reports, especially when many channels or epochs are affected. Successful execution does not establish that an artifact decision is appropriate.

## Segmentation

Event mode follows MNE's event-relative time limits and optional signal baseline. Block modes produce complete non-overlapping windows within labelled intervals, with no signal baseline subtraction. `BAD` annotations reject intersecting epochs. The processing log records generated/kept windows, discarded tails and MNE's drop reasons.

Pooling phase labels weights the subject estimate by the number of surviving windows in each phase. Compare equivalent intervals where that is required by the scientific question. Multiple epochs improve a subject estimate but do not create additional independent participants.

## Welch PSD

Each epoch uses a Hamming window spanning the full epoch, with zero segment overlap, DC removal and no zero padding. This is one Welch segment per epoch; condition estimates then average across epochs and ROI channels in linear units. Bin spacing is `sampling_rate / epoch_samples`. A two-second block window yields 0.5 Hz spacing. [MNE Welch documentation](https://mne.tools/stable/generated/mne.time_frequency.psd_array_welch.html).

The exported subject curve contains linear PSD in V²/Hz and `10 log10(PSD / (1 V²/Hz))`. The group curve averages subject dB curves with equal subject weighting and reports the between-subject SEM. It is a different operation from averaging linear subject power and then taking the logarithm.

## Band features

For each subject, condition and ROI, the workflow exports:

| Suffix | Definition | Units |
| --- | --- | --- |
| `mean_psd` | Mean linear PSD across the configured inclusive band bins | V²/Hz |
| `ln_mean_psd` | Natural log of that mean, relative to 1 V²/Hz | ln(PSD / (1 V²/Hz)) |
| `mean_db` | Mean of the bin-wise dB values within the band | dB re 1 V²/Hz |

These are summaries of spectral density, rather than integrated band power. Adjacent bands with inclusive bounds share their boundary bin. ROI channels, band limits, units and surviving epoch counts are included in the table.

With `Left_Frontal`, `Right_Frontal` and `Alpha` configured, FAA is `ln(right alpha mean PSD) − ln(left alpha mean PSD)`, a dimensionless log ratio. Its sign alone does not establish hemisphere activation or a validated biomarker. Choose electrode groups and frequency bounds consistently across subjects.

## Morlet time-frequency power

Morlet analysis uses zero-mean wavelets and FFT convolution, with `n_cycles = frequency × morlet_cycles_factor`. Linear power is averaged across the selected epochs and saved with channel, frequency and time coordinates. Plots average the selected channels and convert power to dB relative to 1 V². No TFR baseline normalization or ITC is computed.

Wavelets extend around each time point. Inspect edge effects and choose longer epochs when low frequencies need more support. Event epochs retain trigger-relative time; block windows retain time within each window and do not establish an event-onset response.

## Paired group comparisons

Subject IDs determine the pairs. The group command reports mean A-minus-B differences, paired t-tests and two-sided Wilcoxon signed-rank tests for selected metrics. Incomplete pairs and degenerate comparisons are identified explicitly. Optional hemispheric comparisons report right minus left within a condition.

P-values are unadjusted. I treat these outputs as exploratory unless the study has an appropriate analysis plan. Check the assumptions of the chosen test, the number of independent subjects, missingness and multiplicity before drawing conclusions. The synthetic demonstration has no scientific condition effect to interpret.
