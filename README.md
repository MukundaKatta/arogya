# Arogya — AI Diagnostic Bias Detector Across Demographics

Arogya is a Python toolkit for auditing machine-learning diagnostic models for
demographic bias. It evaluates per-group performance across 12 demographic axes
and 6 clinical domains, computes equity metrics, and generates actionable
mitigation recommendations.

## Features

- **BiasAuditor** — evaluate any prediction function against a demographic-annotated
  dataset and receive a full equity breakdown.
- **12 demographic axes** — race, sex, age group, BMI category, skin tone
  (Fitzpatrick), insurance status, language, income bracket, disability status,
  geography, education level, and pregnancy status.
- **6 clinical domains** — dermatology, radiology, cardiology, ophthalmology,
  pathology, and dental.
- **Equity metrics** — sensitivity/specificity per group, calibration error, AUC
  gap, worst-to-best-group ratio, and a composite Diagnostic Equity Score.
- **Mitigation engine** — automatic recommendations for resampling, re-weighting,
  and data-augmentation strategies.
- **Rich reports** — terminal (Rich), JSON, and HTML export.

## Installation

```bash
pip install -e .
```

## Quick start

```bash
# Run an audit from the CLI
arogya audit --data data.csv --model model.pkl

# Generate a report
arogya report --input audit_results.json --format html --output report.html
```

See `examples/run_audit.py` for a programmatic example.

## License

MIT
