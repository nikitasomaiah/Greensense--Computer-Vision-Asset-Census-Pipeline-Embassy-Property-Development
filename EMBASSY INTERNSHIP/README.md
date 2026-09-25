# GreenSense — Tree & Asset Census Module

A computer-vision pipeline that counts landscape assets (trees, palms, shrubs,
hedges, flower beds) from ordinary site photos, logs them over time per
zone using an SQLite register, automatically flags tree loss or damage,
and exports SME audit snapshots.

---

## Why this is worth building for Embassy

Looking at Embassy's own Horticulture & Landscaping SOP, three things line
up almost exactly with what this tool automates:

| SOP requirement | What this tool does |
|---|---|
| "Checking the plants growth (Trees, palms, shrubs, ground covers and lawn)" — Supervisor Roles & Responsibilities | Runs that check automatically from a photo instead of a manual walk-through |
| Daily Landscape Checklist (Annexure 1), filled by hand every day | Becomes a timestamped digital log per zone, per asset type in SQLite |
| "Note: Daily briefing are conducted on site, photos uploaded in the Embassy Boulevard WhatsApp Group" | Input source for the pipeline — supervisors already take and share these daily photos |
| Half-yearly SME Landscaping Audit | The "SME Audit Summary" tab exports CSV/Excel asset snapshots for audit documentation |

The single highest-value feature is the **loss/damage alert**: if a zone's
tree count drops meaningfully versus its recent rolling average (e.g. >30% drop),
that's flagged immediately. This catches storm damage, disease, or theft early for
replacement-cost liability and client SLAs in facilities-management contracts.

---

## How it works

1. **`detect.py`** — Uses OWL-ViT (`google/owlvit-base-patch32`), an open-vocabulary
   object detector, to detect and count assets by plain-text query ("a tree", "a palm
   tree", "a flowering shrub"...). Features include:
   - Input validation & EXIF orientation auto-rotation (`PIL.ImageOps.exif_transpose`).
   - Audit sensitivity tuning (configurable confidence thresholds).
   - `--batch` folder processing CLI mode.
   - Bounding box cropping hook for Nikita's MobileNetV2 plant health classifier stub.

2. **`utils/geotag.py`** — Extracts EXIF GPS metadata from photos to automatically
   suggest known Embassy facility zones (e.g., Manyata Block D Garden, Embassy GolfLinks,
   Boulevard West Lawn) with graceful fallback to manual entry.

3. **`asset_register.py`** — SQLite database backend (`asset_register.db`) preserving
   signature compatibility (`log_census`, `load_register`, `zone_history`, `check_for_loss`, `summary_report`).
   Includes strict data validation, legacy CSV auto-migration, and CSV/Excel audit export (`export_audit_report`).

4. **`app.py`** — Streamlit dashboard featuring:
   - Single photo & Batch walk-through upload modes.
   - EXIF auto-geotag zone suggestions & known zone dropdowns.
   - Plant health classifier stub toggle.
   - Zone count trend line charts.
   - Audit CSV & Excel download buttons.
   - Global exception handling preventing raw tracebacks.

---

## Quick Start & Setup

### Prerequisites
- Python 3.9+ with PyTorch and Transformers installed.

### Automated Setup & Running
```bash
# 1. Install dependencies
make setup

# 2. Generate synthetic sample photos with EXIF tags & pre-populate historical database
make sample_data

# 3. Run unit tests
make test

# 4. Run CLI batch detection on sample data
make batch

# 5. Run accuracy evaluation benchmark against 15 ground-truth scenes
make evaluate

# 6. Launch the Streamlit dashboard
make run
```

---

## Model Accuracy Evaluation Benchmark

To answer **"How accurate is the model?"** with quantitative evidence, we built an empirical evaluation engine ([evaluate.py](file:///Users/nikita/Desktop/EMBASSY%20INTERNSHIP/evaluate.py)) benchmarking OWL-ViT against 15 ground-truth annotated scenes:

- **Benchmark Dataset**: 15 ground-truth annotated fixture scenes (`eval_dataset/`).
- **Mean Absolute Error (MAE)**: `6.15 assets / photo`.
- **Count Bias Direction**: `-5.28` (Systematic negative bias due to canopy occlusion).
- **Evaluation Report**: See full report in [evaluation_report.md](file:///Users/nikita/Desktop/EMBASSY%20INTERNSHIP/evaluation_report.md) or open Tab 4 in the dashboard.

---

## Confidence Threshold Sensitivity Tuning

In a landscape auditing context, choosing the confidence threshold involves a direct trade-off:

- **Low Threshold (0.05 – 0.10)**: High Recall / Sensitivity. Catches small, distant, or heavily occluded plants. *Trade-off*: Higher false positive rate (misidentifying background foliage or shadows).
- **Standard Threshold (0.12 – 0.15, Default)**: Balanced setting optimized for outdoor site survey photos.
- **High Threshold (0.20+)**: High Precision. Ensures high confidence in every detection. *Trade-off*: Higher false negative rate (may miss partially hidden shrubs or trees).

---

## Verification & Testing

To re-verify the full pipeline end-to-end:

1. **Unit Tests**:
   ```bash
   pytest -v tests/
   ```
   *Verifies SQLite logging, negative count rejection, empty zone handling, malformed date validation, loss-alert threshold boundary conditions, evaluation fixtures, and audit export generation.*

2. **Synthetic Data & EXIF Verification**:
   ```bash
   python3 generate_sample_register.py
   ```
   *Creates synthetic garden images with embedded GPS tags in `sample_data/` and populates 6 weeks of census history including a simulated tree loss in 'Manyata Block D Garden'.*

3. **Batch Detection CLI**:
   ```bash
   python3 detect.py --batch sample_data/ --zone "Manyata Block D Garden"
   ```

4. **Model Benchmark Evaluation**:
   ```bash
   python3 evaluate.py
   ```

5. **Dashboard Verification**:
   ```bash
   streamlit run app.py
   ```
   *Navigate to http://localhost:8501 to inspect:*
   - **Tab 1**: EXIF auto-geotag zone detection & single/batch processing.
   - **Tab 2**: Trend chart showing simulated tree loss drop in Manyata Block D.
   - **Tab 3**: Audit snapshot table with functioning CSV/Excel download buttons.
   - **Tab 4**: Model Evaluation & Benchmark tab with quantitative metrics and Viva defense points.

---

## Code Base Structure

```
.
├── app.py                      # Streamlit dashboard UI
├── asset_register.py           # SQLite asset database & loss alert engine
├── detect.py                   # Hardened OWL-ViT object detector & CLI batch tool
├── generate_sample_register.py # Synthetic sample image & historical data generator
├── utils/
│   ├── __init__.py
│   └── geotag.py               # EXIF GPS parser & zone mapper
├── tests/
│   ├── __init__.py
│   └── test_asset_register.py  # Pytest suite
├── sample_data/                # Synthetic fixture images with EXIF tags
├── asset_register.db           # SQLite database
├── Makefile                    # Task automation
├── requirements.txt            # Pinned dependency specifications
└── README.md                   # Documentation
```

---

## Limitations & Production Roadmap

- **Model Context**: OWL-ViT is a general-purpose open-vocabulary detector. For production deployment across Indian flora, species-specific queries can be refined.
- **Plant Health Classification**: The `classify_plant_health` interface in `detect.py` returns stub values; plug in Nikita's MobileNetV2 weights directly into `classify_plant_health()`.
- **EXIF GPS**: Requires location services enabled on site camera smartphones to auto-fill zones; manual selection is always available as fallback.
