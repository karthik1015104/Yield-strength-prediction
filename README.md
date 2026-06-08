# MPEA Yield Strength Predictor

Predicts the **yield strength (MPa)** of Multi-Principal Element Alloys (MPEAs) from elemental composition, microstructure, and processing conditions using gradient boosting and thermodynamic feature engineering.

**Test R² = 0.811 · MAE = 165 MPa · 1,067 samples · 53 features**

---

## Problem

Multi-Principal Element Alloys (also called High-Entropy Alloys) are a class of advanced structural materials with extraordinary mechanical properties. Experimentally measuring yield strength is expensive and slow. This project builds a machine learning model that predicts YS directly from composition and processing conditions, enabling faster alloy screening.

---

## Dataset

- **Source:** MPEA experimental database (publicly available)
- **Raw data:** 1,545 entries, 23 columns
- **After cleaning:** 1,067 samples with valid yield strength measurements
- **Target:** Yield Strength in MPa (range: 24 – 3,416 MPa)
- **Alloy systems:** AlCoCrFeNi family, refractory MPEAs, Ti-based alloys, and more

---

## Feature Engineering

Raw input is a formula string like `Al0.5 Co1 Cr1 Fe1 Ni1`. The pipeline extracts 53 features in three groups:

### Thermodynamic Descriptors (8 features)
| Feature | Description | Physical Meaning |
|---------|-------------|-----------------|
| VEC | Valence electron concentration | Predicts BCC (< 6.87) vs FCC (> 8) phase |
| δ (delta_r) | Atomic size mismatch (%) | Drives solid solution strengthening |
| ΔS_mix | Ideal mixing entropy (J/mol·K) | Thermodynamic stability indicator |
| ΔH_mix | Mixing enthalpy (kJ/mol) | Phase formation tendency |
| T_melt | Weighted avg melting point (K) | High-temperature strength proxy |
| Ω (omega) | T_melt · ΔS / |ΔH| | Single-phase solid solution indicator |
| Δχ (delta_en) | Electronegativity difference | Chemical bonding character |
| num_elements | Number of principal elements | Complexity/entropy contribution |

### Composition Features (30 features)
Normalised mole fractions for all 30 elements found in the dataset (x_Al, x_Co, x_Cr, ... x_Zr). Zero for absent elements.

### Process & Condition Features (15 features)
- **test_temperature** — test temperature in °C (crucial: RT vs 1000°C changes YS by hundreds of MPa)
- **test_type_compression** — binary flag for compression (1) vs tension (0)
- **micro_\*** — simplified microstructure: BCC_single, FCC_single, BCC_based, FCC_based, Mixed, Unknown
- **proc_\*** — processing method: CAST, WROUGHT, ANNEAL, POWDER, OTHER
- **grain_size_known** — binary flag (honest treatment of 81% missing grain size data)
- **grain_size_log** — log-transformed grain size (0 if unknown)

> **Why not impute grain size?** 860 of 1,067 samples have no grain size measurement. Imputing with a single median value (17.1 µm) would make it appear as the 4th most important feature while carrying no real information. We use a binary known/unknown flag instead.

---

## Model

**Algorithm:** XGBoost Regressor with RandomizedSearchCV tuning (60 candidates, 5-fold CV)

**Best hyperparameters:**
```
n_estimators=500, max_depth=6, learning_rate=0.03
subsample=0.8, colsample_bytree=0.8
min_child_weight=1, gamma=0.3
reg_alpha=0.1, reg_lambda=1.5
```

### Results

| Metric | Value |
|--------|-------|
| Train R² | 0.971 |
| **Test R²** | **0.811** |
| Test MAE | 165 MPa |
| Test RMSE | 255 MPa |
| Target mean (test) | 920 MPa |

### Model Comparison (5-fold CV on training set)

| Model | CV R² |
|-------|-------|
| Ridge Regression | 0.665 ± 0.042 |
| Random Forest | 0.730 ± 0.018 |
| Gradient Boosting | 0.794 ± 0.020 |
| **XGBoost (tuned)** | **0.816 ± 0.025** |

### Why the train/test gap (0.97 vs 0.81)?

The gap is not purely overfitting. The dataset contains the same alloy composition tested at multiple temperatures and processing states — the model sees some of these conditions in training. Beyond that, the residual variance reflects genuine **physical uncertainty**: identical composition and processing can yield different YS depending on microstructural details not captured in the features (grain boundary character, dislocation density, precipitate morphology). A test R² of 0.81 is strong for this domain.

---

## Repository Structure

```
mpea-ys-predictor/
│
├── data/
│   ├── MPEA_dataset.csv              # Raw dataset
│   └── YS_MPEA_ML_ready_v2.csv      # Processed feature matrix (53 features)
│
├── notebooks/
│   └── MPEA_YS_Prediction.ipynb     # Full pipeline notebook
│
├── app.py                            # Streamlit web application
├── YS_model_v2.pkl                   # Saved XGBoost model
├── feature_columns.csv               # Feature column order for inference
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## Running the App

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

Make sure `YS_model_v2.pkl` and `feature_columns.csv` are in the same directory as `app.py`.

### Requirements

```
streamlit
scikit-learn
xgboost
lightgbm
shap
pandas
numpy
plotly
joblib
matplotlib
seaborn
```

---

## Making a Prediction (Python)

```python
import joblib
import pandas as pd
import numpy as np
import re

model = joblib.load('YS_model_v2.pkl')
feat_cols = pd.read_csv('feature_columns.csv', header=None)[0].tolist()

# See notebook Cell 20 for the full predict_yield_strength() function
result = predict_yield_strength(
    formula            = 'Al0.5 Co1 Cr1 Fe1 Ni1',
    test_temp_C        = 25,
    test_type          = 'C',
    microstructure     = 'BCC',
    processing_method  = 'CAST',
    grain_size_um      = None
)
# Predicted YS: ~620 MPa
```

---

## Key Findings

- **Test temperature** is the most important feature — high-temperature testing dramatically reduces YS
- **VEC** (valence electron concentration) is the strongest composition-level predictor, consistent with the BCC/FCC phase stability literature
- **Atomic size mismatch (δ)** ranks highly — larger δ means more lattice distortion and stronger solid solution strengthening
- **Mixing enthalpy (ΔH_mix)** captures thermodynamic phase stability
- BCC alloys show higher average YS than FCC alloys in this dataset (consistent with known BCC solid solution strengthening behaviour)
- WROUGHT and ANNEAL processing produces more consistent YS than CAST

---

## Limitations

- Model trained on literature data — experimental scatter and reporting inconsistencies affect accuracy
- Grain size data is available for only 19% of samples
- Does not account for precipitate strengthening, texture, or dislocation density
- Extrapolation to compositions outside training distribution may be unreliable

---

## References

- Takeuchi & Inoue (2005) — Binary mixing enthalpy table
- Yang & Zhang (2012) — Omega parameter for phase stability prediction
- Miracle & Senkov (2017) — *A critical review of MPEA/HEA research*, Acta Materialia

---

*Built by Karthik · IIT Madras · Undergraduate, expected graduation May 2028*
*MS AI/ML applicant, Fall 2028*
