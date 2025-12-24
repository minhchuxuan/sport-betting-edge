# ⚽ Sports Betting Edge System

> **CRISP-DM Sports Outcome Prediction & Value Betting Analysis**

Business Analytics Capstone Project - Predicting football match outcomes with calibrated probabilities and identifying positive expected value betting opportunities.

**⚠️ IMPORTANT:** Backtesting shows **negative ROI** (-64% XGBoost, -41% Baseline). This is an academic project demonstrating ML methodology, not a profitable betting system.

---

## 🎯 Overview

This system predicts football match outcomes (Home/Draw/Away), calibrates probabilities, and identifies value bets where model estimates exceed bookmaker odds.

**Key Features:**
- ✅ Elo-based team strength ratings
- ✅ Probability calibration (Brier Score < 0.20)
- ✅ Kelly Criterion bankroll management
- ✅ Interactive Streamlit dashboard
- ✅ **Time period analysis** (custom date range backtesting)
- ✅ Time-series aware validation (no data leakage)
- ✅ Honest reporting of negative results

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Navigate to project
cd "d:\OneDrive - Hanoi University of Science and Technology\HUST classes\ba-20251\codebase"

# Activate environment
conda activate ba-20251

# Install dependencies (if not already done)
pip install -r requirements.txt
```

### 2. Run Dashboard

```bash
# Launch Streamlit dashboard
streamlit run app/streamlit_app.py
```

Dashboard opens automatically at `http://localhost:8501`

### 3. Explore Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open notebooks in order:
# 01-eda.ipynb
# 02-feature-engineering.ipynb
# 03-model-training.ipynb
# 04-evaluation-backtest.ipynb
```

---

## 📊 Results Summary

### Model Performance

| Model | Accuracy | Brier Score | Log Loss | Backtest ROI |
|-------|----------|-------------|----------|--------------|
| **Baseline (Logistic)** | 57.6% | 0.183 | 0.933 | **-41.1%** |
| **XGBoost (Calibrated)** | 59.7% | 0.184 | 0.944 | **-63.6%** |

**Key Finding:** Models are well-calibrated (Brier < 0.20) but cannot beat bookmaker odds consistently.

### Data Source

- **Source:** Football-Data.co.uk
- **League:** English Premier League (E0)
- **Seasons:** 2019-20 to 2023-24 (5 seasons)
- **Matches:** 1,884 total (1,507 train, 377 test)
- **Features:** 15 engineered features (Elo, rolling stats, form)

---

## 📁 Project Structure

```
codebase/
├── data/
│   ├── raw/              # Downloaded CSV files
│   ├── processed/        # Cleaned data (M2)
│   └── demo/             # Sample data (M6)
│
├── src/
│   ├── data/             # Ingestion & validation
│   ├── features/         # Feature engineering (Elo, rolling stats)
│   ├── models/           # Training & calibration
│   └── evaluation/       # Metrics & backtesting
│
├── notebooks/            # Jupyter notebooks (EDA, experiments)
├── models/               # Trained model artifacts
├── reports/figures/      # Evaluation plots
├── app/                  # Streamlit dashboard (M6)
├── tests/                # Unit tests
│
├── scripts/
│   └── download_data.py  # Data acquisition
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 📊 Methodology (CRISP-DM)

1. **Business Understanding:** Identify value bets with positive EV
2. **Data Understanding:** EDA on 5 seasons of match data
3. **Data Preparation:** Clean data, engineer features (Elo, form)
4. **Modeling:** XGBoost + Isotonic Calibration
5. **Evaluation:** Brier Score, Log Loss, ROI backtesting
6. **Deployment:** Streamlit dashboard with SHAP explainability

---

## 🔬 Key Concepts

### Time-Series Integrity
- `TimeSeriesSplit` for cross-validation
- Chronological sorting enforced
- `shift(1)` on all rolling features
- Pre-match Elo values only

### Probability Calibration
- Isotonic Regression via `CalibratedClassifierCV`
- Reliability diagrams
- Brier Score as primary metric

### Decision-Centric Evaluation
- Expected Value (EV) calculation
- Kelly Criterion for stake sizing
- Bankroll simulation with drawdown analysis

---

## 📦 Dependencies

Core packages (see `requirements.txt`):
- **Data:** pandas, numpy
- **ML:** scikit-learn, xgboost
- **Viz:** matplotlib, seaborn, streamlit
- **Explainability:** shap
- **Testing:** pytest

---

## 🎓 Course Alignment

| Module | Concept | Implementation |
|--------|---------|----------------|
| M1 | CRISP-DM, KPIs | Project structure, DOC framework |
| M3 | Data Quality, Features | Elo ratings, rolling statistics |
| M4 | EDA, Visualization | Exploratory notebooks |
| M6 | Regression | Probability estimation |
| M7 | Classification, Calibration | XGBoost + Isotonic Regression |

---

## 🎓 What This Project Demonstrates

### Academic Success ✅
- **CRISP-DM Methodology:** Complete end-to-end pipeline
- **Feature Engineering:** Elo ratings, rolling statistics
- **Probability Calibration:** Brier Score < 0.20
- **Time-Series Validation:** No data leakage
- **Risk Management:** Kelly Criterion implementation
- **Honest Reporting:** Negative results documented

### Real-World Reality ⚠️
- **Cannot Beat Bookmakers:** Negative ROI in backtesting
- **Missing Critical Data:** No injuries, lineups, weather
- **Bookmaker Efficiency:** Professional odds compilers
- **Built-in Margin:** Overround ensures bookmaker profit

---

## 📂 Repository Structure

```
codebase/
├── app/
│   └── streamlit_app.py          # Interactive dashboard
├── data/
│   ├── raw/                       # 5 seasons CSV files
│   └── processed/                 # Cleaned parquet files
├── models/
│   ├── logistic_baseline.joblib   # Baseline model
│   ├── xgb_calibrated.joblib      # Calibrated XGBoost
│   └── label_map.joblib           # Label encoding
├── notebooks/
│   ├── 01-eda.ipynb              # Exploratory analysis
│   ├── 02-feature-engineering.ipynb
│   ├── 03-model-training.ipynb
│   └── 04-evaluation-backtest.ipynb
├── src/
│   ├── data/                      # Ingestion & validation
│   ├── features/                  # Elo, rolling stats
│   ├── models/                    # Training pipeline
│   └── evaluation/                # Backtesting engine
├── reports/
│   ├── evaluation_report.md       # Detailed analysis
│   └── figures/                   # Charts and plots
└── tests/
    └── test_leakage.py           # Leakage prevention tests
```

---

## 🔧 Usage Examples

### Train Models

```bash
# Train both models from scratch
python src/models/train.py
```

### Run Backtesting

```python
import sys
sys.path.insert(0, 'src')
from evaluation.backtest import BettingBacktester
import joblib
import pandas as pd

# Load model and data
model = joblib.load('models/xgb_calibrated.joblib')
df = pd.read_parquet('data/processed/features.parquet')

# Run backtest
backtester = BettingBacktester(
    initial_bankroll=1000,
    kelly_fraction=0.25,
    ev_threshold=0.0
)
results = backtester.run(df_test, model_proba)
backtester.get_summary(results)
```

### Launch Dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## 📊 Milestones Completed

| Milestone | Status | Description |
|-----------|--------|-------------|
| M1 | ✅ | Project setup & data acquisition |
| M2 | ✅ | Data ingestion & cleaning |
| M3 | ✅ | Feature engineering (Elo, rolling stats) |
| M4 | ✅ | Model training & calibration |
| M5 | ✅ | Evaluation & backtesting |
| M6 | ✅ | Dashboard development |
| M7 | ✅ | Documentation & demo preparation |

---

## ⚠️ Disclaimer

**DO NOT USE FOR REAL GAMBLING**

This is an academic project demonstrating:
- Data science methodology
- Probability theory
- Risk management concepts

**Key Points:**
- Models show **negative ROI** in backtesting
- Cannot beat bookmaker odds consistently
- Missing critical data sources
- High risk of financial loss
- Past performance ≠ future results

**Gambling carries significant financial risk. The authors accept no responsibility for financial losses.**

---

## 📝 License

MIT License - Academic use only

**Data Source:** Football-Data.co.uk (free for academic/personal use)

---

## 📧 Contact

For academic inquiries only.

**Status:** ✅ Project Complete - All 7 Milestones Delivered  
**Last Updated:** December 2024
