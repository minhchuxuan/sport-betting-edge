# 🎬 Demo Script - Sports Betting Edge System

**Duration:** 10 minutes  
**Audience:** Academic review / Capstone presentation  
**Goal:** Demonstrate complete CRISP-DM pipeline with honest results

---

## 📋 Pre-Demo Checklist

### Before Starting
- [ ] Activate conda environment: `conda activate ba-20251`
- [ ] Navigate to project: `cd codebase`
- [ ] Have browser ready
- [ ] Close unnecessary applications
- [ ] Prepare to show code editor

---

## 🎯 Demo Flow (10 Minutes)

### **0:00-1:00 - Introduction & Problem Statement**

**What to Say:**
> "This capstone project implements a complete sports betting analytics system following CRISP-DM methodology. The goal was to predict football match outcomes and identify value betting opportunities where our model's probability estimates exceed bookmaker odds."

**Show:**
- Project folder structure
- README.md overview

**Key Points:**
- Business goal: Find positive expected value bets
- Academic focus: Demonstrate ML pipeline
- Honest reporting: Models show negative ROI

---

### **1:00-2:30 - Data & Feature Engineering**

**What to Say:**
> "We collected 5 seasons of Premier League data - 1,884 matches total. The key challenge was preventing data leakage in time-series data."

**Show:**
- `data/raw/` - 5 CSV files
- `notebooks/01-eda.ipynb` - Quick scroll through EDA

**Demonstrate:**
```bash
# Show data
python -c "import pandas as pd; df = pd.read_parquet('data/processed/features.parquet'); print(f'Shape: {df.shape}'); print(f'Columns: {list(df.columns)[:10]}')"
```

**Key Points:**
- 15 engineered features
- Elo ratings (team strength)
- Rolling statistics (recent form)
- Strict chronological ordering

---

### **2:30-4:00 - Feature Engineering Deep Dive**

**What to Say:**
> "Feature engineering was critical. We implemented Elo ratings and rolling statistics with careful leakage prevention."

**Show:**
- `src/features/elo.py` - Scroll to `calculate_elo_ratings()` function
- `tests/test_leakage.py` - Show leakage prevention tests

**Run Test:**
```bash
python tests/test_leakage.py
```

**Key Points:**
- Pre-match Elo values only
- `shift(1)` on rolling features
- All tests pass - no leakage!

---

### **4:00-5:30 - Model Training & Calibration**

**What to Say:**
> "We trained two models: a baseline Logistic Regression and an advanced XGBoost with Isotonic calibration. Both achieve excellent probability calibration with Brier scores under 0.20."

**Show:**
- `notebooks/03-model-training.ipynb` - Open and show results
- Reliability diagrams
- Confusion matrices

**Key Metrics:**
```
Baseline:  Accuracy 57.6%, Brier 0.183
XGBoost:   Accuracy 59.7%, Brier 0.184
```

**Key Points:**
- Good calibration (Brier < 0.20)
- XGBoost slightly more accurate
- Both well-calibrated

---

### **5:30-7:00 - Backtesting Results (The Honest Part)**

**What to Say:**
> "Here's where it gets interesting - and honest. Despite good calibration, both models show NEGATIVE ROI in backtesting. This demonstrates an important lesson: good ML models don't automatically translate to profitable betting."

**Show:**
- `notebooks/04-evaluation-backtest.ipynb` - Run or show results
- PnL curves (downward trend)
- `reports/evaluation_report.md` - Scroll to "Why Models Lose"

**Key Results:**
```
Baseline:  ROI -41.1%, Win Rate 33.5%
XGBoost:   ROI -63.6%, Win Rate 27.2%
```

**Key Points:**
- Negative ROI despite good calibration
- Bookmakers are very efficient
- Missing critical data (injuries, lineups)
- Honest scientific reporting

---

### **7:00-9:00 - Interactive Dashboard**

**What to Say:**
> "Let me show you the interactive dashboard. Notice the prominent warning banner - we're transparent about the negative results."

**Launch Dashboard:**
```bash
streamlit run app/streamlit_app.py
```

**Demonstrate:**

1. **Warning Banner** (0:30)
   - Point out negative ROI warning
   - "DO NOT use for gambling" message

2. **Tab 1: Match Predictions** (1:00)
   - Select a match from dropdown
   - Show team Elo ratings
   - Display model probabilities
   - Compare to bookmaker odds
   - Point out Expected Value
   - Show Kelly stake suggestion

3. **Tab 2: Model Performance** (0:30)
   - Show backtesting results table
   - Highlight negative ROI
   - Explain why models fail

4. **Interactive Controls** (0:30)
   - Change Kelly fraction
   - Switch models
   - Adjust EV threshold

**Key Points:**
- User-friendly interface
- Transparent about failures
- Educational tool, not betting advice
- Real predictions on test data

---

### **9:00-10:00 - Conclusion & Q&A**

**What to Say:**
> "This project demonstrates a complete CRISP-DM pipeline with honest reporting. While the models can't beat bookmakers, the project successfully shows feature engineering, probability calibration, time-series validation, and responsible AI practices."

**Summary Points:**
1. ✅ **Technical Success:** Good calibration, no leakage, proper validation
2. ✅ **Honest Reporting:** Negative results documented
3. ✅ **Academic Value:** Complete methodology demonstrated
4. ⚠️ **Real-World Reality:** Cannot beat bookmakers

**Prepare for Questions:**
- Q: "Why show negative results?"
  - A: "Honest scientific reporting. Negative results are valuable."
  
- Q: "Could this be improved?"
  - A: "Yes - need more data (injuries, lineups), real-time info, lower-efficiency markets."
  
- Q: "Which model is better?"
  - A: "Surprisingly, the simpler Baseline model loses less money despite lower accuracy."

---

## 🎯 Key Messages to Emphasize

### 1. **Complete CRISP-DM Pipeline** ✅
- Business Understanding → Deployment
- All phases implemented
- Professional structure

### 2. **Technical Competence** ✅
- Feature engineering (Elo, rolling stats)
- Probability calibration (Brier < 0.20)
- Time-series validation
- Leakage prevention
- Kelly Criterion
- Interactive dashboard

### 3. **Intellectual Honesty** ✅
- Negative ROI reported prominently
- Limitations explained clearly
- No misleading claims
- Educational focus

### 4. **Responsible AI** ✅
- Clear disclaimers
- Transparent about failures
- Ethical considerations
- Academic purpose only

---

## 🛠️ Backup Plans

### If Dashboard Won't Start
- Have screenshots ready
- Show notebook outputs instead
- Walk through code

### If Jupyter Won't Open
- Show HTML exports
- Display saved figures
- Explain from code

### If Questions Get Technical
- Reference specific files
- Show code snippets
- Explain methodology

---

## 📸 Screenshots to Prepare

1. **Project Structure** - Folder tree
2. **EDA Notebook** - Key visualizations
3. **Feature Engineering** - Elo calculation code
4. **Model Training** - Reliability diagrams
5. **Backtesting** - PnL curves (downward)
6. **Dashboard** - All three tabs
7. **Warning Banner** - Negative ROI message

---

## 💡 Pro Tips

### Do's ✅
- Emphasize honest reporting
- Show real code, not just slides
- Explain technical decisions
- Acknowledge limitations
- Be confident about negative results

### Don'ts ❌
- Don't apologize for negative ROI
- Don't oversell capabilities
- Don't skip the warning banner
- Don't claim profitability
- Don't rush through methodology

---

## 🎤 Opening Statement (Memorize)

> "Good morning/afternoon. Today I'm presenting my capstone project: a Sports Betting Edge System built following CRISP-DM methodology. This project demonstrates a complete machine learning pipeline from data acquisition through deployment, with a focus on probability calibration and responsible AI practices. 
>
> I want to be upfront: the models show negative ROI in backtesting. But this honest result actually makes the project more valuable academically - it demonstrates both technical competence and scientific integrity. Let me walk you through what we built."

---

## 🎯 Closing Statement (Memorize)

> "To summarize: This project successfully demonstrates a complete data science pipeline with proper time-series validation, excellent probability calibration, and honest reporting of negative results. While we can't beat bookmakers, we've shown mastery of feature engineering, model calibration, backtesting, and responsible AI deployment. The negative results teach an important lesson: good machine learning doesn't automatically solve every business problem - domain expertise and data quality matter enormously. Thank you, and I'm happy to answer questions."

---

## ⏱️ Time Management

| Section | Time | Cumulative |
|---------|------|------------|
| Introduction | 1:00 | 1:00 |
| Data & Features | 1:30 | 2:30 |
| Feature Engineering | 1:30 | 4:00 |
| Model Training | 1:30 | 5:30 |
| Backtesting | 1:30 | 7:00 |
| Dashboard Demo | 2:00 | 9:00 |
| Conclusion | 1:00 | 10:00 |

**Buffer:** Keep 2-3 minutes for questions

---

## ✅ Post-Demo Checklist

- [ ] Stop Streamlit (Ctrl+C)
- [ ] Close browser tabs
- [ ] Save any notes
- [ ] Thank reviewers
- [ ] Collect feedback

---

**Good luck with your demo! Remember: Honest negative results are scientifically valuable!** 🎉

