# 🚀 How to Run the Dashboard

## Quick Start (3 Steps)

### Step 1: Open Terminal
- Open Command Prompt or PowerShell
- Or use terminal in VS Code/Cursor

### Step 2: Navigate to Project
```bash
cd "d:\OneDrive - Hanoi University of Science and Technology\HUST classes\ba-20251\codebase"
```

### Step 3: Run Streamlit
```bash
streamlit run app/streamlit_app.py
```

**That's it!** The dashboard will automatically open in your browser.

---

## What You'll See

### Automatic Browser Launch
- Dashboard opens at `http://localhost:8501`
- If browser doesn't open, manually go to the URL shown in terminal

### Dashboard Features

**Tab 1: Match Predictions** 📊
- Select any match from dropdown (377 test matches)
- See team Elo ratings and recent form
- View model predictions vs bookmaker odds
- Check Expected Value (EV)
- See if it's a "value bet"
- Get Kelly stake suggestions

**Tab 2: Model Performance** 📈
- View backtesting results
- See why models lose money
- Understand limitations

**Tab 3: About** ℹ️
- Project overview
- Model descriptions
- Data sources
- Ethical disclaimer

### Sidebar Controls
- **Model**: Choose Baseline or XGBoost
- **Bankroll**: Set hypothetical amount ($100-$10,000)
- **Kelly Fraction**: Risk level (0.1 = conservative, 1.0 = aggressive)
- **EV Threshold**: Filter value bets (0-10%)

---

## Troubleshooting

### Port Already in Use
If you see "Port 8501 is already in use":
```bash
# Option 1: Use different port
streamlit run app/streamlit_app.py --server.port 8502

# Option 2: Kill existing Streamlit
# Press Ctrl+C in the terminal running Streamlit
```

### Module Not Found Error
If you see "ModuleNotFoundError":
```bash
# Make sure you're in the right directory
cd "d:\OneDrive - Hanoi University of Science and Technology\HUST classes\ba-20251\codebase"

# Check if models exist
dir models\*.joblib
```

### Models Not Loading
If dashboard shows "Failed to load models":
```bash
# Verify models exist
dir models

# Should see:
# - logistic_baseline.joblib
# - xgb_calibrated.joblib
# - label_map.joblib
```

---

## Stopping the Dashboard

### Method 1: Terminal
- Press `Ctrl+C` in the terminal
- Dashboard will stop

### Method 2: Close Terminal
- Just close the terminal window
- Dashboard will stop automatically

---

## Tips for Exploring

### Try Different Matches
- Use dropdown to select different matches
- Each match shows different predictions
- Compare model confidence across matches

### Adjust Kelly Fraction
- Start with 0.25 (quarter Kelly - conservative)
- Try 0.5 (half Kelly - moderate)
- See how stakes change

### Compare Models
- Switch between Baseline and XGBoost
- Notice XGBoost is more confident
- But Baseline performed better in backtesting!

### Look for Value Bets
- Green "VALUE BET" means positive EV
- But remember: backtesting showed losses
- This is for educational purposes only

---

## What to Notice

### 1. Warning Banner
- Top of page shows negative ROI results
- Clear "DO NOT use for real gambling" message
- Honest about model limitations

### 2. Probability Comparison
- Model probabilities vs bookmaker odds
- Chart shows visual comparison
- Notice how close they often are

### 3. Expected Value
- Positive EV = model thinks bet has value
- But many positive EV bets still lost money
- Shows difficulty of beating bookmakers

### 4. Backtesting Results
- Tab 2 shows real results
- Both models lost money
- Honest scientific reporting

---

## Example Session

1. **Start dashboard**
   ```bash
   streamlit run app/streamlit_app.py
   ```

2. **Select a match**
   - Choose "Match 50" from dropdown
   - See Arsenal vs Chelsea (example)

3. **Check predictions**
   - Model says Home Win: 65%
   - Bookmaker odds: 1.67 (60% implied)
   - EV: +3.2% (value bet!)

4. **Adjust settings**
   - Change Kelly fraction to 0.5
   - See stake increase

5. **Compare models**
   - Switch to Baseline
   - See different probabilities
   - Compare confidence levels

6. **Read performance**
   - Go to Tab 2
   - See backtesting results
   - Understand why models lose

---

## Screenshots to Take

For your documentation, capture:
1. Main page with warning banner
2. Match prediction with value bet
3. Probability comparison chart
4. Model performance table
5. About page with disclaimer

---

## Common Questions

**Q: Why does it show negative ROI?**
A: Because that's the honest result! Models lost money in backtesting.

**Q: Can I use this for real betting?**
A: NO! This is academic only. Models show negative returns.

**Q: Which model is better?**
A: Baseline performed better in backtesting despite lower accuracy.

**Q: What's a value bet?**
A: When model probability > bookmaker probability. But doesn't guarantee profit!

**Q: Why show this if models lose money?**
A: Demonstrates complete ML pipeline and honest scientific reporting.

---

## After You're Done

1. **Stop dashboard**: Press `Ctrl+C`
2. **Close browser tab**
3. **Document your findings**

---

## Need Help?

Check these files:
- `M6_COMPLETE.md` - Full documentation
- `reports/evaluation_report.md` - Detailed analysis
- `README.md` - Project overview

---

**Enjoy exploring the dashboard!** 🎉

Remember: This is for learning, not gambling!

