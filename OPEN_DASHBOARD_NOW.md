# 🚀 OPEN DASHBOARD NOW - Simple Guide

## Quick 3-Step Process

### Step 1: Open Terminal
- Open Command Prompt or PowerShell
- Or use terminal in your IDE (VS Code/Cursor)

### Step 2: Copy & Paste This Command
```bash
cd "d:\OneDrive - Hanoi University of Science and Technology\HUST classes\ba-20251\codebase" && streamlit run app/streamlit_app.py
```

### Step 3: Wait for Browser
- Dashboard will automatically open
- URL: `http://localhost:8501`
- If browser doesn't open, manually go to the URL

**That's it!** 🎉

---

## What You'll See

### Top of Page
- **Big Yellow Warning Box** saying "DO NOT use for gambling"
- Shows negative ROI results (-64% XGBoost, -41% Baseline)

### Three Tabs

**📊 Tab 1: Match Predictions**
- Dropdown to select matches (377 available)
- Team Elo ratings and form
- Model predictions vs bookmaker odds
- Expected Value calculations
- "VALUE BET" indicators (green)
- Kelly stake suggestions

**📈 Tab 2: Model Performance**
- Table showing backtesting results
- Both models show negative ROI
- Explanation of why models fail

**ℹ️ Tab 3: About**
- Project description
- Model details
- Ethical disclaimer

### Sidebar (Left)
- Model selection (Baseline / XGBoost)
- Bankroll slider ($100-$10,000)
- Kelly Fraction (0.1-1.0)
- EV Threshold (0%-10%)

---

## Things to Try

### 1. Select Different Matches
- Use dropdown at top
- Try Match 1, Match 50, Match 377
- See how predictions vary

### 2. Compare Models
- Start with XGBoost (default)
- Switch to Baseline in sidebar
- Notice different probabilities

### 3. Adjust Risk
- Change Kelly Fraction to 0.5 (more aggressive)
- See stakes increase
- Try 0.1 (very conservative)

### 4. Look for Value Bets
- Green "VALUE BET" means positive EV
- But remember: backtesting showed losses!
- This is educational only

---

## To Stop Dashboard

Press `Ctrl+C` in the terminal

---

## If Something Goes Wrong

### Port Already in Use?
```bash
streamlit run app/streamlit_app.py --server.port 8502
```

### Can't Find Models?
Make sure you're in the right directory:
```bash
cd "d:\OneDrive - Hanoi University of Science and Technology\HUST classes\ba-20251\codebase"
dir models\*.joblib
```

Should see:
- logistic_baseline.joblib
- xgb_calibrated.joblib
- label_map.joblib

---

## Example Session

1. **Start dashboard** (command above)
2. **Select Match 50** from dropdown
3. **See predictions** - maybe Home Win: 65%
4. **Check EV** - maybe +3.2% (value bet!)
5. **Adjust Kelly** - change to 0.5, see stake double
6. **Switch model** - try Baseline, compare
7. **Go to Tab 2** - see why models lose money
8. **Read Tab 3** - understand the project

---

## Key Things to Notice

### 1. Warning Banner
- Very prominent at top
- Shows real negative results
- Clear "DO NOT gamble" message

### 2. Honest Results
- Tab 2 shows actual backtesting
- Both models lost money
- Transparent about failures

### 3. Interactive
- Change settings, see results update
- Select matches, see predictions
- Compare models side-by-side

### 4. Professional
- Clean design
- Clear metrics
- Easy to understand

---

## Screenshot Checklist

For your documentation, capture:
1. ✅ Warning banner at top
2. ✅ Match prediction with value bet
3. ✅ Probability comparison chart
4. ✅ Model performance table (Tab 2)
5. ✅ About page with disclaimer (Tab 3)

---

## Questions You Might Have

**Q: Why does it show negative ROI?**
A: Because that's the real result! Honest reporting.

**Q: Can I use this for betting?**
A: NO! Academic project only. Models lose money.

**Q: Which model is better?**
A: Baseline actually performed better (-41% vs -64%)

**Q: What's a value bet?**
A: When model probability > bookmaker probability

---

## After You're Done

1. Stop dashboard: `Ctrl+C`
2. Close browser tab
3. Take screenshots if needed
4. Document your observations

---

**Ready? Run the command and explore!** 🎉

```bash
cd "d:\OneDrive - Hanoi University of Science and Technology\HUST classes\ba-20251\codebase" && streamlit run app/streamlit_app.py
```

**Enjoy the demo!**

