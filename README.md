# Credit Card Fraud Detection — VS Code Ready

A local Flask + Random Forest machine-learning demo with a dashboard, single-transaction prediction, and CSV batch scoring.

**Important:** This project generates synthetic demo transactions. Its predictions and metrics are educational only, not real banking fraud detection.

## Requirements
- Windows 10/11 64-bit
- Python 3.10, 3.11, or 3.12
- VS Code (Python extension recommended)
- Internet for the initial pip install
- 8 GB RAM recommended; no GPU required

## Run on Windows
1. Extract this ZIP.
2. In VS Code: File → Open Folder → select `Credit_Card_Fraud_Detection_VSCode`.
3. Terminal → New Terminal. Run each command separately:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

If `py -3.11` is unavailable, install Python 3.11 from https://www.python.org/downloads/ and tick **Add python.exe to PATH**. Alternatively use `python -m venv .venv` if your installed Python is 3.10–3.12.

If PowerShell blocks activation:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Open http://127.0.0.1:5000. Stop with Ctrl+C.

The app creates its synthetic dataset and trains/saves its model automatically on first launch. No separate dataset or model download is required.

## CSV upload columns
`amount,hour,distance_from_home_km,transactions_last_24h,card_age_days,international,merchant_risk,failed_attempts`

Use `sample_transactions.csv` as a template.
