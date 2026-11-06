from pathlib import Path
import io, joblib, pandas as pd
from flask import Flask,render_template,request,redirect,url_for,flash,send_file
from generate_data import generate
from train_model import train,FEATURES

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"dataset"/"synthetic_transactions.csv"
MODEL=ROOT/"models"/"fraud_model.joblib"
app=Flask(__name__); app.secret_key="local-educational-demo"

def ensure():
    if not DATA.exists(): generate()
    if not MODEL.exists(): train()

def bundle():
    ensure(); return joblib.load(MODEL)

@app.route("/")
def index():
    ensure(); d=pd.read_csv(DATA); n=len(d); fraud=int(d.is_fraud.sum())
    return render_template("index.html",total=n,fraud=fraud,legit=n-fraud,
      rate=round(100*fraud/n,2),avg=round(d.amount.mean(),2),metrics=bundle()["metrics"],
      recent=d.tail(8).iloc[::-1].to_dict("records"))

@app.route("/predict",methods=["GET","POST"])
def predict():
    result=None
    if request.method=="POST":
      try:
        row={k:float(request.form[k]) for k in FEATURES}
        if row["hour"]<0 or row["hour"]>23: raise ValueError("Hour must be between 0 and 23.")
        if row["international"] not in (0,1): raise ValueError("International must be 0 or 1.")
        if not 0<=row["merchant_risk"]<=1: raise ValueError("Merchant risk must be between 0 and 1.")
        p=float(bundle()["model"].predict_proba(pd.DataFrame([row],columns=FEATURES))[0,1])
        result={"prob":round(p*100,2),"fraud":p>=.5,"label":"Potential Fraud" if p>=.5 else "Likely Legitimate"}
      except Exception as e: flash("Prediction error: "+str(e))
    return render_template("predict.html",result=result)

@app.route("/batch",methods=["GET","POST"])
def batch():
    if request.method=="POST":
      f=request.files.get("file")
      try:
        if not f or not f.filename.lower().endswith(".csv"): raise ValueError("Choose a CSV file.")
        df=pd.read_csv(f); missing=[c for c in FEATURES if c not in df.columns]
        if missing: raise ValueError("Missing columns: "+", ".join(missing))
        X=df[FEATURES].apply(pd.to_numeric,errors="raise")
        p=bundle()["model"].predict_proba(X)[:,1]
        df["fraud_probability_percent"]=(p*100).round(2)
        df["prediction"]=["Potential Fraud" if x>=.5 else "Likely Legitimate" for x in p]
        out=io.BytesIO(); df.to_csv(out,index=False); out.seek(0)
        return send_file(out,mimetype="text/csv",as_attachment=True,download_name="fraud_predictions.csv")
      except Exception as e: flash("CSV error: "+str(e))
    return render_template("batch.html")

if __name__=="__main__":
    ensure()
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(host="127.0.0.1",port=5000,debug=False)
