from pathlib import Path
import joblib, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"dataset"/"synthetic_transactions.csv"
MODEL=ROOT/"models"/"fraud_model.joblib"
FEATURES=["amount","hour","distance_from_home_km","transactions_last_24h","card_age_days","international","merchant_risk","failed_attempts"]

def train():
    if not DATA.exists():
        from generate_data import generate
        generate()
    df=pd.read_csv(DATA)
    X,y=df[FEATURES],df["is_fraud"]
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    clf=RandomForestClassifier(n_estimators=160,max_depth=12,min_samples_leaf=2,class_weight="balanced_subsample",random_state=42,n_jobs=-1)
    clf.fit(Xtr,ytr)
    p=clf.predict_proba(Xte)[:,1]; pred=(p>=.5).astype(int)
    metrics={"accuracy":round(accuracy_score(yte,pred),3),"precision":round(precision_score(yte,pred,zero_division=0),3),
      "recall":round(recall_score(yte,pred,zero_division=0),3),"f1":round(f1_score(yte,pred,zero_division=0),3),
      "roc_auc":round(roc_auc_score(yte,p),3)}
    MODEL.parent.mkdir(exist_ok=True)
    joblib.dump({"model":clf,"features":FEATURES,"metrics":metrics},MODEL)
    print("Saved model:",MODEL); print("Demo test metrics:",metrics)
    return metrics

if __name__=="__main__": train()
