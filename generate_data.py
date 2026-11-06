from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"dataset"/"synthetic_transactions.csv"

def generate(n=12000, seed=42):
    OUT.parent.mkdir(exist_ok=True)
    rng=np.random.default_rng(seed)
    amount=np.clip(rng.lognormal(3.5,1.0,n),1,5000)
    hour=rng.integers(0,24,n)
    distance=np.clip(rng.gamma(2,18,n),0,500)
    velocity=rng.poisson(2.2,n)
    age=np.clip(rng.gamma(4,180,n),1,4000)
    international=rng.binomial(1,.12,n)
    merchant=rng.beta(2,5,n)
    failed=rng.poisson(.25,n)
    score=-4.7+.00055*amount+.045*distance+.38*velocity+1.25*international+2.1*merchant+.55*failed+.9*(hour<=4)-.00035*age
    prob=1/(1+np.exp(-score))
    fraud=rng.binomial(1,np.clip(prob,.002,.92))
    df=pd.DataFrame({"amount":amount.round(2),"hour":hour,"distance_from_home_km":distance.round(2),
      "transactions_last_24h":velocity,"card_age_days":age.round().astype(int),"international":international,
      "merchant_risk":merchant.round(3),"failed_attempts":failed,"is_fraud":fraud})
    df.to_csv(OUT,index=False)
    return df

if __name__=="__main__":
    d=generate()
    print(f"Generated {len(d)} synthetic demo transactions at {OUT}")
