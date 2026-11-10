from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .config import FEATURES, MODEL_DIR

MODEL_PATH=MODEL_DIR/'crop_advisor.joblib'; META_PATH=MODEL_DIR/'metadata.json'

def train_model(df):
    X,y=df[FEATURES],df['label']
    can_stratify=y.value_counts().min()>=2 and int(round(len(y)*.2))>=y.nunique()
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y if can_stratify else None)
    rf=RandomForestClassifier(n_estimators=400,max_features='sqrt',class_weight='balanced_subsample',random_state=42,n_jobs=-1)
    et=ExtraTreesClassifier(n_estimators=400,max_features='sqrt',class_weight='balanced',random_state=42,n_jobs=-1)
    model=Pipeline([('scale',StandardScaler()),('model',VotingClassifier([('rf',rf),('et',et)],voting='soft',n_jobs=-1))])
    model.fit(Xtr,ytr); pred=model.predict(Xte)
    metrics={'accuracy':float(accuracy_score(yte,pred)),'f1_weighted':float(f1_score(yte,pred,average='weighted'))}
    MODEL_DIR.mkdir(parents=True,exist_ok=True); joblib.dump(model,MODEL_PATH)
    classes=sorted(map(str,model.named_steps['model'].classes_))
    META_PATH.write_text(json.dumps({'features':FEATURES,'classes':classes,'metrics':metrics,'rows':len(df),'model':'RF + ExtraTrees soft voting'},indent=2))
    return model,metrics

def load_or_train(df):
    if MODEL_PATH.exists() and META_PATH.exists():
        try:return joblib.load(MODEL_PATH),json.loads(META_PATH.read_text())
        except Exception:pass
    m,metrics=train_model(df); return m,{'metrics':metrics,'features':FEATURES}

def predict_top(model,values,n=7):
    X=pd.DataFrame([[float(values[f]) for f in FEATURES]],columns=FEATURES)
    probs=model.predict_proba(X)[0]; classes=model.named_steps['model'].classes_
    order=np.argsort(probs)[::-1][:n]
    return [(str(classes[i]),float(probs[i])) for i in order]
