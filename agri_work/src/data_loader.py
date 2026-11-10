import io
from pathlib import Path
import requests
import pandas as pd
from .config import CROP_DATA_URL, SOIL_DATA_URL, DATA_DIR


def _download_csv(url: str, path: Path) -> pd.DataFrame:
    r = requests.get(url, timeout=20, headers={'User-Agent':'AgriPulseAI/1.0'})
    r.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(r.content)
    return pd.read_csv(io.BytesIO(r.content))


def _offline_crop_data() -> pd.DataFrame:
    # Small deterministic fallback so first launch still works without internet.
    profiles = {
      'rice':(90,42,43,23,82,6.5,220),'maize':(78,48,38,25,65,6.3,95),
      'chickpea':(40,68,78,20,30,6.4,75),'kidneybeans':(20,68,20,20,65,5.8,80),
      'pigeonpeas':(20,65,20,28,50,6.0,110),'mothbeans':(20,55,20,29,45,6.5,55),
      'mungbean':(30,55,35,28,55,6.5,70),'blackgram':(40,65,20,27,60,6.8,70),
      'lentil':(20,68,18,23,35,6.8,45),'pomegranate':(18,20,35,22,40,6.5,70),
      'banana':(100,82,50,27,80,6.0,210),'mango':(20,28,30,27,55,6.0,95),
      'grapes':(25,25,30,24,65,6.2,70),'watermelon':(100,17,50,27,75,6.2,55),
      'muskmelon':(100,18,50,28,72,6.3,40),'apple':(20,130,200,18,60,6.4,125),
      'orange':(20,20,20,24,65,6.5,110),'papaya':(50,55,50,28,82,6.2,190),
      'coconut':(22,16,22,28,88,6.0,175),'cotton':(120,45,45,28,65,6.5,95),
      'jute':(80,45,40,28,82,6.8,175),'coffee':(100,20,30,22,72,6.0,160)
    }
    rows=[]
    for crop,p in profiles.items():
        for j in range(12):
            vals=list(p)
            vals[:3]=[max(0,v + ((j%5)-2)*2) for v in vals[:3]]
            vals[3]=v if False else max(5,p[3]+((j%4)-1.5)*1.2)
            vals[4]=max(5,min(98,p[4]+((j%3)-1)*2))
            vals[5]=max(3.5,min(9,p[5]+((j%5)-2)*.08))
            vals[6]=max(1,p[6]+((j%4)-1.5)*6)
            rows.append(vals+[crop])
    return pd.DataFrame(rows,columns=['N','P','K','temperature','humidity','ph','rainfall','label'])


def load_crop_data() -> pd.DataFrame:
    path=DATA_DIR/'Crop_recommendation.csv'
    try:
        df=pd.read_csv(path) if path.exists() else _download_csv(CROP_DATA_URL,path)
    except Exception:
        df=_offline_crop_data()
    df.columns=[str(c).strip() for c in df.columns]
    required=['N','P','K','temperature','humidity','ph','rainfall','label']
    missing=[c for c in required if c not in df.columns]
    if missing: raise ValueError(f'Crop dataset missing columns: {missing}')
    for c in required[:-1]: df[c]=pd.to_numeric(df[c],errors='coerce')
    df['label']=df['label'].astype(str).str.strip().str.lower()
    return df.dropna(subset=required).reset_index(drop=True)


def load_soil_data() -> pd.DataFrame:
    path=DATA_DIR/'punjab_soil_data.csv'
    try:
        return pd.read_csv(path) if path.exists() else _download_csv(SOIL_DATA_URL,path)
    except Exception:
        return pd.DataFrame()
