import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data_loader import _offline_crop_data
from src.model import train_model,predict_top
from src.advisor import analyze

def test_training_and_prediction():
    df=_offline_crop_data(); model,metrics=train_model(df)
    assert metrics['accuracy']>=0
    values={c:float(df[c].iloc[0]) for c in ['N','P','K','temperature','humidity','ph','rainfall']}
    p=predict_top(model,values,5)
    assert len(p)==5 and p[0][1]>=p[-1][1]
    a=analyze(p,{'temperature':25,'humidity':60,'rain_now':0,'wind':5,'rainfall_7d':100})
    assert len(a)==5 and a[0]['advisory_score']>=0
