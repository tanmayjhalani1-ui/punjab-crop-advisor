import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data_loader import load_crop_data
from src.model import train_model

df=load_crop_data(); _,m=train_model(df)
print(f'Rows: {len(df)} | Crops: {df.label.nunique()} | Accuracy: {m["accuracy"]:.4f} | F1: {m["f1_weighted"]:.4f}')
