import os, requests, pandas as pd
from .config import DATA_GOV_RESOURCE,MANDI_RESOURCE_ID

def get_mandi_prices(state=None,commodity=None,limit=100):
    key=os.getenv('DATA_GOV_IN_KEY','').strip()
    if not key:return pd.DataFrame(), 'Live mandi prices require DATA_GOV_IN_KEY in Streamlit secrets/environment.'
    params={'api-key':key,'format':'json','limit':min(int(limit),1000)}
    if state:params['filters[state]']=state
    if commodity:params['filters[commodity]']=commodity
    try:
        r=requests.get(f'{DATA_GOV_RESOURCE}/{MANDI_RESOURCE_ID}',params=params,timeout=20); r.raise_for_status()
        records=r.json().get('records',[]); df=pd.DataFrame(records)
        if df.empty:return df,'No matching mandi records returned.'
        return df,'live'
    except Exception as e:return pd.DataFrame(),f'Mandi API unavailable: {e}'

def market_signal(df,commodity):
    if df.empty:return None
    cols={str(c).lower():c for c in df.columns}
    for wanted in ['modal_price','modal price','modalprice']:
        if wanted in cols:
            s=pd.to_numeric(df[cols[wanted]],errors='coerce').dropna()
            if len(s): return {'modal_mean':float(s.mean()),'min':float(s.min()),'max':float(s.max()),'samples':len(s)}
    return {'samples':len(df)}
