import sys, os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from src.config import DEFAULT_LOCATION
from src.data_loader import load_crop_data,load_soil_data
from src.model import load_or_train,predict_top
from src.weather import get_weather,geocode,summarize
from src.market import get_mandi_prices,market_signal
from src.advisor import analyze

load_dotenv()
st.set_page_config(page_title='AgriPulse AI',page_icon='🌾',layout='wide',initial_sidebar_state='expanded')
st.markdown('''<style>
.block-container{max-width:1500px;padding-top:1rem}.hero{padding:2rem 2.2rem;border-radius:26px;background:linear-gradient(135deg,#123b2a,#1f7a4d);color:white;margin-bottom:1.2rem}.hero h1{margin:0;font-size:2.5rem}.hero p{margin:.45rem 0 0;opacity:.88}.card{padding:1rem;border:1px solid #e5ebe7;border-radius:18px;background:#fff}.muted{color:#667085}.stMetric{border:1px solid #e5ebe7;padding:.5rem;border-radius:14px}.risk{padding:.8rem 1rem;border-radius:14px;background:#fff7ed;border:1px solid #fed7aa;margin:.4rem 0}
</style>''',unsafe_allow_html=True)

@st.cache_data(ttl=3600,show_spinner=False)
def crop_data(): return load_crop_data()
@st.cache_data(ttl=3600,show_spinner=False)
def soil_data(): return load_soil_data()
@st.cache_resource(show_spinner=False)
def model_bundle(): return load_or_train(crop_data())

st.markdown('<div class="hero"><h1>🌾 AgriPulse AI</h1><p>AI crop recommendation + live weather + mandi intelligence + explainable farm advisory</p></div>',unsafe_allow_html=True)

with st.sidebar:
    st.header('🚜 Farm profile')
    place=st.text_input('Village / city / district','Jaipur')
    state=st.selectbox('State',['Rajasthan','Punjab','Haryana','Uttar Pradesh','Madhya Pradesh','Gujarat','Maharashtra','Other'])
    st.divider(); st.subheader('Soil & climate')
    n=st.number_input('Nitrogen (N)',0.0,200.0,80.0)
    p=st.number_input('Phosphorus (P)',0.0,200.0,45.0)
    k=st.number_input('Potassium (K)',0.0,200.0,40.0)
    ph=st.number_input('Soil pH',3.0,10.0,6.5,step=.1)
    temp=st.number_input('Temperature °C',-5.0,55.0,25.0)
    humidity=st.number_input('Humidity %',0.0,100.0,60.0)
    rain=st.number_input('Expected rainfall mm',0.0,1000.0,100.0)
    live=st.checkbox('Use live weather for advisory',True)
    run=st.button('🌱 Analyze my farm',type='primary',use_container_width=True)

values={'N':n,'P':p,'K':k,'temperature':temp,'humidity':humidity,'ph':ph,'rainfall':rain}
mdl,meta=model_bundle()

if run or 'analysis' not in st.session_state:
    with st.spinner('Running crop model and live-data checks...'):
        weather=None; weather_raw=None; location=None; wx_msg='Manual climate values'
        if live:
            try:
                location=geocode(place); weather_raw=get_weather(location['lat'],location['lon']); weather=summarize(weather_raw); wx_msg='Live Open-Meteo weather'
                values['temperature']=weather['temperature']; values['humidity']=weather['humidity']; values['rainfall']=weather['rainfall_7d']
            except Exception as e: wx_msg=f'Live weather unavailable; using entered values ({e})'
        predictions=predict_top(mdl,values,7)
        st.session_state.analysis={'predictions':predictions,'values':values.copy(),'weather':weather,'weather_raw':weather_raw,'location':location,'wx_msg':wx_msg}

A=st.session_state.analysis
pred=A['predictions']; weather=A['weather']; loc=A['location']

# Mandi lookup is deliberately done after crop prediction, so the default commodity is meaningful.
market_df=pd.DataFrame(); market_msg='Not fetched yet'; market_info=None
with st.expander('📈 Fetch live mandi prices for the recommended crop',expanded=False):
    mc1,mc2=st.columns([1,2]); ms=mc1.text_input('Mandi state',state); commodity=mc2.text_input('Commodity',pred[0][0].title())
    if st.button('Fetch live government mandi data'):
        market_df,market_msg=get_mandi_prices(ms,commodity,100)
        if not market_df.empty: market_info=market_signal(market_df,commodity)
        st.session_state.market=(market_df,market_msg,market_info)
if 'market' in st.session_state: market_df,market_msg,market_info=st.session_state.market

ranked=analyze(pred,weather,market_info)

c1,c2,c3,c4=st.columns(4)
c1.metric('Recommended crop',ranked[0]['crop'].title())
c2.metric('Model probability',f"{ranked[0]['model_probability']*100:.1f}%")
c3.metric('Advisory score',f"{ranked[0]['advisory_score']*100:.1f}%")
c4.metric('Model F1',f"{meta.get('metrics',{}).get('f1_weighted',0)*100:.1f}%")

st.info(f"📍 {loc['name']+', '+loc.get('state','') if loc else 'Manual location'} · {A['wx_msg']}")

t1,t2,t3,t4=st.tabs(['🌾 Recommendation','🌦️ Weather','📈 Market','🧪 Soil & Model'])
with t1:
    df=pd.DataFrame(ranked); df['Model probability %']=df['model_probability']*100; df['Advisory score %']=df['advisory_score']*100
    st.plotly_chart(px.bar(df.sort_values('Advisory score %'),x='Advisory score %',y='crop',orientation='h',text_auto='.1f',title='Top crops after weather/market context'),use_container_width=True)
    st.subheader('Why this crop?')
    for item in ranked[:3]:
        st.markdown(f"**{item['crop'].title()}** — model {item['model_probability']*100:.1f}% · advisory {item['advisory_score']*100:.1f}%")
        if item['reasons']: st.caption(' • '.join(item['reasons']))
    st.warning('Decision support only: verify seed suitability, irrigation, soil test and local agricultural advisories before planting.')
with t2:
    if A['weather_raw']:
        c=A['weather_raw'].get('current',{}); d=A['weather_raw'].get('daily',{})
        w1,w2,w3,w4=st.columns(4); w1.metric('Temperature',f"{c.get('temperature_2m','—')} °C"); w2.metric('Humidity',f"{c.get('relative_humidity_2m','—')}%"); w3.metric('Rain now',f"{c.get('rain','—')} mm"); w4.metric('Wind',f"{c.get('wind_speed_10m','—')} km/h")
        forecast=pd.DataFrame({'Date':d.get('time',[]),'Rain mm':d.get('precipitation_sum',[]),'Rain probability %':d.get('precipitation_probability_max',[]),'Max °C':d.get('temperature_2m_max',[])})
        st.plotly_chart(px.line(forecast,x='Date',y=['Rain mm','Rain probability %'],markers=True,title='7-day weather outlook'),use_container_width=True)
    else: st.info(A['wx_msg'])
with t3:
    if not market_df.empty:
        st.success(f'Government OGD data received: {len(market_df)} records.')
        st.dataframe(market_df,use_container_width=True,hide_index=True)
        if market_info and 'modal_mean' in market_info: st.metric('Average modal price in returned sample',f"₹{market_info['modal_mean']:,.0f}")
    else: st.info(market_msg)
with t4:
    st.subheader('Entered / live feature values'); st.dataframe(pd.DataFrame([A['values']]),use_container_width=True,hide_index=True)
    soil=soil_data()
    if not soil.empty:
        st.subheader('Reference soil data'); st.dataframe(soil.head(20),use_container_width=True,hide_index=True)
    st.subheader('Model diagnostics'); st.json(meta)

st.divider(); st.caption('Data layers: reference crop dataset from the Punjab Crop Advisor project; weather via Open-Meteo; mandi prices via India OGD/data.gov.in when a valid key is configured. IMD integration can be added for authenticated production deployments.')
