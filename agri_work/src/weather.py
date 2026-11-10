import requests
from .config import OPEN_METEO_URL,GEOCODING_URL


def geocode(place):
    r=requests.get(GEOCODING_URL,params={'name':place,'count':1,'language':'en','format':'json'},timeout=12); r.raise_for_status()
    x=r.json().get('results',[])
    if not x: raise ValueError(f'Location not found: {place}')
    a=x[0]; return {'name':a.get('name',place),'country':a.get('country',''),'state':a.get('admin1',''),'lat':a['latitude'],'lon':a['longitude']}

def get_weather(lat,lon):
    params={'latitude':lat,'longitude':lon,'current':'temperature_2m,relative_humidity_2m,precipitation,rain,wind_speed_10m,weather_code','daily':'temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,wind_speed_10m_max,weather_code','timezone':'auto','forecast_days':7}
    r=requests.get(OPEN_METEO_URL,params=params,timeout=15); r.raise_for_status(); return r.json()

def summarize(weather):
    c=weather.get('current',{}); d=weather.get('daily',{})
    rain=sum(d.get('precipitation_sum',[])[:7])
    return {'temperature':float(c.get('temperature_2m',25)),'humidity':float(c.get('relative_humidity_2m',60)),'rain_now':float(c.get('rain',0)),'wind':float(c.get('wind_speed_10m',0)),'rainfall_7d':float(rain)}
