from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
MODEL_DIR = ROOT / 'models'
CACHE_DIR = ROOT / '.cache'

CROP_DATA_URL = os.getenv('CROP_DATA_URL', 'https://raw.githubusercontent.com/tanmayjhalani1-ui/punjab-crop-advisor/main/Crop_recommendation.csv')
SOIL_DATA_URL = os.getenv('SOIL_DATA_URL', 'https://raw.githubusercontent.com/tanmayjhalani1-ui/punjab-crop-advisor/main/punjab_soil_data.csv')
OPEN_METEO_URL = 'https://api.open-meteo.com/v1/forecast'
GEOCODING_URL = 'https://geocoding-api.open-meteo.com/v1/search'
DATA_GOV_RESOURCE = 'https://api.data.gov.in/resource'
MANDI_RESOURCE_ID = os.getenv('MANDI_RESOURCE_ID', '9ef84268-d588-465a-a308-a864a43d0070')
FEATURES = ['N','P','K','temperature','humidity','ph','rainfall']
DEFAULT_LOCATION = {'name':'Jaipur','state':'Rajasthan','lat':26.9124,'lon':75.7873}
