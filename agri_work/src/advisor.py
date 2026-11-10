CROP_PROFILES={
'rice':((20,35),(150,300),'high'),'maize':((18,32),(60,120),'medium'),'wheat':((10,25),(50,100),'medium'),
'chickpea':((18,30),(50,100),'low'),'lentil':((18,30),(35,80),'low'),'cotton':((21,35),(60,150),'medium'),
'banana':((20,35),(100,250),'high'),'papaya':((21,35),(100,200),'medium'),'sugarcane':((20,35),(100,250),'high'),
'mungbean':((20,35),(50,100),'low'),'kidneybeans':((18,30),(60,120),'medium'),'pigeonpeas':((20,32),(60,140),'medium'),
'mothbeans':((24,35),(25,80),'low'),'blackgram':((20,35),(50,100),'low'),'pomegranate':((18,35),(50,120),'low'),
'mango':((24,35),(75,180),'medium'),'grapes':((15,32),(40,100),'medium'),'watermelon':((21,35),(40,100),'medium'),
'muskmelon':((22,35),(30,90),'medium'),'apple':((8,24),(80,180),'medium'),'orange':((15,32),(70,180),'medium'),
'coconut':((20,35),(100,300),'high'),'jute':((24,35),(120,250),'high'),'coffee':((18,30),(100,250),'high')}

def _fit(value,lo,hi,margin):
    if lo<=value<=hi:return 1.0
    return max(0.0,1.0-min(abs(value-lo),abs(value-hi))/margin)

def weather_fit(crop,temp,rain):
    p=CROP_PROFILES.get(crop)
    if not p:return .5
    return .55*_fit(temp,*p[0],15)+.45*_fit(rain,*p[1],150)

def analyze(predictions,weather=None,market=None):
    out=[]
    for crop,p in predictions:
        score=p; reasons=[]
        if weather:
            wf=weather_fit(crop,weather['temperature'],weather['rainfall_7d']); score*=.55+.45*wf
            reasons.append(f'7-day weather fit {wf*100:.0f}%')
        if market:
            reasons.append(f'market data available ({market.get("samples",0)} records)')
        out.append({'crop':crop,'model_probability':p,'advisory_score':score,'reasons':reasons})
    return sorted(out,key=lambda x:x['advisory_score'],reverse=True)
