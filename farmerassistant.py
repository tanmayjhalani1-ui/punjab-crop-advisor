import os
import json
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pandas as pd
import requests
import warnings
import google.generativeai as genai
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pyttsx3
from deep_translator import GoogleTranslator

warnings.filterwarnings("ignore")

# --------- Utility for Button Hover Effect ---------
def add_hover_effect(btn, normal_bg, hover_bg, normal_fg="#222", hover_fg=None):
    def on_enter(e):
        btn['background'] = hover_bg
        if hover_fg:
            btn['foreground'] = hover_fg
    def on_leave(e):
        btn['background'] = normal_bg
        btn['foreground'] = normal_fg
    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)

# ---------------- Translation & Voice ----------------
voice_engine = pyttsx3.init()
voice_engine.setProperty('rate', 150)
voice_engine.setProperty('volume', 1.0)

def translate_and_speak(text, lang="English", speak=False):
    lang_map = {"English": "en", "Hindi": "hi", "Punjabi": "pa"}
    tgt_lang = lang_map.get(lang, "en")
    try:
        translated = GoogleTranslator(source='auto', target=tgt_lang).translate(text)
    except Exception:
        translated = text
    if speak:
        try:
            voice_engine.say(translated)
            voice_engine.runAndWait()
        except Exception:
            pass
    return translated

# ---------------- CONFIG ----------------


WEATHERAPI_KEY = "13c8857aeb724a20a8f113659251609"
GEMINI_API_KEY = "AIXXXXXXXXXXapi key4Cy2Zi_Q3T2aQQ"
GEMINI_MODEL = "gemini-2.5-flash-lite"
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".crop_chat_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_TTL_HOURS = 12

# ---------------- Gemini Chat Setup ----------------
genai.configure(api_key=GEMINI_API_KEY)
chat_session = None
try:
    gen_model = genai.GenerativeModel(GEMINI_MODEL)
    try:
        chat_session = gen_model.start_chat(history=[])
    except Exception:
        chat_session = None
except Exception:
    chat_session = None

# ---------------- Load & Train Crop Model ----------------
df = pd.read_csv("Crop_recommendation.csv")
c = df.label.astype('category')
targets = dict(enumerate(c.cat.categories))
df['target'] = c.cat.codes
y = df.target
X = df[['N','P','K','temperature','humidity','ph','rainfall']]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)
model_accuracy = accuracy_score(y_test, y_pred)

# ---------------- Load Soil Data ----------------
soil_df = pd.read_csv("punjab_soil_data.csv")
soil_df['District_norm'] = soil_df['District'].str.strip().str.lower()
districts = sorted(list(soil_df['District'].unique()))

# ---------------- Last 3-year price history ----------------
last_3yr_prices = {
    "wheat": {2021: 2100, 2022: 2200, 2023: 2350},
    "rice": {2021: 2400, 2022: 2500, 2023: 2600},
    "maize": {2021: 1800, 2022: 1900, 2023: 2000},
    "cotton": {2021: 5600, 2022: 6000, 2023: 6500},
    "sugarcane": {2021: 250, 2022: 280, 2023: 320},
    "bajra": {2021: 1600, 2022: 1750, 2023: 1900},
    "mustard": {2021: 4200, 2022: 4300, 2023: 4400},
    "groundnut": {2021: 5200, 2022: 5400, 2023: 5600},
    "orange": {2021: None, 2022: None, 2023: None},
    "muskmelon": {2021: None, 2022: None, 2023: None},
    "tomato": {2021: None, 2022: None, 2023: None}
}

# ---------------- Utility for Stylish GUI ----------------
DEFAULT_FRAME_BG = "#f5eee2"
HIGHLIGHT_FRAME_BG = "#dafbe1"

def style_labelframe(frame, highlighted=False):
    color = HIGHLIGHT_FRAME_BG if highlighted else DEFAULT_FRAME_BG
    frame.config(bg=color, highlightbackground="#cedaad", highlightthickness=2, bd=3)

def styled_feature_button(frame, emoji, text, color, status=None):
    text_lines = f"{emoji}\n{text}"
    if status:
        text_lines += f"\n[{status}]"
    btn = tk.Button(
        frame,
        text=text_lines,
        font=("Arial", 13, "bold"),
        bg=color,
        fg="#222",
        width=14,
        height=3,
        bd=0,
        relief="raised",
        highlightthickness=2,
        activebackground="#fafad2",
        cursor="hand2",
        compound="top"
    )
    # Set intelligent hover color (darker shade for active, lighter for beta/coming)
    if status == "Active":
        hover_bg = "#ffd700"
    elif status == "Beta":
        hover_bg = "#e3f7ff"
    elif status == "Coming":
        hover_bg = "#ffe6e6"
    else:
        hover_bg = "#e9e9e9"
    add_hover_effect(btn, color, hover_bg)
    btn.pack(side="left", padx=7, pady=10)
    return btn

def show_alert(message, level="info"):
    color = {"danger":"#ff5a5a", "warning":"#ffc107", "info":"#09c372"}.get(level, "#09c372")
    alert = tk.Toplevel(root)
    alert.geometry("350x60+600+200")
    alert.overrideredirect(True)
    alert.attributes("-topmost", True)
    tk.Label(alert, text=message, font=("Arial", 12, "bold"),
             bg=color, fg="white", padx=12, pady=10).pack(fill="both", expand=True)
    root.after(3000, alert.destroy)

# ---------------- Caching & Market helpers (unchanged) ----------------
def cache_path(name):
    safe = "".join(ch if ch.isalnum() or ch in ".-" else "" for ch in name)
    return os.path.join(CACHE_DIR, f"{safe}.json")

def is_cache_fresh(path):
    try:
        if not os.path.exists(path):
            return False
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        return datetime.now() - mtime < timedelta(hours=CACHE_TTL_HOURS)
    except Exception:
        return False

def save_cache(path, payload):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def load_cache(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def get_last3_prices_for_crop(crop_name):
    key = crop_name.strip().lower()
    return last_3yr_prices.get(key, None)

def format_last3_prices(price_dict):
    if not price_dict:
        return "Price data not available"
    parts = []
    for yr in sorted(price_dict.keys()):
        val = price_dict[yr]
        parts.append(f"{yr}: ₹{val}" if val else f"{yr}: N/A")
    return " | ".join(parts)

def scrape_agmarknet_modal(crop, district):
    try:
        url = "https://agmarknet.gov.in/PriceAndArrivals/CommodityDailyStateWise.aspx"
        params = {"comm": crop, "State": "Punjab"}
        r = requests.get(url, params=params, timeout=12)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        table = soup.find("table", {"id": "cphBody_GridDaily"})
        if not table:
            return None
        rows = table.find_all("tr")
        for tr in rows[1:]:
            cols = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(cols) >= 6:
                mandi = cols[0]
                modal = cols[5]
                if district.strip().lower() in mandi.strip().lower():
                    return modal
        for tr in rows[1:]:
            cols = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(cols) >= 6:
                return cols[5]
        return None
    except Exception:
        return None

# ---------------- Helper functions & Feature logic (unchanged) ----------------
def autofill_from_dropdown():
    d = district_var.get().strip().lower()
    if not d:
        messagebox.showinfo("Info", "Select a district.")
        return
    row = soil_df[soil_df['District_norm'] == d]
    if row.empty:
        messagebox.showinfo("Info", f"No soil data for '{district_var.get()}'.")
        return
    r = row.iloc[0]
    entry_N.delete(0, tk.END)
    entry_N.insert(0, str(r.get('N (kg/ha)', '')))
    entry_P.delete(0, tk.END)
    entry_P.insert(0, str(r.get('P (kg/ha)', '')))
    entry_K.delete(0, tk.END)
    entry_K.insert(0, str(r.get('K (kg/ha)', '')))
    entry_ph.delete(0, tk.END)
    entry_ph.insert(0, str(r.get('pH', '')))

def get_current_weather(city):
    try:
        url = f"http://api.weatherapi.com/v1/current.json?key={WEATHERAPI_KEY}&q={city}"
        resp = requests.get(url, timeout=8)
        data = resp.json()
        cur = data.get("current")
        if not cur:
            return None
        return float(cur.get("temp_c", 0)), float(cur.get("humidity", 0)), float(cur.get("precip_mm", 0))
    except Exception:
        return None

def get_7day_forecast(city):
    try:
        url = f"http://api.weatherapi.com/v1/forecast.json?key={WEATHERAPI_KEY}&q={city}&days=7&aqi=no&alerts=no"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if "forecast" not in data:
            return None
        out = []
        for d in data["forecast"]["forecastday"]:
            out.append({
                "date": d["date"],
                "min": d["day"]["mintemp_c"],
                "max": d["day"]["maxtemp_c"],
                "cond": d["day"]["condition"]["text"],
                "rain_mm": d["day"]["totalprecip_mm"]
            })
        return out
    except Exception:
        return None

def fetch_current_weather_button():
    city = city_entry.get().strip()
    if not city:
        messagebox.showinfo("Info", "Enter city to fetch weather.")
        return
    w = get_current_weather(city)
    if w:
        t, h, r = w
        entry_temp_var.set(f"{t:.1f}")
        entry_hum_var.set(f"{h:.1f}")
        entry_rain_var.set(f"{r:.1f}")
    else:
        messagebox.showwarning("Weather", "Could not fetch current weather for this city.")

def show_forecast_in_box():
    city = city_entry.get().strip()
    if not city:
        messagebox.showinfo("Info", "Enter nearest city/town for forecast.")
        return
    forecast = get_7day_forecast(city)
    if not forecast:
        messagebox.showerror("Forecast Error", f"Could not fetch forecast for '{city}'.")
        return
    forecast_box.config(state="normal")
    forecast_box.delete("1.0", tk.END)
    forecast_box.insert(tk.END, "📅 7-Day Forecast\n\n")
    for d in forecast:
        line = f"{d['date']}: {d['min']}°C - {d['max']}°C | {d['cond']} | Rain: {d['rain_mm']} mm\n"
        forecast_box.insert(tk.END, line)
        # Show alerts with toast
        if float(d['max']) > 40:
            show_alert(f"Extreme Heat: {d['date']} ({d['max']}°C)! Protect crop!", "danger")
        if float(d['min']) < 5:
            show_alert(f"Frost Alert: {d['date']} ({d['min']}°C)! Take action!", "warning")
        if float(d['rain_mm']) > 50:
            show_alert(f"Heavy Rain: {d['date']} ({d['rain_mm']}mm)", "info")
    forecast_box.config(state="disabled")
    root.last_forecast = forecast

def predict_and_show():
    progress = ttk.Progressbar(action_frame, orient="horizontal", mode="indeterminate", length=250)
    progress.pack(side="left", padx=10)
    root.update()
    progress.start()
    autofill_from_dropdown()
    try:
        N = float(entry_N.get())
        P = float(entry_P.get())
        K = float(entry_K.get())
        ph = float(entry_ph.get())
    except Exception:
        progress.stop()
        progress.pack_forget()
        messagebox.showerror("Input Error", "Enter valid soil values (N,P,K,pH).")
        return
    try:
        temp = float(entry_temp_var.get())
        hum = float(entry_hum_var.get())
        rain = float(entry_rain_var.get())
    except Exception:
        progress.stop()
        progress.pack_forget()
        messagebox.showerror("Weather Error", "Provide current weather or fetch from API.")
        return

    input_df = pd.DataFrame([[N, P, K, temp, hum, ph, rain]],
                            columns=['N','P','K','temperature','humidity','ph','rainfall'])
    input_scaled = scaler.transform(input_df)
    probs = clf.predict_proba(input_scaled)[0]
    top3_idx = probs.argsort()[-3:][::-1]
    top3 = [targets[i] for i in top3_idx]
    top3p = [probs[i]*100 for i in top3_idx]
    district_sel = district_var.get().strip()
    result_lines = []
    for i, crop in enumerate(top3):
        price_hist = get_last3_prices_for_crop(crop)
        if price_hist:
            price_text = format_last3_prices(price_hist)
        else:
            modal = scrape_agmarknet_modal(crop, district_sel)
            price_text = f"Modal (recent): ₹{modal}/quintal" if modal else "Price data not available"
        result_lines.append(f"{i+1}. {crop} ({top3p[i]:.2f}%)\n   Prices: {price_text}")
    # Translate & Voice
    txt_lines = [translate_and_speak("🌾 Top 3 Crops (with last 3 years where available):", language_var.get(), voice_enabled.get())]
    for line in result_lines:
        txt_lines.append(translate_and_speak(line, language_var.get(), voice_enabled.get()))
    txt_lines.append(translate_and_speak(f"\n🌡 Temp: {temp}°C | 💧 Humidity: {hum}% | 🌧 Rain: {rain} mm", language_var.get(), voice_enabled.get()))
    txt_lines.append(translate_and_speak(f"\n🔢 Model Accuracy: {model_accuracy*100:.2f}%", language_var.get(), voice_enabled.get()))
    txt = "\n\n".join(txt_lines)
    result_box.config(state="normal"); result_box.delete("1.0", tk.END); result_box.insert(tk.END, txt); result_box.config(state="disabled")
    root.last_prediction = {
        "crops": top3,
        "probs": top3p,
        "soil": {"N": N, "P": P, "K": K, "pH": ph},
        "weather": {"temp": temp, "humidity": hum, "rainfall": rain}
    }
    chat_button.config(state="normal")
    forecast_button.config(state="normal")
    progress.stop()
    progress.pack_forget()

def show_all_prices():
    district = district_var.get().strip()
    market_box.config(state="normal")
    market_box.delete("1.0", tk.END)
    market_box.insert(tk.END, f"🌾 Last 3 Year Prices (available data) — District: {district or 'N/A'}\n\n")
    for crop, hist in last_3yr_prices.items():
        if hist and any(v is not None for v in hist.values()):
            market_box.insert(tk.END, f"{crop.capitalize()}: {format_last3_prices(hist)}\n")
        else:
            market_box.insert(tk.END, f"{crop.capitalize()}: Price data not available\n")
    market_box.config(state="disabled")

def show_prices_for_top3():
    p = getattr(root, "last_prediction", None)
    if not p:
        messagebox.showinfo("Info", "Predict first to get top crops.")
        return
    district = district_var.get().strip()
    market_box.config(state="normal")
    market_box.delete("1.0", tk.END)
    market_box.insert(tk.END, f"🌾 Prices for Top 3 predicted crops — District: {district or 'N/A'}\n\n")
    for crop in p['crops']:
        hist = get_last3_prices_for_crop(crop)
        if hist:
            market_box.insert(tk.END, f"{crop}: {format_last3_prices(hist)}\n")
        else:
            modal = scrape_agmarknet_modal(crop, district)
            if modal:
                market_box.insert(tk.END, f"{crop}: Modal price (scraped): ₹{modal}/quintal\n")
            else:
                market_box.insert(tk.END, f"{crop}: Price data not available\n")
    market_box.config(state="disabled")

def open_chat_window():
    ctx = getattr(root, "last_prediction", None)
    forecast = getattr(root, "last_forecast", None)
    if ctx is None:
        messagebox.showinfo("Info", "Predict crop first before chatting.")
        return
    cw = tk.Toplevel(root)
    cw.title("AI Chat — Ask about the recommendation")
    cw.geometry("720x520")
    chat_log = scrolledtext.ScrolledText(cw, wrap=tk.WORD, state="disabled", font=("Arial", 11))
    chat_log.pack(padx=8, pady=8, fill="both", expand=True)
    entry_frame = tk.Frame(cw)
    entry_frame.pack(fill="x", padx=8, pady=6)
    user_entry = tk.Entry(entry_frame, font=("Arial", 12))
    user_entry.pack(side="left", fill="x", expand=True, padx=(0,6))
    send_btn = tk.Button(entry_frame, text="Send", bg="#1976d2", fg="white", command=lambda: send_message())
    send_btn.pack(side="right")
    add_hover_effect(send_btn,"#1976d2","#0d47a1","white") # Hover added for chat window send button
    def append(text, who="You"):
        chat_log.config(state="normal")
        chat_log.insert(tk.END, f"{who}: {text}\n\n")
        chat_log.see(tk.END)
        chat_log.config(state="disabled")
        if who=="AI" and voice_enabled.get():
            try:
                voice_engine.say(text)
                voice_engine.runAndWait()
            except Exception:
                pass
    def send_message():
        q = user_entry.get().strip()
        if not q:
            return
        append(q, who="Farmer")
        user_entry.delete(0, tk.END)
        thinking_text = translate_and_speak("Thinking...", language_var.get(), voice_enabled.get())
        append(thinking_text, who="AI")
        p = root.last_prediction
        context_lines = [
            f"Top crops: {', '.join(p['crops'])} with probabilities {', '.join([f'{x:.1f}%' for x in p['probs']])}.",
            f"Soil: N={p['soil']['N']}, P={p['soil']['P']}, K={p['soil']['K']}, pH={p['soil']['pH']}.",
            f"Weather: temp={p['weather']['temp']}C, humidity={p['weather']['humidity']}%, rainfall={p['weather']['rainfall']}mm."
        ]
        if forecast:
            forecast_str = "; ".join([f"{d['date']}: {d['min']}-{d['max']}°C, {d['cond']}" for d in forecast])
            context_lines.append(f"7-day forecast: {forecast_str}")
        context = "\n".join(context_lines)
        prompt = f"You are an agricultural assistant for Punjab farmers. Use the context and answer simply.\n\nContext:\n{context}\nFarmer question: {q}\nAnswer:"
        bot_ans = "Sorry, chatbot unavailable."
        try:
            if chat_session is not None:
                resp = chat_session.send_message(prompt)
                bot_ans = getattr(resp, "text", str(resp))
            else:
                bot_ans = "Chat model not initialized."
        except Exception as e:
            bot_ans = f"Chat error: {e}"
        bot_ans_translated = translate_and_speak(bot_ans, language_var.get(), voice_enabled.get())
        append(bot_ans_translated, who="AI")
    user_entry.bind("<Return>", lambda ev: send_message())

# ---------------- Main Window GUI ----------------
root = tk.Tk()
root.title("Punjab Crop Advisor – SIH Edition")
root.geometry("1100x850")
root.configure(bg="#f3efe6")
try:
    root.iconbitmap("farmer_hat.ico")
except Exception:
    pass

# ---------------- Brand Hero ----------------
hero = tk.Frame(root, bg="#3e674f", pady=18)
tk.Label(hero, text="🚜 PUNJAB CROP ADVISOR", fg="white", bg="#3e674f",
         font=("Georgia", 32, "bold")).pack()
tk.Label(hero, text="AI and ML Powered Farming (SIH 25010)",
         fg="#ffe246", bg="#3e674f", font=("Georgia", 16, "italic")).pack()
hero.pack(fill="x")

# ---------------- Scrollable Frame Setup ----------------
canvas = tk.Canvas(root, bg="#f3efe6")
scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas, bg="#f3efe6")
scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

top_frame = tk.Frame(scrollable_frame, padx=10, pady=10, bg=DEFAULT_FRAME_BG)
top_frame.pack(fill="x")
style_labelframe(top_frame)

soil_frame = tk.LabelFrame(scrollable_frame, text="Soil Input", padx=10, pady=10)
soil_frame.pack(fill="x", padx=10, pady=5)
style_labelframe(soil_frame)

weather_frame = tk.LabelFrame(scrollable_frame, text="Weather Input / Fetch", padx=10, pady=10)
weather_frame.pack(fill="x", padx=10, pady=5)
style_labelframe(weather_frame)

action_frame = tk.Frame(scrollable_frame, padx=10, pady=10, bg=DEFAULT_FRAME_BG)
action_frame.pack(fill="x")
style_labelframe(action_frame)

result_frame = tk.LabelFrame(scrollable_frame, text="Prediction Result", padx=10, pady=10)
result_frame.pack(fill="both", expand=True, padx=10, pady=5)
style_labelframe(result_frame, highlighted=True)

forecast_frame = tk.LabelFrame(scrollable_frame, text="7-Day Forecast", padx=10, pady=10)
forecast_frame.pack(fill="both", expand=True, padx=10, pady=5)
style_labelframe(forecast_frame, highlighted=True)

market_frame = tk.LabelFrame(scrollable_frame, text="Market Prices", padx=10, pady=10)
market_frame.pack(fill="both", expand=True, padx=10, pady=5)
style_labelframe(market_frame)

# District, city, autofill
tk.Label(top_frame, text="District:", font=("Arial", 12), bg=DEFAULT_FRAME_BG).pack(side="left")
district_var = tk.StringVar()
district_menu = ttk.Combobox(top_frame, values=districts, textvariable=district_var, width=20)
district_menu.pack(side="left", padx=5)

autofill_button = tk.Button(top_frame, text="Autofill Soil", command=autofill_from_dropdown, bg="#e9ffd0", font=("Arial", 11, "bold"))
autofill_button.pack(side="left", padx=5)
add_hover_effect(autofill_button, "#e9ffd0", "#caff70")

tk.Label(top_frame, text="Nearest City:", font=("Arial", 12), bg=DEFAULT_FRAME_BG).pack(side="left", padx=10)
city_var = tk.StringVar()
city_entry = tk.Entry(top_frame, textvariable=city_var, font=("Arial", 11), width=18)
city_entry.pack(side="left", padx=5)

fetch_weather_button = tk.Button(top_frame, text="Fetch Weather", command=fetch_current_weather_button, bg="#d7eafd", font=("Arial", 11, "bold"))
fetch_weather_button.pack(side="left", padx=5)
add_hover_effect(fetch_weather_button, "#d7eafd", "#53b2ea")

# ---------------- Soil Entries ----------------
tk.Label(soil_frame, text="N (kg/ha):", font=("Arial", 11), bg=DEFAULT_FRAME_BG).grid(row=0, column=0, sticky="w")
entry_N = tk.Entry(soil_frame, font=("Arial", 11))
entry_N.grid(row=0, column=1, padx=5, pady=2)
tk.Label(soil_frame, text="P (kg/ha):", font=("Arial", 11), bg=DEFAULT_FRAME_BG).grid(row=0, column=2, sticky="w")
entry_P = tk.Entry(soil_frame, font=("Arial", 11))
entry_P.grid(row=0, column=3, padx=5, pady=2)
tk.Label(soil_frame, text="K (kg/ha):", font=("Arial", 11), bg=DEFAULT_FRAME_BG).grid(row=1, column=0, sticky="w")
entry_K = tk.Entry(soil_frame, font=("Arial", 11))
entry_K.grid(row=1, column=1, padx=5, pady=2)
tk.Label(soil_frame, text="pH:", font=("Arial", 11), bg=DEFAULT_FRAME_BG).grid(row=1, column=2, sticky="w")
entry_ph = tk.Entry(soil_frame, font=("Arial", 11))
entry_ph.grid(row=1, column=3, padx=5, pady=2)

# ---------------- Weather Entries ----------------
entry_temp_var = tk.StringVar()
entry_hum_var = tk.StringVar()
entry_rain_var = tk.StringVar()
tk.Label(weather_frame, text="Temp (°C):", font=("Arial", 11), bg=DEFAULT_FRAME_BG).grid(row=0, column=0, sticky="w")
tk.Entry(weather_frame, textvariable=entry_temp_var, font=("Arial", 11)).grid(row=0, column=1, padx=5, pady=2)
tk.Label(weather_frame, text="Humidity (%):", font=("Arial", 11), bg=DEFAULT_FRAME_BG).grid(row=0, column=2, sticky="w")
tk.Entry(weather_frame, textvariable=entry_hum_var, font=("Arial", 11)).grid(row=0, column=3, padx=5, pady=2)
tk.Label(weather_frame, text="Rainfall (mm):", font=("Arial", 11), bg=DEFAULT_FRAME_BG).grid(row=1, column=0, sticky="w")
tk.Entry(weather_frame, textvariable=entry_rain_var, font=("Arial", 11)).grid(row=1, column=1, padx=5, pady=2)

# ---------------- Language & Voice ----------------
language_var = tk.StringVar(value="English")
voice_enabled = tk.BooleanVar(value=True)
tk.Label(action_frame, text="Language:", font=("Arial", 12), bg=DEFAULT_FRAME_BG).pack(side="left")
lang_menu = ttk.Combobox(action_frame, values=["English", "Hindi", "Punjabi"], textvariable=language_var, width=10)
lang_menu.pack(side="left", padx=5)
tk.Checkbutton(action_frame, text="Voice Enabled", variable=voice_enabled, bg=DEFAULT_FRAME_BG).pack(side="left", padx=10)

# ---------------- Action Buttons ----------------
predict_button = tk.Button(action_frame, text="Predict Crop", bg="#4caf50", fg="white", font=("Arial", 12),
                           command=predict_and_show)
predict_button.pack(side="left", padx=5)
add_hover_effect(predict_button, "#4caf50", "#388e3c", normal_fg="white")
forecast_button = tk.Button(action_frame, text="Show Forecast", command=show_forecast_in_box, bg="#0072e6", fg="white", font=("Arial", 12), state="disabled")
forecast_button.pack(side="left", padx=5)
add_hover_effect(forecast_button, "#0072e6", "#005bb5", normal_fg="white")
chat_button = tk.Button(action_frame, text="Chat with AI", command=open_chat_window, bg="#6f42c1", fg='white', font=("Arial", 12), state="disabled")
chat_button.pack(side="left", padx=5)
add_hover_effect(chat_button, "#6f42c1", "#442183", normal_fg="white")
all_prices_button = tk.Button(action_frame, text="Show All Prices", command=show_all_prices, bg="#ffa740", fg="white", font=("Arial", 12))
all_prices_button.pack(side="left", padx=5)
add_hover_effect(all_prices_button, "#ffa740", "#ff9100", normal_fg="white")
top3_prices_button = tk.Button(action_frame, text="Show Top 3 Prices", command=show_prices_for_top3, bg="#f44269", fg="white", font=("Arial", 12))
top3_prices_button.pack(side="left", padx=5)
add_hover_effect(top3_prices_button, "#f44269", "#b81b38", normal_fg="white")

# ---------------- Result / Forecast / Market Boxes ----------------
result_box = scrolledtext.ScrolledText(result_frame, height=8, state="disabled", font=("Consolas", 12))
result_box.pack(fill="both", expand=True)
result_box.configure(bg="#f8fbe6", fg="#1b2a00")

forecast_box = scrolledtext.ScrolledText(forecast_frame, height=6, state="disabled", font=("Consolas", 12, "bold"))
forecast_box.pack(fill="both", expand=True)
forecast_box.configure(bg="#e9f8fc", fg="#223a4a")

market_box = scrolledtext.ScrolledText(market_frame, height=6, state="disabled", font=("Arial", 11, "italic"))
market_box.pack(fill="both", expand=True)
market_box.configure(bg="#fff0e9", fg="#5b3525")

# -------------- Advanced Features Section as Cards --------------
dummy_frame = tk.LabelFrame(scrollable_frame, text="Professional Features", padx=10, pady=10)
dummy_frame.pack(fill="x", expand=True, padx=10, pady=10)
style_labelframe(dummy_frame)
styled_feature_button(dummy_frame, "📷", "Pest Detector", "#ffe3b3", "Active")
styled_feature_button(dummy_frame, "🌐", "Punjabi\nTranslator", "#e0edfc", "Active")
styled_feature_button(dummy_frame, "🟢", "WhatsApp Bot", "#cdf4d9", "Active")
styled_feature_button(dummy_frame, "📜", "Govt Schemes", "#e0ebff", "Beta")
styled_feature_button(dummy_frame, "🗺", "Smart Roadmap", "#fff6cc", "Beta")
styled_feature_button(dummy_frame, "👥", "Community Hub", "#fbe9e7", "Coming")
styled_feature_button(dummy_frame, "🧑‍🌾", "Expert Connect", "#ffecb3", "Coming")
styled_feature_button(dummy_frame, "🚨", "Early Warning", "#ffd6d6", "Active")

# -------------- Footer ------------
footer = tk.Frame(root, bg="#3e674f", height=30)
footer.pack(fill="x", side="bottom")
root.mainloop()