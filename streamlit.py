import asyncio
import io
import json
import os
from pathlib import Path
import streamlit as st
import requests
from groq import Groq

WEATHERAPI_KEY = st.secrets["WEATHERAPI_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
import numpy as np
import pandas as pd
import requests
import streamlit as st
#from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

BASE = Path(__file__).resolve().parent
CROP_CSV = BASE / "Crop_recommendation.csv"
SOIL_CSV = BASE / "punjab_soil_data.csv"
LEAF_MODEL = BASE / "leaf_disease.keras"
LEAF_CLASSES = BASE / "leaf_classes.json"

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
LANGUAGES = {"English": "en", "Hindi": "hi", "Punjabi": "pa"}

st.set_page_config(
    page_title="Punjab Crop Advisor",
    page_icon="🌾",
    layout="wide",
)


# -------------------------- Configuration --------------------------

def setting(name, default=""):
    value = os.getenv(name)

    if value:
        return value

    try:
        return st.secrets.get(name, default)
    except (FileNotFoundError, KeyError):
        return default


WEATHERAPI_KEY = setting("WEATHERAPI_KEY")
GROQ_API_KEY = setting("GROQ_API_KEY")
GROQ_MODEL = setting("GROQ_MODEL", "openai/gpt-oss-20b")


# ----------------------- Translation and voice ---------------------

def translate(text, language):
    if language == "English":
        return text

    from deep_translator import GoogleTranslator

    translator = GoogleTranslator(
        source="auto",
        target=LANGUAGES[language],
    )

    # Keep individual requests reasonably small.
    chunks = [text[i:i + 3000] for i in range(0, len(text), 3000)]
    return "\n".join(translator.translate(chunk) for chunk in chunks)


async def make_speech_async(text, language):
    import edge_tts

    locale_prefix = {
        "English": "en-IN",
        "Hindi": "hi-IN",
        "Punjabi": "pa-IN",
    }[language]

    voices = await edge_tts.list_voices()
    matching = [
        voice["ShortName"]
        for voice in voices
        if voice["ShortName"].startswith(locale_prefix + "-")
    ]

    if not matching:
        raise RuntimeError(
            f"No available Edge TTS voice found for {language}."
        )

    audio = bytearray()
    speech = edge_tts.Communicate(text[:4000], matching[0])

    async for chunk in speech.stream():
        if chunk["type"] == "audio":
            audio.extend(chunk["data"])

    if not audio:
        raise RuntimeError("Voice service returned no audio.")

    return bytes(audio)


def speak_button(text, language, key):
    if st.button("🔊 Play audio", key=key):
        try:
            audio = asyncio.run(make_speech_async(text, language))
            st.audio(audio, format="audio/mp3")
        except Exception as exc:
            st.error(f"Voice unavailable: {exc}")


# ---------------------- Original crop model ------------------------

@st.cache_resource(show_spinner="Training crop recommendation model...")
def load_crop_model(file_path, modified_time):
    df = pd.read_csv(file_path)

    missing = set(FEATURES + ["label"]) - set(df.columns)
    if missing:
        raise ValueError(f"Crop CSV missing columns: {sorted(missing)}")

    df = df.dropna(subset=FEATURES + ["label"]).copy()

    for column in FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=FEATURES + ["label"])

    if len(df) < 10 or df["label"].nunique() < 2:
        raise ValueError("Crop CSV needs at least 10 rows and two crop classes.")

    categories = df["label"].astype("category")
    targets = dict(enumerate(categories.cat.categories))

    X = df[FEATURES]
    y = categories.cat.codes

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        random_state=1,
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )
    classifier.fit(X_train_scaled, y_train)

    accuracy = accuracy_score(
        y_test,
        classifier.predict(X_test_scaled),
    )

    return classifier, scaler, targets, accuracy


@st.cache_data(show_spinner=False)
def load_soil_data(file_path, modified_time):
    df = pd.read_csv(file_path)

    required = {"District", "N (kg/ha)", "P (kg/ha)", "K (kg/ha)", "pH"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Soil CSV missing columns: {sorted(missing)}")

    df["District"] = df["District"].astype(str).str.strip()
    return df


# ---------------------------- Weather -----------------------------

@st.cache_data(ttl=900, show_spinner=False)

def fetch_weather(city, kind, api_key=WEATHERAPI_KEY):
    params = {
        "key": api_key,
        "q": city
    }

    if kind == "forecast":
        params.update({
            "days": 7,
            "aqi": "no",
            "alerts": "no"
        })

    response = requests.get(
        f"https://api.weatherapi.com/v1/{kind}.json",
        params=params,
        timeout=15,
    )

    if not response.ok:
        try:
            reason = response.json()["error"]["message"]
        except (ValueError, KeyError, TypeError):
            reason = f"HTTP {response.status_code}"
        raise RuntimeError(reason)

    return response.json()



def ask_gemini(question, context, history):
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    messages = [
        {
            "role": "system",
            "content": (
                "You are an agricultural assistant for Punjab farmers. "
                "Answer simply using the provided context. "
                "Do not invent market prices or claim a photo prediction "
                "is a confirmed diagnosis. Context:\n"
                + context
            ),
        }
    ]

    for message in history[-8:]:
        messages.append({
            "role": message["role"],
            "content": message["text"],
        })

    messages.append({
        "role": "user",
        "content": question,
    })

    payload = {
        "model": "openai/gpt-oss-20b",
        "messages": messages,
        "temperature": 0.3,
    }

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=45,
    )

    if not response.ok:
        raise RuntimeError(
            f"Groq API error: HTTP {response.status_code}"
        )

    data = response.json()

    return data["choices"][0]["message"]["content"]

# ----------------- Your leaf-disease CNN architecture -------------

def build_leaf_model(number_of_classes):
    """
    Same Conv2D / MaxPooling2D / Flatten / Dense architecture
    as the notebook you supplied.
    """
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(256, 256, 3)),

        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", padding="same"
        ),
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", padding="same"
        ),
        tf.keras.layers.MaxPooling2D(3, 3),

        tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", padding="same"
        ),
        tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", padding="same"
        ),
        tf.keras.layers.MaxPooling2D(3, 3),

        tf.keras.layers.Conv2D(
            128, (3, 3), activation="relu", padding="same"
        ),
        tf.keras.layers.Conv2D(
            128, (3, 3), activation="relu", padding="same"
        ),
        tf.keras.layers.MaxPooling2D(3, 3),

        tf.keras.layers.Conv2D(
            256, (3, 3), activation="relu", padding="same"
        ),
        tf.keras.layers.Conv2D(
            256, (3, 3), activation="relu", padding="same"
        ),

        tf.keras.layers.Conv2D(
            512, (5, 5), activation="relu", padding="same"
        ),
        tf.keras.layers.Conv2D(
            512, (5, 5), activation="relu", padding="same"
        ),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1568, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(
            number_of_classes,
            activation="softmax",
        ),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def find_leaf_folders(root):
    """Locate the dataset's sibling train/valid folders."""
    root = Path(root)

    for candidate in [root, *root.rglob("train")]:
        train_dir = candidate if candidate.name == "train" else candidate / "train"
        valid_dir = train_dir.parent / "valid"

        if train_dir.is_dir() and valid_dir.is_dir():
            return train_dir, valid_dir

    raise FileNotFoundError(
        "Downloaded dataset has no matching train/valid folders."
    )


def train_leaf_model(epochs):
    import kagglehub
    import tensorflow as tf

    tf.keras.utils.set_random_seed(0)

    # Downloads to KaggleHub's cache; existing downloads are reused.
    dataset_root = kagglehub.dataset_download(
        "vipoooool/new-plant-diseases-dataset"
    )

    train_dir, valid_dir = find_leaf_folders(dataset_root)

    train_data = tf.keras.utils.image_dataset_from_directory(
        str(train_dir),
        image_size=(256, 256),
        batch_size=8,
        shuffle=True,
        seed=0,
    )

    # Save class order BEFORE .map(), which removes class_names.
    class_names = list(train_data.class_names)

    if len(class_names) != 38:
        raise ValueError(
            f"Expected the notebook's 38 disease classes; "
            f"found {len(class_names)}."
        )

    valid_data = tf.keras.utils.image_dataset_from_directory(
        str(valid_dir),
        image_size=(256, 256),
        batch_size=8,
        shuffle=False,
        class_names=class_names,
    )

    rescale = tf.keras.layers.Rescaling(1.0 / 255)

    train_data = train_data.map(
        lambda image, label: (rescale(image), label)
    ).prefetch(tf.data.AUTOTUNE)

    valid_data = valid_data.map(
        lambda image, label: (rescale(image), label)
    ).prefetch(tf.data.AUTOTUNE)

    model = build_leaf_model(len(class_names))

    # fit_generator() from the notebook is replaced with fit();
    # the datasets and training objective remain the same.
    history = model.fit(
        train_data,
        validation_data=valid_data,
        epochs=epochs,
    )

    # Only write artifacts after successful training.
    model.save(LEAF_MODEL)
    LEAF_CLASSES.write_text(
        json.dumps(class_names, indent=2),
        encoding="utf-8",
    )

    return history.history


@st.cache_resource(show_spinner="Loading trained leaf model...")
def load_leaf_model(model_path, modified_time):
    import tensorflow as tf

    return tf.keras.models.load_model(
        model_path,
        compile=False,
    )


# The same simplified mapping supplied in your notebook.
# These are example treatment names, NOT verified local prescriptions.
DISEASE_TO_PESTICIDE = {
    "Apple___Apple_scab": "Captan, Mancozeb",
    "Apple___Black_rot": "Ziram, Captan",
    "Apple___Cedar_apple_rust": "Myclobutanil",
    "Apple___healthy": "No pesticide needed",
    "Blueberry___healthy": "No pesticide needed",
    "Cherry_(including_sour)___Powdery_mildew": "Sulfur, Myclobutanil",
    "Cherry_(including_sour)___healthy": "No pesticide needed",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Strobilurins",
    "Corn_(maize)___Common_rust_": "Propiconazole",
    "Corn_(maize)___Northern_Leaf_Blight": "Azoxystrobin",
    "Corn_(maize)___healthy": "No pesticide needed",
    "Grape___Black_rot": "Mancozeb, Captan",
    "Grape___Esca_(Black_Measles)": "No effective chemical control",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Mancozeb",
    "Grape___healthy": "No pesticide needed",
    "Orange___Haunglongbing_(Citrus_greening)": (
        "Vector control: Imidacloprid"
    ),
    "Peach___Bacterial_spot": "Copper-based sprays",
    "Peach___healthy": "No pesticide needed",
    "Pepper,_bell___Bacterial_spot": "Copper-based sprays",
    "Pepper,_bell___healthy": "No pesticide needed",
    "Potato___Early_blight": "Chlorothalonil, Mancozeb",
    "Potato___Late_blight": "Metalaxyl, Mancozeb",
    "Potato___healthy": "No pesticide needed",
    "Raspberry___healthy": "No pesticide needed",
    "Soybean___healthy": "No pesticide needed",
    "Squash___Powdery_mildew": "Sulfur, Neem oil",
    "Strawberry___Leaf_scorch": "Captan",
    "Strawberry___healthy": "No pesticide needed",
    "Tomato___Bacterial_spot": "Copper sprays",
    "Tomato___Early_blight": "Chlorothalonil, Mancozeb",
    "Tomato___Late_blight": "Metalaxyl",
    "Tomato___Leaf_Mold": "Copper-based fungicides",
    "Tomato___Septoria_leaf_spot": "Chlorothalonil",
    "Tomato___Spider_mites Two-spotted_spider_mite": (
        "Insecticidal soap, Neem oil"
    ),
    "Tomato___Target_Spot": "Chlorothalonil",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": (
        "Vector control: Imidacloprid"
    ),
    "Tomato___Tomato_mosaic_virus": (
        "Sanitation, resistant varieties"
    ),
    "Tomato___healthy": "No pesticide needed",
}


# ------------------------------- UI -------------------------------

st.markdown(
    """
    <style>
    .block-container {max-width: 1200px; padding-top: 1.5rem;}
    .hero {
        background: linear-gradient(110deg, #173f2c, #438b54);
        color: white;
        border-radius: 18px;
        padding: 28px;
        margin-bottom: 18px;
    }
    .hero h1 {color: white; margin: 0;}
    .hero p {margin: 8px 0 0;}
    </style>
    <div class="hero">
      <h1>🌾 Punjab Crop Advisor</h1>
      <p>Crop recommendation · Weather · Leaf disease · Nearby mandi price</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    language = st.selectbox(
        "Language",
        ["English", "Hindi", "Punjabi"],
    )
    voice_enabled = st.checkbox("Voice enabled", value=False)

tabs = st.tabs([
    "🌱 Predict Crop",
    "🌤️ Weather & Alerts",
    "🍃 Leaf Disease",
    "🏪 Nearby Mandi Price",
    "💬 Chat with AI",
])


# -------------------------- Crop prediction ------------------------

with tabs[0]:
    st.subheader("Crop recommendation")

    soil_df = None

    if SOIL_CSV.is_file():
        try:
            soil_df = load_soil_data(
                str(SOIL_CSV),
                SOIL_CSV.stat().st_mtime_ns,
            )
        except Exception as exc:
            st.warning(f"District autofill unavailable: {exc}")

    districts = (
        sorted(soil_df["District"].dropna().unique().tolist())
        if soil_df is not None
        else []
    )

    district = st.selectbox(
        "District",
        ["Enter values manually"] + districts,
    )

    defaults = {
        "input_N": 50.0,
        "input_P": 50.0,
        "input_K": 50.0,
        "input_ph": 6.5,
        "input_temp": 25.0,
        "input_humidity": 60.0,
        "input_rainfall": 100.0,
    }

    for key, default in defaults.items():
        st.session_state.setdefault(key, default)

    if soil_df is not None and district != "Enter values manually":
        if st.button("Autofill Soil"):
            try:
                row = soil_df.loc[
                    soil_df["District"] == district
                ].iloc[0]

                st.session_state["input_N"] = float(
                    row["N (kg/ha)"]
                )
                st.session_state["input_P"] = float(
                    row["P (kg/ha)"]
                )
                st.session_state["input_K"] = float(
                    row["K (kg/ha)"]
                )
                st.session_state["input_ph"] = float(row["pH"])
                st.rerun()
            except (TypeError, ValueError) as exc:
                st.error(f"Invalid soil data for district: {exc}")

    left, right = st.columns(2)

    with left:
        st.number_input("N (kg/ha)", min_value=0.0, key="input_N")
        st.number_input("P (kg/ha)", min_value=0.0, key="input_P")
        st.number_input("K (kg/ha)", min_value=0.0, key="input_K")
        st.number_input(
            "pH",
            min_value=0.0,
            max_value=14.0,
            key="input_ph",
        )

    with right:
        st.number_input(
            "Temperature (°C)",
            key="input_temp",
        )
        st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            key="input_humidity",
        )
        st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            key="input_rainfall",
        )

    st.caption(
        "Enter rainfall for the same measurement period used in "
        "Crop_recommendation.csv. Current precipitation is not "
        "automatically substituted."
    )

    if st.button("Predict Crop", type="primary"):
        if not CROP_CSV.is_file():
            st.error(
                "Place Crop_recommendation.csv beside app.py."
            )
        else:
            try:
                classifier, scaler, targets, accuracy = (
                    load_crop_model(
                        str(CROP_CSV),
                        CROP_CSV.stat().st_mtime_ns,
                    )
                )

                inputs = {
                    "N": st.session_state["input_N"],
                    "P": st.session_state["input_P"],
                    "K": st.session_state["input_K"],
                    "temperature": st.session_state["input_temp"],
                    "humidity": st.session_state["input_humidity"],
                    "ph": st.session_state["input_ph"],
                    "rainfall": st.session_state["input_rainfall"],
                }

                observation = pd.DataFrame(
                    [inputs],
                    columns=FEATURES,
                )

                scores = classifier.predict_proba(
                    scaler.transform(observation)
                )[0]

                # predict_proba columns follow classifier.classes_,
                # not necessarily the integer 0, 1, 2, ...
                top_indices = np.argsort(scores)[-3:][::-1]

                results = [
                    (
                        str(targets[int(classifier.classes_[index])]),
                        float(scores[index]),
                    )
                    for index in top_indices
                ]

                st.session_state["prediction"] = {
                    "district": district,
                    "inputs": inputs,
                    "top_crops": results,
                    "accuracy": accuracy,
                }

            except Exception as exc:
                st.error(f"Crop prediction failed: {exc}")

    prediction = st.session_state.get("prediction")

    if prediction:
        lines = [
            "Top 3 Crops:",
            *[
                f"{index}. {crop}: {score:.2%} model score"
                for index, (crop, score) in enumerate(
                    prediction["top_crops"],
                    start=1,
                )
            ],
            (
                "Model holdout accuracy: "
                f"{prediction['accuracy']:.2%}"
            ),
        ]

        message = "\n".join(lines)

        try:
            displayed = translate(message, language)
        except Exception as exc:
            displayed = message
            st.warning(f"Translation unavailable: {exc}")

        st.text(displayed)
        st.caption(
            "Model scores are not guaranteed to be calibrated "
            "probabilities of farm success."
        )

        if voice_enabled:
            speak_button(
                displayed,
                language,
                "crop_audio",
            )


# ------------------------- Weather and alerts ----------------------

with tabs[1]:
    st.subheader("Weather and early warning")

    city = st.text_input(
        "Nearest city or town",
        value="Ludhiana",
    )

    col1, col2 = st.columns(2)

    with col1:
        current_clicked = st.button("Fetch Current Weather")

    with col2:
        forecast_clicked = st.button("Show 7-Day Forecast")

    if current_clicked or forecast_clicked:
        if not WEATHERAPI_KEY:
            st.error(
                "Add WEATHERAPI_KEY to .streamlit/secrets.toml."
            )
        elif not city.strip():
            st.error("Enter a city.")
        else:
            try:
                if current_clicked:
                    data = fetch_weather(
                        city.strip(),
                        "current",
                        WEATHERAPI_KEY,
                    )

                    current = data["current"]

                    st.session_state["current_weather"] = {
                        "city": data["location"]["name"],
                        "temperature": float(
                            current["temp_c"]
                        ),
                        "humidity": float(
                            current["humidity"]
                        ),
                        "precipitation": float(
                            current["precip_mm"]
                        ),
                    }

                if forecast_clicked:
                    data = fetch_weather(
                        city.strip(),
                        "forecast",
                        WEATHERAPI_KEY,
                    )

                    st.session_state["forecast"] = [
                        {
                            "Date": item["date"],
                            "Min °C": item["day"]["mintemp_c"],
                            "Max °C": item["day"]["maxtemp_c"],
                            "Condition": (
                                item["day"]["condition"]["text"]
                            ),
                            "Rain mm": (
                                item["day"]["totalprecip_mm"]
                            ),
                        }
                        for item in data["forecast"]["forecastday"]
                    ]

            except Exception as exc:
                st.error(f"Weather request failed: {exc}")

    current = st.session_state.get("current_weather")

    if current:
        st.write(f"**Current weather: {current['city']}**")

        a, b, c = st.columns(3)
        a.metric(
            "Temperature",
            f"{current['temperature']:.1f} °C",
        )
        b.metric(
            "Humidity",
            f"{current['humidity']:.0f}%",
        )
        c.metric(
            "Current precipitation",
            f"{current['precipitation']:.1f} mm",
        )

        if st.button("Use Temperature and Humidity in Crop Input"):
            st.session_state["input_temp"] = (
                current["temperature"]
            )
            st.session_state["input_humidity"] = (
                current["humidity"]
            )
            st.success(
                "Temperature and humidity updated. "
                "Rainfall was left unchanged."
            )

    forecast = st.session_state.get("forecast")

    if forecast:
        st.dataframe(
            pd.DataFrame(forecast),
            hide_index=True,
            use_container_width=True,
        )

        for day in forecast:
            alerts = []

            if day["Max °C"] > 40:
                alerts.append("Extreme heat")

            if day["Min °C"] < 5:
                alerts.append("Frost risk")

            if day["Rain mm"] > 50:
                alerts.append("Heavy rain")

            if alerts:
                st.warning(
                    f"{day['Date']}: {', '.join(alerts)}"
                )

        if len(forecast) < 7:
            st.info(
                f"The weather service returned "
                f"{len(forecast)} forecast day(s), "
                "not seven."
            )


# ------------------------ Leaf disease -----------------------------

with tabs[2]:
    st.subheader("Leaf Disease Detection")

    trained = LEAF_MODEL.is_file() and LEAF_CLASSES.is_file()

    if not trained:
        st.info(
            "Train the leaf model once. The button downloads the "
            "Kaggle dataset and trains your 38-class CNN."
        )

        epochs = st.number_input(
            "Training epochs",
            min_value=1,
            max_value=50,
            value=10,
        )

        if st.button(
            "Download Kaggle Dataset and Train Leaf Model"
        ):
            try:
                with st.spinner(
                    "Downloading/training. This can take a long "
                    "time; leave this page open."
                ):
                    history = train_leaf_model(int(epochs))

                st.success(
                    "Leaf model saved. You can now upload an image."
                )

                st.line_chart(
                    pd.DataFrame({
                        "Training accuracy": history["accuracy"],
                        "Validation accuracy": (
                            history["val_accuracy"]
                        ),
                    })
                )

                st.rerun()

            except Exception as exc:
                st.error(f"Leaf training failed: {exc}")

    else:
        st.success("Trained leaf model found.")

        uploaded_image = st.file_uploader(
            "Upload a leaf image",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded_image is not None:
            try:
                image = Image.open(
                    io.BytesIO(uploaded_image.getvalue())
                ).convert("RGB")

                st.image(
                    image,
                    caption="Uploaded leaf",
                    width=350,
                )

                if st.button(
                    "Detect Disease",
                    type="primary",
                ):
                    model = load_leaf_model(
                        str(LEAF_MODEL),
                        LEAF_MODEL.stat().st_mtime_ns,
                    )

                    class_names = json.loads(
                        LEAF_CLASSES.read_text(
                            encoding="utf-8"
                        )
                    )

                    resized = image.resize((256, 256))
                    pixels = (
                        np.asarray(
                            resized,
                            dtype=np.float32,
                        ) / 255.0
                    )
                    pixels = np.expand_dims(pixels, axis=0)

                    scores = model.predict(
                        pixels,
                        verbose=0,
                    )[0]

                    if len(scores) != len(class_names):
                        raise ValueError(
                            "Saved model and class names disagree. "
                            "Retrain the leaf model."
                        )

                    index = int(np.argmax(scores))
                    disease = class_names[index]
                    score = float(scores[index])

                    pesticide = (
                        DISEASE_TO_PESTICIDE.get(
                            disease,
                            "No mapping available",
                        )
                    )

                    st.session_state["leaf_result"] = {
                        "disease": disease,
                        "score": score,
                        "example_mapping": pesticide,
                    }

            except Exception as exc:
                st.error(
                    f"Could not process the leaf image: {exc}"
                )

        result = st.session_state.get("leaf_result")

        if result:
            text = (
                f"Predicted disease: {result['disease']}\n"
                f"Model score: {result['score']:.2%}\n"
                "Your notebook's example pesticide mapping: "
                f"{result['example_mapping']}"
            )

            try:
                displayed = translate(text, language)
            except Exception as exc:
                displayed = text
                st.warning(
                    f"Translation unavailable: {exc}"
                )

            st.text(displayed)
            st.warning(
                "This is an image-model prediction, not a "
                "confirmed diagnosis or a verified local "
                "pesticide recommendation. Check any treatment "
                "with a qualified local adviser and its label."
            )

            if voice_enabled:
                speak_button(
                    displayed,
                    language,
                    "leaf_audio",
                )


# --------------------- Nearby mandi price -------------------------

with tabs[3]:
    st.subheader("Nearby Mandi Price")

    st.caption(
        "These are the two requested placeholders. "
        "No live mandi-price source or next-year forecasting "
        "model has been supplied."
    )

    first, second = st.columns(2)

    with first:
        if st.button("Show Current Price"):
            st.info(
                "Coming soon: a verified live mandi-price "
                "source must be connected."
            )

    with second:
        if st.button("Predict Next Year Prices"):
            st.info(
                "Coming soon: historical mandi data and a "
                "validated forecasting model are required."
            )


# ---------------------------- AI chat ------------------------------

with tabs[4]:
    st.subheader("Chat with AI")

    if not GROQ_API_KEY:
        st.info(
            "Add GEMINI_API_KEY to .streamlit/secrets.toml "
            "to enable chat."
        )
    else:
        messages = st.session_state.setdefault(
            "chat_messages",
            [],
        )

        for message in messages:
            with st.chat_message(message["role"]):
                st.write(message["text"])

        question = st.chat_input(
            "Ask about your crop recommendation"
        )

        if question:
            context = json.dumps(
                {
                    "crop_prediction": (
                        st.session_state.get("prediction")
                    ),
                    "weather": (
                        st.session_state.get(
                            "current_weather"
                        )
                    ),
                    "forecast": (
                        st.session_state.get("forecast")
                    ),
                    "leaf_prediction": (
                        st.session_state.get("leaf_result")
                    ),
                },
                default=str,
            )[:12000]

            try:
                answer = ask_gemini(
                    question,
                    context,
                    messages,
                )

                if language != "English":
                    try:
                        answer = translate(
                            answer,
                            language,
                        )
                    except Exception as exc:
                        st.warning(
                            "Answer translation failed; "
                            f"showing original: {exc}"
                        )

                messages.append({
                    "role": "user",
                    "text": question,
                })
                messages.append({
                    "role": "assistant",
                    "text": answer,
                })

                st.rerun()

            except Exception as exc:
                st.error(f"AI chat failed: {exc}")

        if (
            voice_enabled
            and messages
            and messages[-1]["role"] == "assistant"
        ):
            speak_button(
                messages[-1]["text"],
                language,
                "chat_audio",
            )
