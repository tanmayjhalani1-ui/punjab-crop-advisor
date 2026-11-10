# 🌾 AgriPulse AI — Advanced Crop Advisor

An advanced, production-oriented evolution of the **Punjab Crop Advisor** reference project.

## What it does

AgriPulse combines a crop ML model with live environmental and market context:

1. **Crop ML** — Random Forest + Extra Trees soft-voting ensemble.
2. **Farm profile** — N, P, K, pH and climate features.
3. **Live weather** — geocodes a village/city and retrieves current + 7-day weather from Open-Meteo.
4. **Dynamic advisory** — model probabilities are adjusted with a crop-specific weather-fit layer.
5. **Live mandi prices** — optional Government of India OGD/data.gov.in integration using the configured API key.
6. **Explainability** — shows model probability, advisory score and the contextual reasons behind ranking.
7. **Offline resilience** — if weather, internet or the training dataset is unavailable, the application remains runnable with a deterministic fallback dataset.
8. **Diagnostics** — model accuracy/F1, feature values and reference soil data.

## Run locally

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train.py
streamlit run app/app.py
```

## Real-time mandi data

Create `.env` from `.env.example` and add your `DATA_GOV_IN_KEY`. The application queries the configured Government OGD resource. Without a key, only the market layer is disabled; crop prediction and weather still work.

## Architecture

```text
Streamlit UI
    │
    ├── Farm profile
    ├── Location → Geocoding → Weather API
    └── Crop model → Top-K probabilities
                 │
                 ├── Weather suitability
                 └── Mandi market context
                         │
                         ▼
                 Explainable advisory
```

## Important

This is a decision-support system, not a guarantee of yield or profit. Before a real planting decision, validate the recommendation against a current soil test, local agronomic guidance, water availability, seed variety and official advisories.

## Reference

Original project: https://github.com/tanmayjhalani1-ui/punjab-crop-advisor

The original crop feature schema is retained: `N, P, K, temperature, humidity, ph, rainfall → label`.
