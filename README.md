# Urban Dengue Outbreak Early-Warning System

An AI-powered **2-to-4-week Dengue outbreak early-warning system** and interactive municipal surveillance dashboard built for the **EDU AI Engineering Hackathon 2026 — Phase 1 (Challenge: AI for Urban Dengue Outbreak)**.

## Overview

Across Bangladesh's urban centers (Dhaka, Chattogram, Gazipur, and Narayanganj), monsoon rainfall creates stagnant breeding water for *Aedes aegypti* mosquitoes. Because egg maturation and viral incubation take 2 to 4 weeks, reactive spraying after hospital admissions spike is always weeks behind transmission.

This system models both **multi-week ward-grouped weather lags (1–4 weeks)** and **hyper-local ward vulnerability** (`water_logging_risk_index`, `population_density_sqkm`, `sanitation_score`) across **60 wards in 5 cities** (`31,320` weekly records from `2016–2025`), delivering:

- **Dynamic Outbreak Predictions & Risk Tiers:** Weekly case forecasts mapped to operational tiers (`Low`, `Medium`, `High`, `Critical`) with actionable public health advisories.
- **Explainable Weather & Ward Drivers:** Real-time per-prediction Top 5 feature contributions via LightGBM TreeSHAP (`pred_contrib=True`).
- **Interactive Scenario Simulator:** Forecast any future date with customizable temperature, humidity, rainfall, a **Prior 3-Week Lagged Rainfall** slider (`0–250 mm`), and 4 one-click 2026 seasonal presets.
- **Blind 2025 Chronological Validation Dashboard:** Live Chart.js comparison of actual vs. predicted weekly totals on the untouched 2025 test year (`MAE: 0.800`, `RMSE: 1.546`, `R²: 0.846`, `4-Tier Accuracy: 84.2%`).

---

## Team


| :--- | :--- |
| **Team Name** | `Raindrop` |
| **Member 1 (Team Leader)** | `Maruf Hasnain` — `253019412` |
| **Member 2** | `Humaira Tasnim` — `253018912` |
| **Member 3** | `Orpita Barua Sarna` — `243011712` |

---

## Tech Stack

- **Language:** Python 3.10+
- **Backend & API:** FastAPI, Uvicorn, Pydantic, Jinja2
- **Machine Learning & Data Pipeline:** LightGBM (Tweedie regression), Scikit-Learn, Pandas, NumPy, Joblib
- **Frontend Dashboard:** HTML5, Tailwind CSS (CDN), Chart.js (CDN), Vanilla JavaScript

---

## Project Structure

```text
edu-dengue-predictor/
├── app/
│   ├── templates/
│   │   └── index.html               # Interactive dark-mode surveillance dashboard
│   ├── __init__.py
│   └── main.py                      # FastAPI server, lifespan loader & REST API
├── artifacts/
│   └── dengue_pipeline.joblib       # Pre-trained LightGBM bundle + metadata
├── data/
│   └── dengue.csv                   # Historical weekly dengue & weather data (2016–2025)
├── notebooks/                       # Exploratory analysis workspace
├── src/
│   ├── __init__.py
│   ├── features.py                  # Ward-grouped 1–4 week weather lag pipeline
│   └── train.py                     # 3-way chronological split & LightGBM training
├── .env.example                     # Safe example environment variables (no secrets needed)
├── .gitignore
├── ARCHITECTURE.md                  # System architecture, data flow & efficiency design
├── README.md
└── requirements.txt                 # Standard pip dependencies
```

---

## Setup & Run Instructions

No external API keys or special setup are required. Requires **Python 3.10+** and `pip`.

### 1. Clone the Repository
```bash
git clone https://github.com/maruf-hasnain/edu-dengue-predictor
cd edu-dengue-predictor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Application
```bash
uvicorn app.main:app --reload
```

> **Note on Reproducibility:** A pre-trained model artifact (`artifacts/dengue_pipeline.joblib`) is bundled for instant startup. If you want to retrain from scratch (takes ~3–5 seconds) or regenerate the artifact for your local Python/NumPy version, run:
> ```bash
> python -m src.train
> ```
> *(If the bundled artifact ever hits a Python/NumPy version mismatch on startup, FastAPI automatically regenerates it via `train_and_export()` during startup.)*

### 5. Open in Browser
- **Interactive Dashboard:** `http://127.0.0.1:8000`
- **Interactive Swagger API Docs:** `http://127.0.0.1:8000/docs`
- **Health Check:** `http://127.0.0.1:8000/health`

---

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Renders the interactive public health dashboard with 2025 test metrics |
| `GET` | `/health` | Server and model artifact readiness check |
| `GET` | `/api/ward/{ward_id}` | Returns ward vulnerability metadata and latest 4-week weather defaults |
| `GET` | `/api/validation-chart` | Returns the 52-week untouched 2025 actual vs. predicted validation series |
| `POST` | `/api/predict` | Computes multi-week lags, predicts weekly cases, maps Risk Tier, and returns Top 5 feature drivers |

---

## Model Validation Summary (Chronological Split)

| Split Period | Purpose | Metric | Result |
| :--- | :--- | :--- | :--- |
| **2016–2023** | Training Set | — | Historical multi-ward baseline |
| **2024** | Chronological Validation | RMSE | `1.501` |
| **2025 (Untouched)** | Blind Chronological Test | **MAE** | **`0.800`** |
| **2025 (Untouched)** | Blind Chronological Test | **RMSE** | **`1.546`** |
| **2025 (Untouched)** | Blind Chronological Test | **R²** | **`0.846`** |
| **2025 (Untouched)** | Blind Chronological Test | **4-Tier Risk Accuracy** | **`84.2%`** |

---

## AI Tools & External APIs Disclosure

In accordance with **Section 9 & Appendix B** of the EDU AI Engineering Hackathon 2026 Official Rulebook:

| Tool / API | Type | Usage & Purpose |
| :--- | :--- | :--- |
| **Google Antigravity (Gemini)** | AI Coding Assistant | Assisted with FastAPI backend scaffolding, grouped lag pipeline structuring, Tailwind/Chart.js dashboard templating, and documentation drafting. All code was reviewed, tested, and verified locally by the team. |
| **External Paid LLM / Weather APIs** | Runtime External APIs | **None used at runtime.** All forecasting and TreeSHAP explainability run 100% locally via serialized LightGBM (`joblib`) at zero API/token cost. |

---

## Attributions

- **Dataset:** Official EDU AI Engineering Hackathon 2026 Phase 1 Urban Dengue dataset (`data/dengue.csv`).
- **Open-Source Frameworks & Libraries:** [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/), [Pydantic](https://docs.pydantic.dev/), [Jinja2](https://jinja.palletsprojects.com/), [LightGBM](https://lightgbm.readthedocs.io/), [scikit-learn](https://scikit-learn.org/), [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [Joblib](https://joblib.readthedocs.io/).
- **Frontend CDN Libraries:** [Tailwind CSS](https://tailwindcss.com/) and [Chart.js](https://www.chartjs.org/).

---

