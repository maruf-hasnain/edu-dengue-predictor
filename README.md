# Urban Dengue Outbreak Early-Warning System

A 2-to-4-week early-warning prediction system and interactive dashboard for municipal public health officers, built for the **EDU AI Engineering Hackathon 2026 (Phase 1)**.

## Project Structure

```text
edu-dengue-predictor/
├── app/
│   ├── templates/
│   │   └── index.html       # Tailwind CSS + Chart.js dashboard
│   ├── __init__.py
│   └── main.py              # FastAPI server & lifespan artifact loader
├── artifacts/               # Serialized ML models (joblib)
├── data/
│   └── dengue.csv           # Historical dengue & weather dataset
├── notebooks/               # Exploratory data analysis notebooks
├── src/
│   ├── __init__.py
│   ├── features.py          # Multi-week grouped weather lag pipeline
│   └── train.py             # Chronological validation & LightGBM training
├── .env.example
├── .gitignore
├── ARCHITECTURE.md
├── README.md
└── requirements.txt
```

## Quick Start (<= 3 Minutes)

1. **Create & activate environment:**
   ```cmd
   conda create -n fastapi python=3.11 -y
   conda activate fastapi
   ```

2. **Install dependencies:**
   ```cmd
   pip install -r requirements.txt
   ```

3. **Run the FastAPI server:**
   ```cmd
   uvicorn app.main:app --reload
   ```

4. **Open the Dashboard:**
   Visit `http://127.0.0.1:8000` in your browser (API docs at `http://127.0.0.1:8000/docs`).

## AI Disclosure
This project utilizes AI coding assistance (Google Antigravity / Gemini) for scaffolding, feature pipeline structuring, and UI templating in compliance with the hackathon disclosure rules.
