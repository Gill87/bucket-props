# 🏀 BucketProps

**BucketProps** is a data-driven NBA player props platform that uses machine learning to generate **over/under point predictions** for daily games.
The goal is simple: turn historical NBA data into clear, confidence-weighted insights that focus on the numbers to make predictions.

---

<img width="2452" height="868" alt="Screenshot 2026-01-04 215528" src="https://github.com/user-attachments/assets/60edfc83-5f0d-43e9-ac5d-96800482bb6f" />

## 🚀 What BucketProps Does

* Predicts **NBA player points** using machine learning
* Outputs **Over / Under picks** with confidence scores
* Updates player data continuously via cached NBA stats
* Separates **data updates** from **model retraining** for efficiency
* Displays picks in a clean **Next.js frontend**

---

## 🧠 How It Works

### 1. Data Collection

* Historical NBA player game logs
* Vegas lines from prop platforms (e.g. PrizePicks-style props)
* Player-level stats cached locally to avoid repeated API hits

### 2. Feature Engineering

Features include (but aren’t limited to):

* Rolling & expanding averages
* Recent game performance trends
* Usage-based indicators
* Opponent-adjusted context (where available)

### 3. Model Training

* Machine learning model trained across **multiple seasons**
* Designed to generalize across players, not overfit to single games
* Retrained periodically (not daily) to stay aligned with league trends
* Utilizes **Gradient Boosting Regression** to make accurate predictions

### 4. Prediction Output

Each pick includes:

* Player name
* Vegas line
* Model-predicted points
* Over / Under recommendation
* Confidence score

---

## 🖥️ Tech Stack

### Backend / ML

* **Python**
* **pandas**, **NumPy**
* **scikit-learn**
* NBA stats APIs
* Local player cache system

### Frontend

* **Next.js**
* **React**
* Client-side data fetching from `picks.json`
* Simple, fast, and mobile-friendly UI

---

## 📁 Project Structure (Simplified)

```
bucketprops/
│
├── scripts/
│   ├── update_player_cache.py
│   ├── train_model.py
│   └── predict.py
│
├── data/
│   ├── player_cache/
│   ├── training/
│   └── picks.json
│
├── model/
│   └── .pkl file
│   └── metadata
|
├── frontend/
│   └── Next.js app
│
└── README.md
```

---

## 🔁 Model Update Strategy

* **Player stats** → updated frequently (daily or near-daily)
* **Model retraining** → done periodically (e.g. bi-weekly)
* This avoids noise from single games while keeping predictions relevant

---

## ⚠️ Disclaimer

BucketProps is for **educational and informational purposes only**.
Predictions are not guarantees. No financial or betting advice is provided.

---

## 👤 Author

* Built by **Guritfak Gill**
* Computer Science student at UC Davis & ML-focused software engineer,
* Passionate about sports analytics, data systems, and machine learning models

---
