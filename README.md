# 📡 MarketFit — AI-Powered Electronic Product Market Fit Predictor

A full-stack system that predicts Market Fit Score (0–100) for electronic products using ML + AI recommendations.

---

## 🏗 Architecture

```
Frontend (HTML/JS)  →  Node.js/Express :5000  →  FastAPI ML :8000
                    ↘                    ↘           ↗
                     MongoDB            TinyLlama (Local LLM)
```

---

## 📂 Project Structure

```
market-fit-predictor/
├── frontend/
│   └── index.html              # Single-page app (open in browser)
├── backend/
│   ├── server.js               # Express API server
│   ├── models.js               # Mongoose User + Product schemas
│   ├── aiRecommendation.js     # TinyLlama integration
│   ├── package.json
│   └── .env.example
├── ml-service/
│   ├── generate_and_train.py   # Train the RandomForest model
│   ├── main.py                 # FastAPI prediction service
│   └── requirements.txt
├── models/                     # Auto-created: saved ML artifacts
│   ├── market_fit_model.joblib
│   ├── scaler.joblib
│   └── feature_meta.json
└── dataset/                    # Auto-created: synthetic CSV
    └── market_fit_data.csv
```

---

## 🚀 Setup Instructions

### Prerequisites
- Node.js ≥ 18
- Python ≥ 3.10
- MongoDB (local or Atlas)
- [Ollama](https://ollama.com/) (to run TinyLlama locally)

---

### Step 0 — Install & Run TinyLlama via Ollama

```bash
# Install Ollama (if not already installed)
# Visit https://ollama.com/download for your OS

# Pull the TinyLlama model
ollama pull tinyllama

# Start the Ollama server (runs on http://localhost:11434 by default)
ollama serve
```

Verify: http://localhost:11434

---

### Step 1 — Train the ML Model

```bash
cd ml-service

# Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate dataset + train model (creates models/ and dataset/)
python generate_and_train.py
```

You should see output like:
```
Dataset saved → ../dataset/market_fit_data.csv  (5000 rows)
MAE: 3.21  |  R²: 0.9712
Model  saved → ../models/market_fit_model.joblib
✅ Training complete!
```

---

### Step 2 — Start the FastAPI ML Service

```bash
# Still in ml-service/ with venv active
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Verify: http://localhost:8000/health  
API docs: http://localhost:8000/docs

---

### Step 3 — Configure & Start the Node.js Backend

```bash
cd backend

# Install dependencies
npm install

# Copy and edit environment variables
cp .env.example .env
# Edit .env with your values (see below)

# Start the server
npm run dev        # development (nodemon)
# or
npm start          # production
```

#### Required `.env` settings:

```env
MONGO_URI=mongodb://localhost:27017/market_fit_db
JWT_SECRET=change_this_to_a_long_random_string
FASTAPI_URL=http://localhost:8000

# TinyLlama via Ollama (running locally)
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=tinyllama
AI_PROVIDER=ollama
```

Verify: http://localhost:5000/health

---

### Step 4 — Open the Frontend

Simply open `frontend/index.html` in your browser:

```bash
open frontend/index.html        # macOS
# or double-click the file in your file explorer
```

---

## 🔌 API Reference

### Backend (Node.js :5000)

| Method | Endpoint           | Auth | Description                     |
|--------|--------------------|------|---------------------------------|
| POST   | /register          | No   | Create account                  |
| POST   | /login             | No   | Get JWT token                   |
| POST   | /predict-product   | JWT  | Run prediction + get AI insight |
| GET    | /user-products     | JWT  | Fetch prediction history        |
| GET    | /health            | No   | Server status                   |

### ML Service (FastAPI :8000)

| Method | Endpoint    | Description                   |
|--------|-------------|-------------------------------|
| POST   | /predict    | Predict market fit score      |
| GET    | /categories | List categories + features    |
| GET    | /health     | Service health                |

#### Example predict call:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "category": "smartphone",
    "features": {
      "ram_gb": 12,
      "storage_gb": 256,
      "battery_mah": 5000,
      "camera_mp": 108,
      "display_size": 6.7
    }
  }'
```

Response:
```json
{
  "market_fit_score": 84.52,
  "category": "smartphone",
  "features_used": { ... }
}
```

---

## 🧠 ML Model Details

- **Algorithm**: RandomForestRegressor (200 estimators, depth 12)
- **Dataset**: 5,000 synthetic samples (1,250 per category)
- **Features**: Up to 5 numeric features per category, padded + one-hot encoded
- **Preprocessing**: StandardScaler
- **Typical accuracy**: MAE < 5, R² > 0.95

---

## 🤖 AI Recommendation — TinyLlama

AI-powered product insights are generated locally using **TinyLlama** via [Ollama](https://ollama.com/), with no external API calls or API keys required.

- **Model**: `tinyllama` (1.1B parameters, runs on CPU/GPU)
- **Runtime**: Ollama (local inference server)
- **Endpoint**: `http://localhost:11434/api/generate`
- **Privacy**: All inference happens on your machine — no data leaves your system

> ⚠️ TinyLlama is a lightweight model. Response quality may be limited compared to larger models. For more detailed recommendations, consider switching to `llama3` or `mistral` via Ollama by updating `OLLAMA_MODEL` in your `.env`.

---

## 🗄 MongoDB Schema

**User**
```js
{ name, email, password (hashed), createdAt }
```

**Product**
```js
{
  userId, category, specifications (Map<String,Number>),
  market_fit_score, ai_recommendation: {
    best_use_case, advantages[], disadvantages[], suggestions[]
  },
  createdAt
}
```

---

## ⚙️ Quick Start (All at once)

```bash
# Terminal 1 — TinyLlama (Ollama)
ollama serve

# Terminal 2 — ML service
cd ml-service && source venv/bin/activate
python generate_and_train.py && uvicorn main:app --port 8000

# Terminal 3 — Backend
cd backend && npm run dev

# Browser — Frontend
open frontend/index.html
```
