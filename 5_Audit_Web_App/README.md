# Fraud Detection Audit Web Application

A Flask web application for auditing fraud detection model predictions. This app connects to the FastAPI backend (`model-serving`) to fetch predictions and manage labeling.

## Architecture

```
┌─────────────────┐     ┌─────────────────────┐     ┌──────────────┐
│   Flask App     │────▶│  FastAPI Backend    │────▶│  PostgreSQL  │
│  (localhost:5000)│     │ (localhost:8000)    │     │   (fraud-db) │
└─────────────────┘     └─────────────────────┘     └──────────────┘
```

The Flask app communicates with the FastAPI backend via REST API calls:

- `GET /query/GET/predictions` - Fetch all predictions
- `GET /query/GET/frauds` - Fetch fraud predictions (prediction=True)
- `GET /query/GET/non_frauds` - Fetch non-fraud predictions (prediction=False)
- `GET /query/GET/stats` - Fetch prediction statistics
- `PUT /query/PUT/predictions` - Update actual_label for a prediction

## Features

### 🎯 Dashboard (`/dashboard`)
- Accuracy metrics (F1, Precision, Recall, Accuracy)
- Confusion matrix visualization
- Prediction distribution charts
- Quick links to other pages

### 🚨 Fraud Predictions (`/frauds`)
- View all transactions predicted as fraud
- See transaction details and probability scores
- Label transactions directly from the list

### ✅ Non-Fraud Predictions (`/non_frauds`)
- View all transactions predicted as legitimate
- See transaction details and probability scores
- Label transactions directly from the list

### ❌ False Cases Analysis (`/false_cases`)
- **False Positives**: Legitimate transactions flagged as fraud
- **False Negatives**: Fraud transactions missed by the model
- Only shows cases where `actual_label` has been set

### 🏷️ Manual Labeling (`/manual_labeling`)
- Review unlabeled predictions (actual_label is NULL)
- Interactive cards with risk indicators
- One-click labeling as Fraud or Legitimate
- Labels saved to the backend database

## Prerequisites

1. **Docker containers running**:
   ```bash
   cd c:\Users\Sawit\Desktop\final-project-mlops
   docker compose up -d
   ```

2. **Predictions in the database** (run the DAG or make predictions via API)

## Installation

1. **Navigate to the audit web app directory**:
   ```bash
   cd 5_Audit_Web_App
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

### Option 1: Run the Flask app directly
```bash
python app.py
```
Then visit: http://localhost:5000

### Option 2: Run with custom backend URL
```bash
set API_BASE_URL=http://localhost:8000
python app.py
```

### Option 3: Run the demo script
```bash
python demo.py
```
This tests the backend connection and shows sample data.

## API Endpoints

The Flask app also exposes its own API endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predictions` | GET | All predictions from backend |
| `/api/frauds` | GET | Fraud predictions only |
| `/api/non_frauds` | GET | Non-fraud predictions only |
| `/api/metrics` | GET | Calculated accuracy metrics |
| `/api/stats` | GET | Prediction statistics |
| `/api/false_cases` | GET | False positives and negatives |
| `/api/unlabeled` | GET | Unlabeled predictions |
| `/api/label` | POST | Label a prediction |
| `/api/transaction/<id>` | GET | Transaction details |
| `/api/health` | GET | Backend health check |

## Labeling Workflow

1. Go to **Manual Labeling** page
2. Review each prediction card:
   - Transaction type and amount
   - Fraud probability score
   - Origin/destination account details
   - Risk indicators
3. Click **Mark as Fraud** or **Mark as Legit**
4. The label is sent to the backend via `PUT /query/PUT/predictions`
5. Refresh Dashboard to see updated accuracy metrics

## File Structure

```
5_Audit_Web_App/
├── app.py              # Main Flask application (template-based)
├── backend.py          # Alternative Flask app (for React frontend)
├── demo.py             # Demo/test script
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── templates/          # Jinja2 HTML templates
│   ├── base.html
│   ├── dashboard.html
│   ├── predictions_list.html
│   ├── manual_labeling.html
│   └── false_cases.html
└── build/              # React build (for backend.py)
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE_URL` | `http://localhost:8000` | FastAPI backend URL |

## Troubleshooting

### Backend not available
```
⚠️ Warning: Cannot connect to backend at http://localhost:8000
```
**Solution**: Start the Docker containers:
```bash
docker compose up -d model-serving fraud-db
```

### No predictions found
**Solution**: Run the prediction DAG or make manual predictions via the API:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"new_data": [{"type": "TRANSFER", "amount": 50000, ...}]}'
```

### Metrics show None
**Solution**: Label some predictions first. Accuracy metrics require at least one prediction with `actual_label` set.
