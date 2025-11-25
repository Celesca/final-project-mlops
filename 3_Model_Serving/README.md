# Part 3: Fraud Detection REST API Service

A FastAPI-based REST API service for real-time fraud detection in financial transactions.


## Installation (with Docker) 🐳

The easiest way to run the API is using Docker, which packages all dependencies and the application into a container.

#### Prerequisites
- Docker Desktop installed ([Download here](https://www.docker.com/products/docker-desktop/))
- Docker Compose (included with Docker Desktop)

#### Using Docker Compose (Recommended)

**Step 1: Navigate to the API directory**
```bash
cd 3_Model_Serving
```

**Step 2: Build and start the container**
```bash
docker-compose up --build
```

The API will be available at http://localhost:8000

To run in detached mode (background):
```bash
docker-compose up -d --build
```

**Step 3: View logs**
```bash
docker-compose logs -f
```

**Step 4: Stop the container**
```bash
docker-compose down
```

Then your server should be ready to use now at port 8000 (localhost:8000)

### ⚠️ Important: Model Artifacts Required

The API requires pre-trained model files to run predictions. These files must be present in the `models/` directory:

```
models/
├── xgb_model.joblib                # Trained XGBoost model
├── preprocessing_artifacts.joblib  # Feature scaler and metadata
├── train_cols.json                 # Training column order
└── xgb_model.json                  # XGBoost metadata
```

**If model files are missing**, you'll see this error:
```
FileNotFoundError: [Errno 2] No such file or directory: 'models/xgb_model.joblib'
```

**To generate model artifacts:**

1. Ensure the training notebook has been run:
   ```bash
   cd scb-fraud-detection
   jupyter notebook 2_Model_Training.ipynb
   ```

2. The notebook will automatically save artifacts to `3_Model_Serving/models/`

3. Rebuild and restart Docker:
```bash
cd 3_Model_Serving
docker-compose down
docker-compose up --build
```

**To test without a trained model** (development only):
- Skip Docker and run locally with `python server.py`
- The API will fail gracefully with "No model loaded" error for predictions
- Use this to verify API infrastructure while training your model

---

## 📋 Features

### API Endpoints
- **POST /train** - Train a new model using the latest master table snapshot
  - Pulls data directly from the `all_transactions` Postgres table (no payload required)
  - Logs every training run to the local MLflow server
  - Automatically promotes the run to “serving” if its validation AUC improves on the current best model

- **POST /predict** - Batch inference endpoint
  - Accepts a list of transactions (same schema as `all_transactions`)
  - Returns the original payload with two additional columns: `predict_proba` and `prediction`
  - Uses the currently promoted model (initially the pre-trained artifact)

- **GET /** and **GET /docs**
  - Health/status summary and interactive Swagger UI

### Machine Learning & Experiment Tracking
- **XGBoost Classifier** - Production-ready fraud detection model
  - Trained on historical transaction patterns
  - Binary classification (fraud/legitimate)
  - Probability scores for risk assessment
- **MLflow integration**
  - Local MLflow tracking server (via docker-compose) to log runs, metrics, and artifacts
  - Automatic model promotion when validation AUC improves
  - Persisted metadata (`models/best_model_meta.json`) keeps track of the best serving model

### Data Management
- **Pydantic Validation** - Request/response data integrity
  - Type-safe transaction models
  - Automatic validation with clear error messages
  - Enum support for transaction types
  - Optional fields for flexible integration

### Infrastructure
- **Docker Support**
  - Compose stack now ships with:
    - `scb-fraud-model-api` (FastAPI service)
    - `fraud-db` (Postgres master table)
    - `mlflow-server` (local MLflow UI + tracking backend)
  - Persistent named volumes for the Postgres DB and MLflow artifacts/metadata
- **Logging System**
  - Structured logs for training jobs, promotions, and inference batches

- **Load Testing Ready** - Performance validation
  - Locust integration with realistic CSV data
  - Concurrent user simulation
  - Performance metrics and bottleneck identification

---

## 📂 Project Structure

```
3_Model_API/
├── server.py                           # Main FastAPI application entry point
├── requirements.txt                    # Python dependencies
├── README.md                           # This documentation
├── Dockerfile                          # Docker image configuration
├── docker-compose.yml                  # Docker Compose orchestration
├── .dockerignore                       # Docker build exclusions
│
├── model_serving/                      # Core application modules
│   ├── schemas.py                      # Pydantic request/response schemas
│   ├── model.py                        # ML model wrapper and inference logic
│   ├── preprocessing.py                # Feature engineering and transformation
│   └── data_access.py                  # Postgres helper for the master table
│
├── models/                             # Trained ML artifacts
│   ├── xgb_model.joblib                # Serialized XGBoost model
│   ├── xgb_model.json                  # XGBoost model metadata
│   ├── preprocessing_artifacts.joblib  # Feature scaler and metadata
│   └── train_cols.json                 # Training column order for inference
│
└── tests/                              # Testing utilities
    ├── locustfile.py                   # Load testing script (Locust)
    └── test.md                         # Test examples and documentation
```

### Key Files Explained

**`server.py`**
- FastAPI application initialization
- `/train` + `/predict` endpoints
- Startup event handlers (model/artifact loading, baseline evaluation)
- MLflow logging + model promotion logic

**`model_serving/schemas.py`**
- `MasterTransaction`, `TrainRequest`, `PredictRequest`, `EnrichedTransaction` schemas
- `TRANSAC_TYPE` enum for validation / preprocessing alignment

**`model_serving/model.py`**
- `FraudDetectionModel` class
- Model loading from joblib artifacts
- Prediction logic (XGBoost Booster and sklearn interfaces)
- Batch probability helper for evaluation/baseline metrics

**`model_serving/preprocessing.py`**
- `transform_transaction()` - Feature engineering pipeline
- `load_preprocessing_artifacts()` - Load scaler and metadata
- One-hot encoding for categorical features
- Feature scaling and column alignment
- Handles enum values from Pydantic

**`model_serving/data_access.py`**
- Helper to connect to Postgres (`all_transactions`) using env vars
- Pandas-friendly fetchers for training + evaluation

**`models/` Directory**
- Generated by the training notebook (`2_Model_Training.ipynb`)
- Contains all artifacts needed for inference
- Stores `best_model_meta.json` describing the currently promoted model
- Must be present for model-based predictions

**`tests/locustfile.py`**
- Load testing configuration
- CSV data loading from `data/fraud_mock.csv`
- Realistic transaction payload generation
- Error logging and debugging hooks

---

## 🧪 Testing the API

### Option 1: Interactive API Documentation (Recommended)

Open your browser and navigate to:

```
http://localhost:8000/docs
```

This opens the **Swagger UI** where you can:
- View all endpoints and their schemas
- Test endpoints directly in the browser
- See request/response examples

---

## Load Testing with Locust

Locust is a scalable load testing tool that simulates concurrent users sending requests to your API.

#### Prerequisites
Install Locust in your virtual environment:
```bash
pip install locust
```

#### Running Locust Tests

**Navigate to the project root:**
```bash
cd scb-fraud-detection
```

**Start Locust with the test file:**
```bash
locust -f 3_Model_API/tests/locustfile.py --host=http://localhost:8000
```

**Open the Locust Web UI:**
- Open your browser and go to http://localhost:8089
- You'll see the Locust interface where you can configure:
  - **Number of users**: Total concurrent users to simulate
  - **Spawn rate**: How many users to start per second
  - **Host**: The target API (already set to http://localhost:8000)

**Example Configuration:**
- Number of users: `100`
- Spawn rate: `10` (ramps up 10 users/second)
- Host: `http://localhost:8000`

Click **Start Swarming** to begin the test.

#### Locust Test Details

The `locustfile.py` includes:
- **CSV Data Loading**: Loads transactions from `data/fraud_mock.csv`
- **Realistic Payloads**: Sends actual transaction data to `/predict`
- **Automatic Type Normalization**: Handles transaction type enums correctly
- **Random Sampling**: Each simulated user picks random transactions
- **Error Logging**: Logs first 5 payloads and all failed requests for debugging

#### Interpreting Results

In the Locust UI, monitor:
- **RPS (Requests Per Second)**: Throughput of your API
- **Response Time**: 50th, 95th, 99th percentiles
- **Failures**: Any 4xx/5xx errors
- **Number of Users**: Current active simulated users

---