# Fraud Detection MLOps Pipeline

A comprehensive end-to-end MLOps system for fraud detection in financial transactions. This project implements a production-ready pipeline with automated data ingestion, model training, serving, monitoring, and auditing capabilities.

## 🎯 Project Overview

This project provides a complete MLOps solution for fraud detection, featuring:

- **Automated Data Pipeline**: Apache Airflow DAGs for data ingestion, preprocessing, and partitioning
- **Model Training**: XGBoost-based fraud detection model with MLflow experiment tracking
- **Model Serving**: FastAPI REST API for real-time predictions and model retraining
- **Data Drift Detection**: Automated monitoring using Evidently AI
- **Audit Web Application**: Flask-based dashboard for reviewing predictions and manual labeling
- **Dockerized Infrastructure**: Complete containerized setup with Docker Compose

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Apache Airflow │────▶│  Model Serving   │────▶│  PostgreSQL     │
│   (Orchestration)│     │   (FastAPI)      │     │   (fraud-db)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                        │                        │
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Data Pipeline  │     │  MLflow Server   │     │  Audit Web App  │
│  (Partitioning) │     │  (Tracking)       │     │   (Flask)       │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

### Component Flow

1. **Data Ingestion**: Airflow DAGs fetch and partition financial transaction data
2. **Model Training**: XGBoost model trained on historical data with MLflow tracking
3. **Model Serving**: FastAPI service provides `/predict` and `/train` endpoints
4. **Drift Detection**: Evidently AI monitors data distribution changes
5. **Audit & Labeling**: Flask web app for reviewing predictions and manual labeling

## 📦 Components

### 1. Data Pipeline (`dags/`)
- **Airflow DAGs**: Orchestrates daily data ingestion, drift detection, and model retraining
- **Tasks**:
  - Database setup and schema initialization
  - Data partitioning by simulation day
  - Daily data ingestion from partitioned datasets
  - Data drift detection using Evidently AI
  - Model retraining triggers
  - Batch prediction scoring

### 2. Model Training (`2_Model_Training/`)
- **Jupyter Notebook**: `MLOps_Train.ipynb` for training XGBoost models
- **Outputs**: Model artifacts saved to `3_Model_Serving/models/`
  - `xgb_model.joblib`: Trained XGBoost model
  - `preprocessing_artifacts.joblib`: Feature scaler and metadata
  - `train_cols.json`: Training column order
  - `training_metadata.json`: Model metadata

### 3. Model Serving (`3_Model_Serving/`)
- **FastAPI REST API** with endpoints:
  - `POST /predict`: Batch inference on transactions
  - `POST /train`: Retrain model from master table
  - `GET /query/GET/predictions`: Query stored predictions
  - `GET /query/GET/frauds`: Get fraud predictions
  - `GET /query/GET/non_frauds`: Get non-fraud predictions
  - `GET /query/GET/stats`: Get prediction statistics
  - `PUT /query/PUT/predictions`: Update actual labels
- **MLflow Integration**: Automatic experiment tracking and model promotion
- **Dockerized**: Standalone container with PostgreSQL connection

### 4. Audit Web Application (`5_Audit_Web_App/`)
- **Flask Web Dashboard** with pages:
  - `/dashboard`: Accuracy metrics and confusion matrix
  - `/frauds`: List of fraud predictions
  - `/non_frauds`: List of legitimate predictions
  - `/false_cases`: False positives and false negatives analysis
  - `/manual_labeling`: Interactive labeling interface
- **Features**: Real-time metrics, interactive charts, one-click labeling

## 🚀 Prerequisites

- **Docker Desktop** (with Docker Compose)
- **Python 3.8+** (for local development)
- **PostgreSQL** (handled by Docker)
- **Jupyter Notebook** (for model training)

## 📥 Installation & Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd final-project-mlops
```

### 2. Start All Services with Docker Compose

```bash
docker compose up -d --build
```

This starts:
- **Airflow Webserver** (http://localhost:8080)
  - Username: `airflow`
  - Password: `airflow`
- **Model Serving API** (http://localhost:8000)
- **MLflow Server** (http://localhost:5000)
- **PostgreSQL Databases**:
  - Airflow DB (internal)
  - Fraud DB (port 5433)

### 3. Initialize Airflow

The first time you run Docker Compose, Airflow will automatically:
- Initialize the database
- Create the default admin user
- Load DAGs from `dags/` directory

### 4. Train Initial Model (Optional)

If you want to train a model before starting the pipeline:

```bash
# Navigate to training directory
cd 2_Model_Training

# Open and run the notebook
jupyter notebook MLOps_Train.ipynb
```

The notebook will save model artifacts to `3_Model_Serving/models/`.

## 🎮 Usage

### Running the Data Pipeline

1. **Access Airflow UI**: http://localhost:8080
2. **Enable the DAG**: `kaggle_fraud_simulation_daily`
3. **Trigger manually** or wait for scheduled runs (every 5 minutes in simulation mode)

The DAG will:
- Partition data by simulation day
- Ingest daily transaction data
- Detect data drift
- Trigger model retraining if drift detected
- Score predictions and store in database

### Using the Model Serving API

#### Make Predictions

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "new_data": [
      {
        "type": "TRANSFER",
        "amount": 50000,
        "oldbalanceOrg": 100000,
        "newbalanceOrig": 50000,
        "oldbalanceDest": 0,
        "newbalanceDest": 50000,
        "nameOrig": "C123456789",
        "nameDest": "M987654321"
      }
    ]
  }'
```

#### Retrain Model

```bash
curl -X POST "http://localhost:8000/train"
```

#### View API Documentation

Visit http://localhost:8000/docs for interactive Swagger UI.

### Using the Audit Web Application

1. **Start the Flask app**:
```bash
cd 5_Audit_Web_App
pip install -r requirements.txt
python app.py
```

2. **Access the dashboard**: http://localhost:5000

3. **Navigate to pages**:
   - Dashboard: Overview metrics
   - Manual Labeling: Review and label predictions
   - False Cases: Analyze model errors
   - Frauds/Non-Frauds: Filter predictions

### Monitoring with MLflow

1. **Access MLflow UI**: http://localhost:5000
2. **View experiments**: All training runs are logged automatically
3. **Compare models**: Metrics, parameters, and artifacts tracked

## 📁 Project Structure

```
final-project-mlops/
├── dags/                          # Apache Airflow DAGs
│   ├── kaggle_fraud_simulation_dag.py
│   ├── config.py
│   ├── tasks/                     # DAG task modules
│   │   ├── data_ingestion.py
│   │   ├── data_preparation.py
│   │   ├── database_setup.py
│   │   ├── drift_detection.py
│   │   └── model_serving.py
│   └── utils/                     # Utility modules
│       ├── database.py
│       ├── dataset_utils.py
│       ├── drift_detector.py
│       └── prediction_storage.py
│
├── 2_Model_Training/               # Model training notebook
│   └── MLOps_Train.ipynb
│
├── 3_Model_Serving/               # FastAPI model serving
│   ├── server.py
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── model_serving/             # Core modules
│   │   ├── model.py
│   │   ├── preprocessing.py
│   │   ├── schemas.py
│   │   ├── data_access.py
│   │   └── training.py
│   ├── models/                    # Model artifacts
│   │   ├── xgb_model.joblib
│   │   ├── preprocessing_artifacts.joblib
│   │   └── train_cols.json
│   └── tests/
│       └── locustfile.py          # Load testing
│
├── 5_Audit_Web_App/               # Flask audit dashboard
│   ├── app.py
│   ├── backend.py
│   ├── requirements.txt
│   ├── templates/                 # HTML templates
│   │   ├── dashboard.html
│   │   ├── manual_labeling.html
│   │   ├── false_cases.html
│   │   └── ...
│   └── build/                     # React build (optional)
│
├── data/                          # Data storage
│   ├── partitioned_data/          # Partitioned by simulation_day
│   └── processed_fraud_data_*.csv
│
├── db_init/                       # Database initialization
│   └── 001-init.sql
│
├── docker-compose.yml             # Docker orchestration
├── Dockerfile                     # Airflow base image
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🔧 Configuration

### Environment Variables

Key environment variables (set in `docker-compose.yml`):

| Variable | Default | Description |
|----------|---------|-------------|
| `FRAUD_DB_HOST` | `fraud-db` | PostgreSQL host |
| `FRAUD_DB_PORT` | `5432` | PostgreSQL port |
| `FRAUD_DB_NAME` | `frauddb` | Database name |
| `FRAUD_DB_USER` | `fraud` | Database user |
| `FRAUD_DB_PASSWORD` | `fraud123` | Database password |
| `MODEL_SERVING_BASE_URL` | `http://model-serving:8000` | Model API URL |
| `MLFLOW_TRACKING_URI` | `http://mlflow-server:5000` | MLflow server URL |

### DAG Configuration

Edit `dags/config.py` to customize:
- Dataset source (Kaggle)
- Data paths
- Database connections
- Drift detection thresholds
- Simulation parameters

## 🧪 Testing

### Load Testing with Locust

```bash
cd 3_Model_Serving/tests
locust -f locustfile.py --host=http://localhost:8000
```

Access Locust UI at http://localhost:8089

### API Testing

Use the interactive Swagger UI at http://localhost:8000/docs or test endpoints with curl/Postman.

## 📊 Features

### Data Pipeline
- ✅ Automated daily data ingestion
- ✅ Data partitioning by simulation day
- ✅ Data drift detection with Evidently AI
- ✅ Automatic model retraining on drift

### Model Management
- ✅ XGBoost fraud detection model
- ✅ MLflow experiment tracking
- ✅ Automatic model promotion based on validation AUC
- ✅ Feature preprocessing pipeline
- ✅ Batch and real-time inference

### Monitoring & Auditing
- ✅ Prediction storage in PostgreSQL
- ✅ Accuracy metrics (F1, Precision, Recall)
- ✅ Confusion matrix visualization
- ✅ False positive/negative analysis
- ✅ Manual labeling interface

### Infrastructure
- ✅ Dockerized services
- ✅ PostgreSQL for data storage
- ✅ MLflow for experiment tracking
- ✅ Airflow for orchestration

## 🐛 Troubleshooting

### Airflow DAG Not Appearing

1. Check DAG files are in `dags/` directory
2. Verify Python syntax: `docker compose logs airflow-scheduler`
3. Refresh Airflow UI

### Model Serving API Errors

1. **Model not found**: Ensure model artifacts exist in `3_Model_Serving/models/`
2. **Database connection**: Check `fraud-db` container is running
3. **Port conflicts**: Verify port 8000 is available

### Database Connection Issues

1. Check containers are running: `docker compose ps`
2. Verify database health: `docker compose logs fraud-db`
3. Check connection strings in `docker-compose.yml`

### Audit Web App Can't Connect

1. Verify model serving API is running: `curl http://localhost:8000/`
2. Check `API_BASE_URL` environment variable
3. Ensure predictions exist in database (run DAG first)

## 📚 Additional Documentation

- **Model Serving**: See `3_Model_Serving/README.md`
- **Audit Web App**: See `5_Audit_Web_App/README.md`
- **API Documentation**: http://localhost:8000/docs (when running)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

[Add your license information here]

## 👥 Authors

[Add author information here]

---

**Note**: This is a simulation-based project using the Kaggle Financial Fraud Detection dataset. The pipeline runs in simulation mode (every 5 minutes) for demonstration purposes. Adjust the schedule in `dags/kaggle_fraud_simulation_dag.py` for production use.

