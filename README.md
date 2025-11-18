# CPE393: Machine Learning Operations

## Fraud Detection System

Project: End-to-end fraud detection MLOps pipeline used for the CPE393 course. This repository is organized into five main parts (folders) that show stages of an MLOps workflow from data ingestion to model retraining and audit.

**Repository Structure**
- `1_DAG_Data/` — Part 1
- `2_Data_Drifts/` — Part 2
- `3_Model_Serving/` — Part 3
- `4_Retraining_MLFlow/` — Part 4
- `5_Audit_Web_App/` — Part 5

**How to use**
- **Overview:** Explore each folder to see code, notebooks, or manifests specific to that stage. Each part is designed to be self-contained with its own scripts or notes.
- **Run order:** Work through Parts 1 → 5 to follow a typical pipeline: data ingestion → monitoring/drift detection → serving → retraining & tracking → audit/web UI.
- **Notes:** Many subfolders may include their own `README` or instructions — check them for commands, dependencies, and environment details.

**Part 1 — DAG & Data (`1_DAG_Data`)**
- **Purpose:** Data ingestion orchestration and preprocessing (Airflow DAGs, ingestion scripts, sample data/exporters).
- **Contents (typical):** DAG definitions, ingestion scripts, sample datasets.
- **How to run:** Follow the folder's README for Airflow or script invocation details.

**Part 2 — Data Drifts (`2_Data_Drifts`)**
- **Purpose:** Data quality checks and drift detection components (monitoring notebooks, statistical tests, alerts).
- **Contents (typical):** Drift detection scripts, EDA notebooks, monitoring configurations.
- **How to run:** Use the included notebooks or scripts; integrate with monitoring backends if provided.

**Part 3 — Model Serving (`3_Model_Serving`)**
- **Purpose:** Model deployment and serving examples (API wrappers, container configs, inference code).
- **Contents (typical):** Serving code, example REST endpoints, Dockerfiles or model wrappers.
- **How to run:** See the folder README for instructions to start the model server or container.

**Part 4 — Retraining & MLflow (`4_Retraining_MLFlow`)**
- **Purpose:** Retraining pipelines, experiment tracking, and MLflow integration for model versioning.
- **Contents (typical):** Retraining scripts, MLflow experiment examples, evaluation metrics and artifacts.
- **How to run:** Use MLflow commands or provided scripts to reproduce experiments and register models.

**Part 5 — Audit Web App (`5_Audit_Web_App`)**
- **Purpose:** Lightweight web UI for audit, model explainability, and human review of predictions.
- **Contents (typical):** Web app source, UI assets, endpoints for audit workflows.
- **How to run:** Check the app README for framework-specific run commands (Flask/FastAPI/Streamlit etc.).

**Contributing & Next Steps**
- **Add more docs:** Add per-folder `README.md` files with commands and dependencies.
- **Environment:** Provide a top-level `requirements.txt` or `environment.yml` for reproducibility.
- **Contact:** For questions about this project or course, open an issue or contact the maintainers.

---
_This README is a high-level index; open each folder to see specific code, notebooks, and run instructions._