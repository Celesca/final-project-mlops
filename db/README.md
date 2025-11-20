# Database Management System for Financial Data

This directory contains the necessary files to set up a PostgreSQL database with `pgAdmin` for managing financial log data. The `db.py` script handles reading a CSV file, processing it, and inserting it into the database. A test script (`test.py`) is also provided to verify the integration.

## Table of Contents

-   [Overview](#overview)
-   [Prerequisites](#prerequisites)
-   [Setup](#setup)
    -   [1. Create .env File](#1-create-env-file)
    -   [2. Start Database Services (Docker Compose)](#2-start-database-services-docker-compose)
    -   [3. Install Python Dependencies](#3-install-python-dependencies)
    -   [4. Place CSV File](#4-place-csv-file)
-   [Usage](#usage)
    -   [Running db.py](#running-dbpy)
    -   [Verifying with pgAdmin](#verifying-with-pgadmin)
-   [Testing](#testing)

## Overview

The `db.py` script is designed to:
1.  Connect to a PostgreSQL database using credentials loaded from a `.env` file.
2.  Read financial log data from `dataset/Synthetic_Financial_datasets_log.csv` into a Pandas DataFrame.
3.  Add a new `probability` column with random values to the DataFrame.
4.  Dynamically create a `financial_data` table in the database, matching the DataFrame's structure.
5.  Efficiently insert all processed data from the DataFrame into the `financial_data` table.

## Prerequisites

Before you begin, ensure you have the following installed:

-   **Docker Desktop:** For running PostgreSQL and pgAdmin containers.
-   **Python 3.8+:** For running the Python scripts.
-   **`Synthetic_Financial_datasets_log.csv`:** This financial dataset file.

## Setup

Follow these steps to set up your database environment and prepare the Python script for execution.

### 1. Create .env File

Create a file named `.env` in the **root directory of your project** (the directory containing `docker-compose.yml`) with the following content:

```
# PostgreSQL Connection Details
DB_NAME=mydatabase
DB_USER=admin
DB_PASS=password
DB_HOST=localhost
DB_PORT=5432

# pgAdmin Connection Details
PGADMIN_DEFAULT_EMAIL=admin@example.com
PGADMIN_DEFAULT_PASSWORD=password
```

### 2. Start Database Services (Docker Compose)

Navigate to the **root directory of your project** in your terminal and run the following command. This will start the PostgreSQL database and the pgAdmin web interface.

```bash
docker-compose up -d
```
This command will create two containers: `postgres_db` (PostgreSQL server) and `pgadmin_gui` (pgAdmin web interface).

### 3. Install Python Dependencies

Install the required Python libraries using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Place CSV File

Ensure that the `Synthetic_Financial_datasets_log.csv` file is placed in the `dataset/` directory. For example, the full path should be `your_project_root/dataset/Synthetic_Financial_datasets_log.csv`.

## Usage

### Running db.py

After completing the setup, you can run the `db.py` script. This script will read your CSV, process it, and load it into the database.

```bash
python db/db.py
```

Upon successful execution, you will see output indicating database connection, data loading, table creation, and data insertion.

### Verifying with pgAdmin

You can use `pgAdmin` to visually inspect the database and confirm that the `financial_data` table has been created and populated correctly.

1.  Open your web browser and go to `http://localhost:5050`.
2.  Log in using the credentials from your `.env` file (`PGADMIN_DEFAULT_EMAIL` and `PGADMIN_DEFAULT_PASSWORD`).
3.  If you haven't already, add a new server connection to your PostgreSQL database:
    *   **General Tab:** Name (e.g., `local_postgres`)
    *   **Connection Tab:**
        *   Host name/address: `postgres_db` (This is the Docker service name)
        *   Port: `5432`
        *   Maintenance database: `mydatabase`
        *   Username: `admin`
        *   Password: `password`
    *   Click "Save".
4.  Navigate through the server to `Databases > mydatabase > Schemas > public > Tables` to find the `financial_data` table. You can right-click and "View/Edit Data" to see the inserted records.

## Testing

A unit test script `test.py` is provided to ensure the database integration works as expected.

To run the tests:

```bash
python db/test.py
```

The test will perform the following:
1.  Connect to the database.
2.  Create the `financial_data` table (dropping it first if it exists).
3.  Read a small sample of data from `Synthetic_Financial_datasets_log.csv`.
4.  Add a `probability` column to this sample.
5.  Insert the sample into the `financial_data` table.
6.  Retrieve the inserted data and assert its correctness.

A successful test run will output `OK`.
