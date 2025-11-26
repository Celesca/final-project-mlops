#!/bin/bash
set -euo pipefail

# Initialization script for fraud-db service.
# This script will run inside the Postgres container at first startup.
# It creates a second database named `fraud` (some clients expect it)
# and ensures the `all_transactions` table exists in both `frauddb` and `fraud`.

# Use the POSTGRES_USER and POSTGRES_DB environment variables provided
# by the container entrypoint.

psql_cmd() {
  local dbname="$1"
  shift
  psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$dbname" "$@"
}

# Create the "fraud" database if it doesn't exist. (The user "$POSTGRES_USER" is created by the image.)
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-'EOSQL'
CREATE DATABASE IF NOT EXISTS fraud;
EOSQL

# Define the DDL for the all_transactions table used by the project.
read -r -d '' ALL_TX_DDL <<'EOSQL'
CREATE TABLE IF NOT EXISTS all_transactions (
  id serial PRIMARY KEY,
  step integer,
  type text,
  amount double precision,
  "nameOrig" text,
  "oldbalanceOrg" double precision,
  "newbalanceOrig" double precision,
  "nameDest" text,
  "oldbalanceDest" double precision,
  "newbalanceDest" double precision,
  "isFraud" boolean,
  "isFlaggedFraud" boolean,
  ingest_date timestamp with time zone DEFAULT now(),
  source_file text,
  created_at timestamp with time zone DEFAULT now()
);
EOSQL

# Create table in the default database created by env (POSTGRES_DB)
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
${ALL_TX_DDL}
EOSQL

# Also create table in the alternate `fraud` database (some clients attempt to connect there)
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname fraud <<-EOSQL
${ALL_TX_DDL}
EOSQL

echo "fraud-db init script completed. Created databases/tables if needed."

docker exec -it fraud-db-1 psql -U fraud -d frauddb -c "CREATE TABLE IF NOT EXISTS all_transactions (id serial PRIMARY KEY, step integer, type text, amount double precision, \"nameOrig\" text, \"oldbalanceOrg\" double precision, \"newbalanceOrig\" double precision, \"nameDest\" text, \"oldbalanceDest\" double precision, \"newbalanceDest\" double precision, \"isFraud\" boolean, \"isFlaggedFraud\" boolean, ingest_date timestamptz DEFAULT now(), source_file text, created_at timestamptz DEFAULT now());"

docker exec -it final-project-mlops-fraud-db-1 psql -U fraud -d postgres -c "CREATE DATABASE fraud;"

docker exec -it final-project-mlops-fraud-db-1 psql -U fraud -d frauddb -c "CREATE TABLE IF NOT EXISTS all_transactions (id serial PRIMARY KEY, step integer, type text, amount double precision, \"nameOrig\" text, \"oldbalanceOrg\" double precision, \"newbalanceOrig\" double precision, \"nameDest\" text, \"oldbalanceDest\" double precision, \"newbalanceDest\" double precision, \"isFraud\" boolean, \"isFlaggedFraud\" boolean, ingest_date timestamptz DEFAULT now(), source_file text, created_at timestamptz DEFAULT now());"

docker exec -it final-project-mlops-fraud-db-1 psql -U fraud -d fraud -c "CREATE TABLE IF NOT EXISTS all_transactions (id serial PRIMARY KEY, step integer, type text, amount double precision, \"nameOrig\" text, \"oldbalanceOrg\" double precision, \"newbalanceOrig\" double precision, \"nameDest\" text, \"oldbalanceDest\" double precision, \"newbalanceDest\" double precision, \"isFraud\" boolean, \"isFlaggedFraud\" boolean, ingest_date timestamptz DEFAULT now(), source_file text, created_at timestamptz DEFAULT now());"