#!/usr/bin/env bash
set -euo pipefail

# Helper to idempotently ensure the `fraud` database and
# `all_transactions` table exist inside the Postgres container.
# Usage: ./scripts/fix_db.sh [container-name]

CONTAINER=${1:-$(docker ps --filter "name=fraud-db" --format '{{.Names}}' | head -n1)}

if [ -z "$CONTAINER" ]; then
  echo "No container found for service 'fraud-db'. Provide container name as first arg." >&2
  docker ps --format 'table {{.Names}}	{{.Image}}	{{.Status}}'
  exit 1
fi

echo "Using container: $CONTAINER"

# helper to run psql as the 'fraud' user
psql_exec() {
  local db="$1"; shift
  docker exec -i "$CONTAINER" psql -U fraud -d "$db" -v ON_ERROR_STOP=1 -c "$*"
}

# 1) Ensure database 'fraud' exists. Connect to template1 (always present).
exists=$(docker exec -i "$CONTAINER" psql -U fraud -d template1 -tAc "SELECT 1 FROM pg_database WHERE datname='fraud';" || true)
if [ "${exists}" != "1" ]; then
  echo "Creating database 'fraud'..."
  docker exec -i "$CONTAINER" psql -U fraud -d template1 -c "CREATE DATABASE fraud;"
else
  echo "Database 'fraud' already exists"
fi

read -r -d '' ALL_TX_DDL <<'SQL'
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
SQL

echo "Creating table in 'frauddb' (if missing)..."
psql_exec frauddb "$ALL_TX_DDL"

echo "Creating table in 'fraud' (if missing)..."
psql_exec fraud "$ALL_TX_DDL"

echo "Done. You can verify with: docker exec -it $CONTAINER psql -U fraud -d frauddb -c '\\dt all_transactions'"
