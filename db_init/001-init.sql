-- Initialize databases and tables for fraud project
-- This file is executed by the postgres docker-entrypoint using psql -f

-- Create alternate database `fraud` if it doesn't exist
CREATE DATABASE fraud;

-- Connect to the newly created `fraud` database and create table
\connect fraud

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

-- Connect back to the primary project DB (frauddb) and ensure the same table exists there
\connect frauddb

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

-- End of init script
