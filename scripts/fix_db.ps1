<#
Idempotent helper to ensure the `fraud` database and `all_transactions`
table exist in the Postgres container. Usage:
  .\scripts\fix_db.ps1 [container-name]

When run without args, the script attempts to find a container with
`fraud-db` in its name.
#>
param(
  [string]$ContainerName
)

if (-not $ContainerName) {
  $ContainerName = docker ps --filter "name=fraud-db" --format "{{.Names}}" | Select-Object -First 1
}

if (-not $ContainerName) {
  Write-Error "No fraud-db container found. Provide container name as argument."
  docker ps --format "table {{.Names}}	{{.Image}}	{{.Status}}"
  exit 1
}

Write-Host "Using container: $ContainerName"

function Invoke-Psql($db, $sql) {
  docker exec -i $ContainerName psql -U fraud -d $db -v ON_ERROR_STOP=1 -c $sql
}

# Check if database 'fraud' exists (connect to template1)
$exists = docker exec -i $ContainerName psql -U fraud -d template1 -tAc "SELECT 1 FROM pg_database WHERE datname='fraud';" 2>$null
if ($exists.Trim() -ne '1') {
  Write-Host "Creating database 'fraud'..."
  docker exec -i $ContainerName psql -U fraud -d template1 -c "CREATE DATABASE fraud;"
} else {
  Write-Host "Database 'fraud' already exists"
}

$ddl = @'
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
'@

Write-Host "Creating table in 'frauddb' (if missing)..."
Invoke-Psql frauddb $ddl

Write-Host "Creating table in 'fraud' (if missing)..."
Invoke-Psql fraud $ddl

Write-Host "Done. Verify with: docker exec -it $ContainerName psql -U fraud -d frauddb -c '\\dt all_transactions'"
