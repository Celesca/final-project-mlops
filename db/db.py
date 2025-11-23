import os
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

# --- Load Environment Variables ---
# Load environment variables from the .env file located in the parent directory
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

# --- Database Connection Details ---
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

def connect_to_db():
    """Establishes and returns a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
        )
        print("Database connection established.")
        return conn
    except psycopg2.OperationalError as e:
        print(f"Could not connect to the database: {e}")
        return None

def create_financial_data_table(conn, df):
    """
    Creates a table based on the DataFrame's columns and data types,
    including the new 'probability' column.
    """
    # A mapping from pandas dtype to PostgreSQL dtype
    dtype_mapping = {
        'int64': 'INTEGER',
        'float64': 'FLOAT',
        'object': 'VARCHAR',
        'datetime64[ns]': 'TIMESTAMP'
    }
    
    # Start the CREATE TABLE statement
    # Using 'SERIAL PRIMARY KEY' for the first column assuming it's an ID
    cols = list(df.columns)
    table_name = "financial_data"
    
    # Drop table if it exists to avoid errors on re-run
    drop_sql = f"DROP TABLE IF EXISTS {table_name};"

    # Build the columns part of the SQL statement
    sql_cols = []
    for col_name in cols:
        dtype = str(df[col_name].dtype)
        pg_dtype = dtype_mapping.get(dtype, 'VARCHAR') # Default to VARCHAR
        sql_cols.append(f'"{col_name}" {pg_dtype}')
    
    create_sql = f"CREATE TABLE {table_name} (id SERIAL PRIMARY KEY, {', '.join(sql_cols)});"

    try:
        with conn.cursor() as cur:
            print(f"Dropping table '{table_name}' if it exists...")
            cur.execute(drop_sql)
            print(f"Creating table '{table_name}'...")
            cur.execute(create_sql)
            conn.commit()
            print("Table 'financial_data' created successfully.")
    except psycopg2.Error as e:
        print(f"Error creating table: {e}")
        conn.rollback()

def insert_dataframe(conn, df, table_name):
    """
    Efficiently inserts a pandas DataFrame into a PostgreSQL table.
    
    Args:
        conn: The database connection.
        df: The DataFrame to insert.
        table_name: The name of the target table.
    """
    if df.empty:
        print("DataFrame is empty, no data to insert.")
        return

    # Prepare data for insertion
    cols = '", "'.join(df.columns)
    values = [tuple(row) for row in df.itertuples(index=False)]
    
    insert_sql = f'INSERT INTO {table_name} ("{cols}") VALUES %s'
    
    try:
        with conn.cursor() as cur:
            print(f"Inserting {len(df)} rows into '{table_name}'...")
            execute_values(cur, insert_sql, values)
            conn.commit()
            print("Data inserted successfully.")
    except psycopg2.Error as e:
        print(f"Error inserting data: {e}")
        conn.rollback()

if __name__ == "__main__":
    # --- Main Execution ---
    
    # 1. Load data from CSV
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'Synthetic_Financial_datasets_log.csv')
    try:
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        print("CSV data loaded into DataFrame.")
    except FileNotFoundError:
        print(f"Error: The file was not found at {csv_path}.")
        print("Please ensure the file 'Synthetic_Financial_datasets_log.csv' is in the 'dataset' directory.")
        exit()

    # 2. Add 'probability' column
    print("Adding 'probability' column with random values...")
    df['probability'] = np.random.rand(len(df))

    # 3. Connect to database and perform operations
    conn = connect_to_db()
    if conn:
        # Create table based on DataFrame structure
        create_financial_data_table(conn, df)
        
        # Insert DataFrame into the new table
        insert_dataframe(conn, df, 'financial_data')

        # Close the connection
        conn.close()
        print("Database connection closed.")