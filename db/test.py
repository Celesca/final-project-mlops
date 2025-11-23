import unittest
import os
import pandas as pd
import numpy as np
import psycopg2

# Import functions from the script we want to test
from db import connect_to_db, create_financial_data_table, insert_dataframe

class TestDatabaseIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Set up the database connection and create the table once for all tests.
        """
        cls.conn = connect_to_db()
        if cls.conn is None:
            raise ConnectionError("Could not connect to the database. Make sure the DB is running.")
        
        # Define a sample dataframe structure to create the table
        # This ensures the test doesn't fail if the CSV is missing,
        # but the core DB logic can still be tested.
        cls.sample_df_structure = pd.DataFrame({
            'step': [1], 'type': ['PAYMENT'], 'amount': [0.0], 'nameOrig': ['C0'],
            'oldbalanceOrg': [0.0], 'newbalanceOrig': [0.0], 'nameDest': ['M0'],
            'oldbalanceDest': [0.0], 'newbalanceDest': [0.0], 'isFraud': [0],
            'isFlaggedFraud': [0], 'probability': [0.5]
        })
        create_financial_data_table(cls.conn, cls.sample_df_structure)


    @classmethod
    def tearDownClass(cls):
        """
        Close the database connection after all tests are done.
        """
        if cls.conn:
            cls.conn.close()
            print("\nTest database connection closed.")

    def test_insert_and_retrieve_sample_data(self):
        """
        Tests inserting a sample of the CSV data and retrieving it to verify correctness.
        """
        # 1. Load data from CSV and take a sample
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'Synthetic_Financial_datasets_log.csv')
        try:
            full_df = pd.read_csv(csv_path)
        except FileNotFoundError:
            self.fail(f"Test failed: CSV file not found at {csv_path}")

        sample_size = 5
        test_df = full_df.sample(n=sample_size)
        
        # 2. Add 'probability' column
        test_df['probability'] = np.random.rand(len(test_df))
        test_df = test_df.reset_index(drop=True) # Reset index for easier comparison

        # 3. Insert the sample dataframe into the database
        insert_dataframe(self.conn, test_df, 'financial_data')

        # 4. Retrieve the data from the database to verify
        with self.conn.cursor() as cur:
            # The 'id' column is auto-generated, so we query for the other columns
            # Ordering by a unique column to ensure consistency
            db_cols = '", "'.join(test_df.columns)
            cur.execute(f'SELECT "{db_cols}" FROM financial_data ORDER BY "nameOrig" DESC LIMIT {sample_size}')
            retrieved_data = cur.fetchall()
        
        # 5. Perform assertions
        self.assertEqual(len(retrieved_data), sample_size, "Number of retrieved rows should match inserted rows.")

        # Convert retrieved data to a DataFrame for easier comparison
        retrieved_df = pd.DataFrame(retrieved_data, columns=test_df.columns).sort_values(by="nameOrig", ascending=False).reset_index(drop=True)
        
        # Sort original test_df for consistent comparison
        test_df_sorted = test_df.sort_values(by="nameOrig", ascending=False).reset_index(drop=True)

        # Check a few values to ensure data integrity
        # Comparing floating point numbers requires checking for closeness, not exact equality
        self.assertAlmostEqual(retrieved_df['amount'][0], test_df_sorted['amount'][0], places=2)
        self.assertEqual(retrieved_df['type'][0], test_df_sorted['type'][0])
        self.assertAlmostEqual(retrieved_df['probability'][0], test_df_sorted['probability'][0], places=5)
        
        print("\nTest `test_insert_and_retrieve_sample_data` passed successfully.")


if __name__ == '__main__':
    # This allows running the test script directly
    unittest.main()
