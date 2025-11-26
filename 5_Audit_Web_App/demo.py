"""
Demo script for the Fraud Detection Audit Web Application

This script demonstrates the key features and functionality of the audit web app
by populating the database with realistic sample data and showing example usage.
"""

import os
import sys
try:
    import psycopg2
    import psycopg2.extras
    DB_AVAILABLE = True
except ImportError:
    print("Warning: psycopg2 not available. Database features will be disabled.")
    DB_AVAILABLE = False
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from db.db import connect_to_db

def create_sample_data():
    """Create comprehensive sample data for demonstration."""
    
    # Sample transaction types and patterns
    transaction_types = ['CASH_OUT', 'PAYMENT', 'TRANSFER', 'CASH_IN', 'DEBIT']
    
    # Create diverse sample data
    sample_transactions = []
    
    # True Positives (Correctly identified fraud)
    for i in range(15):
        sample_transactions.append({
            'step': i + 1,
            'type': np.random.choice(['CASH_OUT', 'TRANSFER']),
            'amount': np.random.uniform(10000, 50000),  # High amounts
            'nameOrig': f'C{np.random.randint(1000000, 9999999)}',
            'oldbalanceOrg': np.random.uniform(10000, 100000),
            'newbalanceOrig': 0,  # Account emptied - suspicious
            'nameDest': f'C{np.random.randint(1000000, 9999999)}',
            'oldbalanceDest': np.random.uniform(0, 10000),
            'newbalanceDest': np.random.uniform(10000, 60000),
            'isFraud': 1,
            'isFlaggedFraud': 1,
            'probability': np.random.uniform(0.7, 0.98),  # High confidence
            'manual_status': False,
            'manual_label': None
        })
    
    # True Negatives (Correctly identified as legitimate)
    for i in range(30):
        sample_transactions.append({
            'step': i + 16,
            'type': np.random.choice(['PAYMENT', 'CASH_IN', 'DEBIT']),
            'amount': np.random.uniform(10, 5000),  # Normal amounts
            'nameOrig': f'C{np.random.randint(1000000, 9999999)}',
            'oldbalanceOrg': np.random.uniform(1000, 50000),
            'newbalanceOrig': np.random.uniform(500, 48000),
            'nameDest': f'M{np.random.randint(1000000, 9999999)}',  # Merchant
            'oldbalanceDest': np.random.uniform(0, 10000),
            'newbalanceDest': np.random.uniform(100, 15000),
            'isFraud': 0,
            'isFlaggedFraud': 0,
            'probability': np.random.uniform(0.01, 0.3),  # Low confidence
            'manual_status': False,
            'manual_label': None
        })
    
    # False Positives (Incorrectly flagged as fraud)
    for i in range(8):
        sample_transactions.append({
            'step': i + 46,
            'type': np.random.choice(['CASH_OUT', 'TRANSFER']),
            'amount': np.random.uniform(5000, 15000),  # Medium-high amounts
            'nameOrig': f'C{np.random.randint(1000000, 9999999)}',
            'oldbalanceOrg': np.random.uniform(5000, 20000),
            'newbalanceOrig': np.random.uniform(0, 5000),
            'nameDest': f'C{np.random.randint(1000000, 9999999)}',
            'oldbalanceDest': np.random.uniform(0, 5000),
            'newbalanceDest': np.random.uniform(5000, 20000),
            'isFraud': 0,  # Actually legitimate
            'isFlaggedFraud': 0,
            'probability': np.random.uniform(0.6, 0.9),  # But model thinks fraud
            'manual_status': False,
            'manual_label': None
        })
    
    # False Negatives (Missed fraud cases)
    for i in range(5):
        sample_transactions.append({
            'step': i + 54,
            'type': np.random.choice(['PAYMENT', 'TRANSFER']),
            'amount': np.random.uniform(1000, 8000),  # Lower amounts
            'nameOrig': f'C{np.random.randint(1000000, 9999999)}',
            'oldbalanceOrg': np.random.uniform(2000, 10000),
            'newbalanceOrig': np.random.uniform(0, 2000),
            'nameDest': f'C{np.random.randint(1000000, 9999999)}',
            'oldbalanceDest': np.random.uniform(0, 1000),
            'newbalanceDest': np.random.uniform(1000, 9000),
            'isFraud': 1,  # Actually fraud
            'isFlaggedFraud': 0,
            'probability': np.random.uniform(0.1, 0.4),  # But model missed it
            'manual_status': False,
            'manual_label': None
        })
    
    # Some manually reviewed cases
    for i in range(7):
        tx = sample_transactions[i]
        tx['manual_status'] = True
        tx['manual_label'] = tx['isFraud']  # Correct manual label
    
    return sample_transactions

def populate_database():
    """Populate database with sample data."""
    if not DB_AVAILABLE:
        print("❌ Database not available. Please install psycopg2-binary")
        return False
    
    conn = connect_to_db()
    if not conn:
        print("❌ Failed to connect to database")
        return False
    
    try:
        # Clear existing data (optional)
        with conn.cursor() as cur:
            cur.execute("DELETE FROM financial_data")
            print("🗑️  Cleared existing data")
        
        # Insert sample data
        sample_data = create_sample_data()
        
        insert_query = """
        INSERT INTO financial_data (
            step, type, amount, "nameOrig", "oldbalanceOrg", "newbalanceOrig",
            "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", 
            "isFlaggedFraud", probability, manual_status, manual_label
        ) VALUES (
            %(step)s, %(type)s, %(amount)s, %(nameOrig)s, %(oldbalanceOrg)s, 
            %(newbalanceOrig)s, %(nameDest)s, %(oldbalanceDest)s, %(newbalanceDest)s,
            %(isFraud)s, %(isFlaggedFraud)s, %(probability)s, %(manual_status)s, %(manual_label)s
        )
        """
        
        with conn.cursor() as cur:
            cur.executemany(insert_query, sample_data)
            
        conn.commit()
        print(f"✅ Inserted {len(sample_data)} sample transactions")
        
        # Show summary
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN "isFraud" = 1 THEN 1 ELSE 0 END) as actual_fraud,
                    SUM(CASE WHEN probability >= 0.5 THEN 1 ELSE 0 END) as predicted_fraud,
                    SUM(CASE WHEN manual_status = true THEN 1 ELSE 0 END) as manually_reviewed
                FROM financial_data
            """)
            
            stats = cur.fetchone()
            
        print(f"""
📊 Sample Data Summary:
   Total Transactions: {stats[0]}
   Actual Fraud Cases: {stats[1]}
   Predicted Fraud Cases: {stats[2]}
   Manually Reviewed: {stats[3]}
   Pending Review: {stats[0] - stats[3]}
        """)
        
        return True
        
    except Exception as e:
        print(f"❌ Error populating database: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def show_demo_guide():
    """Show demonstration guide."""
    print("""
🎯 Demo Guide for Fraud Detection Audit Web App
===============================================

The sample data has been loaded! Here's what you can explore:

📈 DASHBOARD (/dashboard):
   • View accuracy metrics (F1, Precision, Recall, Accuracy)
   • See confusion matrix visualization
   • Check summary statistics
   • Preview top false cases

❌ FALSE CASES (/false_cases):
   • Review False Positives (8 cases) - legitimate transactions flagged as fraud
   • Review False Negatives (5 cases) - fraud cases missed by model
   • Use quick action buttons to label transactions
   • Analyze transaction patterns

🏷️ MANUAL LABELING (/manual_labeling):
   • Review transactions pending manual validation
   • Use keyboard shortcuts: 'F' for fraud, 'L' for legitimate
   • View detailed transaction analysis with risk factors
   • Track labeling progress

🔍 Key Features to Test:
   • Auto-refreshing metrics (every 30 seconds)
   • Interactive confusion matrix chart
   • Transaction detail modals
   • Risk factor analysis
   • Responsive design on different screen sizes

📝 Suggested Testing Flow:
   1. Start at Dashboard to see overall metrics
   2. Navigate to False Cases to review prediction errors
   3. Label some transactions using Manual Labeling
   4. Return to Dashboard to see updated metrics
   5. Try keyboard shortcuts for faster labeling

💡 Tips:
   • The sample data includes realistic fraud patterns
   • Manual labels will update the metrics in real-time
   • Use the analysis modal for detailed transaction insights
   • Check different transaction types and amounts

Start the web app with: python app.py
Then visit: http://localhost:5000
    """)

def main():
    """Main demo function."""
    print("🎭 Fraud Detection Audit Web App Demo")
    print("=" * 50)
    
    # Load environment
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(dotenv_path=dotenv_path)
    
    print("🔧 Setting up demonstration data...")
    
    if populate_database():
        print("✅ Demo data setup completed!")
        show_demo_guide()
        
        response = input("\n🚀 Start the web application now? (y/n): ")
        if response.lower() == 'y':
            os.system("python app.py")
        else:
            print("\n📝 To start the application: python app.py")
            print("   Then visit: http://localhost:5000")
    else:
        print("❌ Demo setup failed!")
        print("   Please check database connection and try again")

if __name__ == "__main__":
    main()