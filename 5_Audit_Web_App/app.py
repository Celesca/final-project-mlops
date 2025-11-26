"""
Fraud Detection Audit Web Application

A Flask web application for auditing fraud detection model predictions.
Provides dashboards for accuracy metrics, false cases review, and manual labeling.
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
try:
    import psycopg2
    import psycopg2.extras
    DB_AVAILABLE = True
except ImportError:
    print("Warning: psycopg2 not available. Database features will be disabled.")
    DB_AVAILABLE = False
import os
import sys
from datetime import datetime
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from dotenv import load_dotenv

# Add parent directory to path for importing db functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from db.db import connect_to_db

app = Flask(__name__)
app.secret_key = 'fraud_audit_secret_key_2025'

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

def get_db_connection():
    """Get database connection for the web app."""
    if not DB_AVAILABLE:
        print("Database not available. Please install psycopg2-binary")
        return None
    return connect_to_db()

def calculate_metrics():
    """Calculate accuracy metrics from the database."""
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        query = """
        SELECT 
            "isFraud" as actual_label,
            CASE 
                WHEN probability >= 0.5 THEN 1 
                ELSE 0 
            END as predicted_label,
            probability,
            manual_label,
            manual_status
        FROM financial_data
        WHERE "isFraud" IS NOT NULL 
        AND probability IS NOT NULL
        """
        
        df = pd.read_sql_query(query, conn)
        
        if len(df) == 0:
            return None
        
        # Calculate basic metrics
        y_true = df['actual_label']
        y_pred = df['predicted_label']
        
        metrics = {
            'total_transactions': len(df),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        }
        
        # Calculate additional stats
        metrics['true_fraud_count'] = int(sum(y_true))
        metrics['predicted_fraud_count'] = int(sum(y_pred))
        metrics['manual_reviewed_count'] = int(sum(df['manual_status'] == True))
        metrics['pending_review_count'] = int(sum(df['manual_status'] == False))
        
        return metrics
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None
    finally:
        conn.close()

def get_false_cases():
    """Get false positive and false negative cases."""
    conn = get_db_connection()
    if not conn:
        return [], []
    
    try:
        # False Positives: Model predicted fraud (prob >= 0.5) but actual is not fraud (isFraud = 0)
        fp_query = """
        SELECT id, step, type, amount, "nameOrig", "oldbalanceOrg", "newbalanceOrig",
               "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", 
               "isFlaggedFraud", probability, manual_status, manual_label
        FROM financial_data
        WHERE "isFraud" = 0 AND probability >= 0.5
        ORDER BY probability DESC
        LIMIT 50
        """
        
        # False Negatives: Model predicted not fraud (prob < 0.5) but actual is fraud (isFraud = 1)
        fn_query = """
        SELECT id, step, type, amount, "nameOrig", "oldbalanceOrg", "newbalanceOrig",
               "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", 
               "isFlaggedFraud", probability, manual_status, manual_label
        FROM financial_data
        WHERE "isFraud" = 1 AND probability < 0.5
        ORDER BY probability ASC
        LIMIT 50
        """
        
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(fp_query)
            false_positives = cur.fetchall()
            
            cur.execute(fn_query)
            false_negatives = cur.fetchall()
        
        return false_positives, false_negatives
        
    except Exception as e:
        print(f"Error getting false cases: {e}")
        return [], []
    finally:
        conn.close()

def get_unlabeled_transactions(limit=20):
    """Get transactions that haven't been manually reviewed."""
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        query = """
        SELECT id, step, type, amount, "nameOrig", "oldbalanceOrg", "newbalanceOrig",
               "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", 
               "isFlaggedFraud", probability, manual_status, manual_label
        FROM financial_data
        WHERE manual_status = FALSE OR manual_status IS NULL
        ORDER BY probability DESC
        LIMIT %s
        """
        
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, (limit,))
            transactions = cur.fetchall()
        
        return transactions
        
    except Exception as e:
        print(f"Error getting unlabeled transactions: {e}")
        return []
    finally:
        conn.close()

@app.route('/')
def index():
    """Main dashboard page."""
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    """Dashboard with accuracy metrics."""
    metrics = calculate_metrics()
    false_positives, false_negatives = get_false_cases()
    
    return render_template('dashboard.html', 
                         metrics=metrics, 
                         false_positives=false_positives[:10],  # Show top 10
                         false_negatives=false_negatives[:10])

@app.route('/false_cases')
def false_cases():
    """Page showing all false positive and false negative cases."""
    false_positives, false_negatives = get_false_cases()
    
    return render_template('false_cases.html',
                         false_positives=false_positives,
                         false_negatives=false_negatives)

@app.route('/manual_labeling')
def manual_labeling():
    """Manual labeling interface."""
    transactions = get_unlabeled_transactions(20)
    return render_template('manual_labeling.html', transactions=transactions)

@app.route('/label_transaction', methods=['POST'])
def label_transaction():
    """API endpoint to save manual labels."""
    try:
        transaction_id = request.form.get('transaction_id')
        manual_label = request.form.get('manual_label')  # '0' for not fraud, '1' for fraud
        
        if not transaction_id or manual_label not in ['0', '1']:
            flash('Invalid data provided', 'error')
            return redirect(url_for('manual_labeling'))
        
        conn = get_db_connection()
        if not conn:
            flash('Database connection error', 'error')
            return redirect(url_for('manual_labeling'))
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE financial_data 
                    SET manual_status = TRUE, manual_label = %s
                    WHERE id = %s
                """, (int(manual_label), int(transaction_id)))
                
                conn.commit()
                flash('Transaction labeled successfully!', 'success')
                
        except Exception as e:
            print(f"Error updating label: {e}")
            flash('Error saving label', 'error')
            conn.rollback()
        finally:
            conn.close()
            
    except Exception as e:
        print(f"Error in label_transaction: {e}")
        flash('Error processing request', 'error')
    
    return redirect(url_for('manual_labeling'))

@app.route('/api/metrics')
def api_metrics():
    """API endpoint to get current metrics."""
    metrics = calculate_metrics()
    if metrics:
        return jsonify(metrics)
    else:
        return jsonify({'error': 'No data available'}), 404

@app.route('/api/transaction/<int:transaction_id>')
def api_transaction_details(transaction_id):
    """API endpoint to get transaction details."""
    conn = get_db_connection()
    if not conn:
        return jsonify({'error': 'Database connection error'}), 500
    
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM financial_data WHERE id = %s
            """, (transaction_id,))
            
            transaction = cur.fetchone()
            
            if transaction:
                return jsonify(dict(transaction))
            else:
                return jsonify({'error': 'Transaction not found'}), 404
                
    except Exception as e:
        print(f"Error getting transaction details: {e}")
        return jsonify({'error': 'Database error'}), 500
    finally:
        conn.close()

if __name__ == '__main__':
    # Run the application
    print("Starting Fraud Detection Audit Web App...")
    print("Dashboard will be available at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)