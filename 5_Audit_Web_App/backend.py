from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json

app = Flask(__name__)
app.secret_key = 'fraud_audit_secret_key_2025'

# Global variables to store data
transactions_df = None
manual_labels = {}  # Store manual labels: {transaction_id: label}
manual_status = {}  # Store manual status: {transaction_id: True/False}

# Add context processor for templates
@app.context_processor
def inject_current_date():
    return {'current_date': datetime.now().strftime('%b %d, %Y')}

def load_sample_data():
    """Load sample data from CSV or create mock data."""
    global transactions_df
    
    try:
        # Try to load from the dataset directory
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'Synthetic_Financial_datasets_log.csv')
        if os.path.exists(csv_path):
            print(f"Loading data from {csv_path}")
            df = pd.read_csv(csv_path)
            # Take a sample for demo
            df = df.sample(n=min(100, len(df))).reset_index(drop=True)
        else:
            print("CSV not found, creating mock data")
            df = create_mock_data()
        
        # Add probability column if not exists
        if 'probability' not in df.columns:
            df['probability'] = np.random.rand(len(df))
            # Make some high probability for fraud cases
            fraud_mask = df['isFraud'] == 1
            df.loc[fraud_mask, 'probability'] = np.random.uniform(0.6, 0.95, fraud_mask.sum())
            # Make some low probability for non-fraud cases
            non_fraud_mask = df['isFraud'] == 0
            df.loc[non_fraud_mask, 'probability'] = np.random.uniform(0.05, 0.4, non_fraud_mask.sum())
        
        # Add ID column if not exists
        if 'id' not in df.columns:
            df.insert(0, 'id', range(1, len(df) + 1))
        
        transactions_df = df
        print(f"Loaded {len(df)} transactions")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        transactions_df = create_mock_data()

def create_mock_data():
    """Create mock transaction data for testing."""
    np.random.seed(42)  # For reproducible results
    
    n_samples = 80
    
    data = {
        'id': range(1, n_samples + 1),
        'step': np.random.randint(1, 100, n_samples),
        'type': np.random.choice(['CASH_OUT', 'PAYMENT', 'TRANSFER', 'CASH_IN', 'DEBIT'], n_samples),
        'amount': np.random.lognormal(8, 1.5, n_samples),
        'nameOrig': [f'C{np.random.randint(1000000, 9999999)}' for _ in range(n_samples)],
        'oldbalanceOrg': np.random.lognormal(9, 1.2, n_samples),
        'newbalanceOrig': np.random.lognormal(8.5, 1.3, n_samples),
        'nameDest': [f'{"M" if np.random.random() < 0.3 else "C"}{np.random.randint(1000000, 9999999)}' for _ in range(n_samples)],
        'oldbalanceDest': np.random.lognormal(8, 1.5, n_samples),
        'newbalanceDest': np.random.lognormal(8.2, 1.4, n_samples),
        'isFraud': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),  # 15% fraud
        'isFlaggedFraud': np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    }
    
    df = pd.DataFrame(data)
    
    # Add probability column based on fraud status with some noise
    df['probability'] = np.where(
        df['isFraud'] == 1,
        np.random.uniform(0.4, 0.95, n_samples),  # Higher prob for fraud
        np.random.uniform(0.02, 0.6, n_samples)   # Lower prob for non-fraud
    )
    
    # Create some false positives and negatives
    false_positive_mask = (df['isFraud'] == 0) & (np.random.random(n_samples) < 0.1)
    df.loc[false_positive_mask, 'probability'] = np.random.uniform(0.6, 0.9, false_positive_mask.sum())
    
    false_negative_mask = (df['isFraud'] == 1) & (np.random.random(n_samples) < 0.15)
    df.loc[false_negative_mask, 'probability'] = np.random.uniform(0.1, 0.4, false_negative_mask.sum())
    
    return df

def get_current_metrics():
    """Calculate current metrics including manual labels."""
    if transactions_df is None:
        return None
    
    df = transactions_df.copy()
    
    # Use manual labels where available, otherwise use original labels
    actual_labels = []
    predicted_labels = []
    
    for idx, row in df.iterrows():
        transaction_id = row['id']
        
        # Use manual label if available, otherwise use original
        if transaction_id in manual_labels:
            actual_label = manual_labels[transaction_id]
        else:
            actual_label = row['isFraud']
        
        # Model prediction based on probability threshold
        predicted_label = 1 if row['probability'] >= 0.5 else 0
        
        actual_labels.append(actual_label)
        predicted_labels.append(predicted_label)
    
    # Calculate metrics
    metrics = {
        'total_transactions': len(df),
        'accuracy': accuracy_score(actual_labels, predicted_labels),
        'precision': precision_score(actual_labels, predicted_labels, zero_division=0),
        'recall': recall_score(actual_labels, predicted_labels, zero_division=0),
        'f1_score': f1_score(actual_labels, predicted_labels, zero_division=0),
        'confusion_matrix': confusion_matrix(actual_labels, predicted_labels).tolist(),
        'true_fraud_count': sum(actual_labels),
        'predicted_fraud_count': sum(predicted_labels),
        'manual_reviewed_count': len(manual_labels),
        'pending_review_count': len(df) - len(manual_labels)
    }
    
    return metrics

def get_false_cases():
    """Get false positive and false negative cases."""
    if transactions_df is None:
        return [], []
    
    df = transactions_df.copy()
    
    false_positives = []
    false_negatives = []
    
    for idx, row in df.iterrows():
        transaction_id = row['id']
        
        # Use manual label if available
        if transaction_id in manual_labels:
            actual_label = manual_labels[transaction_id]
        else:
            actual_label = row['isFraud']
        
        predicted_label = 1 if row['probability'] >= 0.5 else 0
        
        # Check for false positives
        if actual_label == 0 and predicted_label == 1:
            row_dict = row.to_dict()
            row_dict['manual_status'] = transaction_id in manual_status
            row_dict['manual_label'] = manual_labels.get(transaction_id)
            false_positives.append(row_dict)
        
        # Check for false negatives
        elif actual_label == 1 and predicted_label == 0:
            row_dict = row.to_dict()
            row_dict['manual_status'] = transaction_id in manual_status
            row_dict['manual_label'] = manual_labels.get(transaction_id)
            false_negatives.append(row_dict)
    
    # Sort by probability
    false_positives.sort(key=lambda x: x['probability'], reverse=True)
    false_negatives.sort(key=lambda x: x['probability'])
    
    return false_positives[:50], false_negatives[:50]

def get_unlabeled_transactions(limit=20):
    """Get transactions that haven't been manually labeled."""
    if transactions_df is None:
        return []
    
    df = transactions_df.copy()
    unlabeled = []
    
    for idx, row in df.iterrows():
        transaction_id = row['id']
        if transaction_id not in manual_status:
            row_dict = row.to_dict()
            row_dict['manual_status'] = False
            row_dict['manual_label'] = None
            unlabeled.append(row_dict)
    
    # Sort by probability (highest first for more interesting cases)
    unlabeled.sort(key=lambda x: x['probability'], reverse=True)
    
    return unlabeled[:limit]

@app.route('/')
def index():
    """Main dashboard page."""
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    """Dashboard with accuracy metrics."""
    metrics = get_current_metrics()
    false_positives, false_negatives = get_false_cases()
    
    return render_template('dashboard.html', 
                         metrics=metrics, 
                         false_positives=false_positives[:10],
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

@app.route('/api_test')
def api_test():
    """API testing interface."""
    return render_template('api_test.html')

@app.route('/label_transaction', methods=['POST'])
def label_transaction():
    """API endpoint to save manual labels."""
    try:
        transaction_id = int(request.form.get('transaction_id'))
        manual_label = int(request.form.get('manual_label'))  # 0 or 1
        
        if manual_label not in [0, 1]:
            flash('Invalid label provided', 'error')
            return redirect(url_for('manual_labeling'))
        
        # Store the manual label
        manual_labels[transaction_id] = manual_label
        manual_status[transaction_id] = True
        
        # Save to file for persistence
        save_manual_labels()
        
        flash('Transaction labeled successfully!', 'success')
        
    except Exception as e:
        print(f"Error in label_transaction: {e}")
        flash('Error processing request', 'error')
    
    return redirect(url_for('manual_labeling'))

@app.route('/api/metrics')
def api_metrics():
    """API endpoint to get current metrics."""
    metrics = get_current_metrics()
    if metrics:
        return jsonify(metrics)
    else:
        return jsonify({'error': 'No data available'}), 404

@app.route('/api/transaction/<int:transaction_id>')
def api_transaction_details(transaction_id):
    """API endpoint to get transaction details."""
    if transactions_df is None:
        return jsonify({'error': 'No data loaded'}), 500
    
    try:
        transaction_row = transactions_df[transactions_df['id'] == transaction_id]
        
        if len(transaction_row) == 0:
            return jsonify({'error': 'Transaction not found'}), 404
        
        transaction = transaction_row.iloc[0].to_dict()
        transaction['manual_status'] = transaction_id in manual_status
        transaction['manual_label'] = manual_labels.get(transaction_id)
        
        # Convert numpy types to Python types for JSON serialization
        for key, value in transaction.items():
            if isinstance(value, (np.integer, np.floating)):
                transaction[key] = float(value)
            elif isinstance(value, np.bool_):
                transaction[key] = bool(value)
        
        return jsonify(transaction)
        
    except Exception as e:
        print(f"Error getting transaction details: {e}")
        return jsonify({'error': 'Database error'}), 500

@app.route('/api/data/query')
def api_query_data():
    """Query data endpoint - returns paginated transaction data."""
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        filter_type = request.args.get('type', 'all')  # all, fraud, legitimate, pending
        
        if transactions_df is None:
            return jsonify({'error': 'No data loaded'}), 500
        
        df = transactions_df.copy()
        
        # Apply filters
        if filter_type == 'fraud':
            df = df[df['isFraud'] == 1]
        elif filter_type == 'legitimate':
            df = df[df['isFraud'] == 0]
        elif filter_type == 'pending':
            pending_ids = [tid for tid in df['id'] if tid not in manual_status]
            df = df[df['id'].isin(pending_ids)]
        
        # Pagination
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        page_data = df.iloc[start_idx:end_idx]
        
        # Convert to list of dicts and add manual info
        transactions = []
        for idx, row in page_data.iterrows():
            transaction = row.to_dict()
            transaction_id = transaction['id']
            transaction['manual_status'] = transaction_id in manual_status
            transaction['manual_label'] = manual_labels.get(transaction_id)
            
            # Convert numpy types
            for key, value in transaction.items():
                if isinstance(value, (np.integer, np.floating)):
                    transaction[key] = float(value)
                elif isinstance(value, np.bool_):
                    transaction[key] = bool(value)
            
            transactions.append(transaction)
        
        return jsonify({
            'transactions': transactions,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': len(df),
                'pages': (len(df) + per_page - 1) // per_page
            }
        })
        
    except Exception as e:
        print(f"Error querying data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/data/edit/<int:transaction_id>', methods=['PUT'])
def api_edit_transaction(transaction_id):
    """Edit transaction manual label."""
    try:
        data = request.get_json()
        manual_label = data.get('manual_label')
        
        if manual_label not in [0, 1, None]:
            return jsonify({'error': 'Invalid manual_label'}), 400
        
        if manual_label is not None:
            manual_labels[transaction_id] = manual_label
            manual_status[transaction_id] = True
        else:
            # Remove manual label
            manual_labels.pop(transaction_id, None)
            manual_status.pop(transaction_id, None)
        
        save_manual_labels()
        
        return jsonify({'success': True, 'message': 'Transaction updated'})
        
    except Exception as e:
        print(f"Error editing transaction: {e}")
        return jsonify({'error': str(e)}), 500

def save_manual_labels():
    """Save manual labels to file for persistence."""
    try:
        data = {
            'manual_labels': manual_labels,
            'manual_status': manual_status,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('manual_labels.json', 'w') as f:
            json.dump(data, f, indent=2)
            
    except Exception as e:
        print(f"Error saving manual labels: {e}")

def load_manual_labels():
    """Load manual labels from file."""
    global manual_labels, manual_status
    
    try:
        if os.path.exists('manual_labels.json'):
            with open('manual_labels.json', 'r') as f:
                data = json.load(f)
                manual_labels = {int(k): v for k, v in data.get('manual_labels', {}).items()}
                manual_status = {int(k): v for k, v in data.get('manual_status', {}).items()}
                print(f"Loaded {len(manual_labels)} manual labels")
    except Exception as e:
        print(f"Error loading manual labels: {e}")

if __name__ == '__main__':
    print("🔍 Starting Fraud Detection Audit Backend (CSV Mode)")
    print("=" * 60)
    
    # Load data and manual labels
    load_sample_data()
    load_manual_labels()
    
    # Print summary
    if transactions_df is not None:
        total_transactions = len(transactions_df)
        fraud_count = sum(transactions_df['isFraud'])
        manual_count = len(manual_labels)
        
        print(f"📊 Data Summary:")
        print(f"   Total Transactions: {total_transactions}")
        print(f"   Fraud Cases: {fraud_count} ({fraud_count/total_transactions*100:.1f}%)")
        print(f"   Manual Labels: {manual_count}")
        print(f"   Pending Review: {total_transactions - manual_count}")
    
    print(f"\n🚀 Server starting at: http://localhost:5000")
    print("📝 Available endpoints:")
    print("   GET  /dashboard           - Main dashboard")
    print("   GET  /false_cases         - False cases analysis") 
    print("   GET  /manual_labeling     - Manual labeling interface")
    print("   POST /label_transaction   - Submit manual label")
    print("   GET  /api/metrics         - Current metrics JSON")
    print("   GET  /api/data/query      - Query transaction data")
    print("   PUT  /api/data/edit/<id>  - Edit transaction label")
    print("-" * 60)
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)