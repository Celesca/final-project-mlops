"""
Fraud Detection Audit Web Application

A Flask web application for auditing fraud detection model predictions.
Integrates with the FastAPI backend at http://localhost:8000 for:
- Fetching predictions (frauds, non-frauds, all)
- Updating actual labels for predictions
- Viewing prediction statistics

Provides dashboards for accuracy metrics, false cases review, and manual labeling.
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import requests
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

app = Flask(__name__)
app.secret_key = 'fraud_audit_secret_key_2025'

# Backend API configuration
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')


# ============================================================================
# API Client Functions - Communicate with FastAPI Backend
# ============================================================================

def fetch_all_predictions(limit=None):
    """Fetch all predictions from the FastAPI backend."""
    try:
        url = f"{API_BASE_URL}/query/GET/predictions"
        if limit:
            url += f"?limit={limit}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching predictions: {e}")
        return []


def fetch_frauds(limit=None):
    """Fetch fraud predictions (prediction=True) from the FastAPI backend."""
    try:
        url = f"{API_BASE_URL}/query/GET/frauds"
        if limit:
            url += f"?limit={limit}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching frauds: {e}")
        return []


def fetch_non_frauds(limit=None):
    """Fetch non-fraud predictions (prediction=False) from the FastAPI backend."""
    try:
        url = f"{API_BASE_URL}/query/GET/non_frauds"
        if limit:
            url += f"?limit={limit}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching non-frauds: {e}")
        return []


def fetch_stats():
    """Fetch prediction statistics from the FastAPI backend."""
    try:
        url = f"{API_BASE_URL}/query/GET/stats"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching stats: {e}")
        return {}


def update_prediction_label(prediction_id, actual_label):
    """Update a prediction's actual_label via the FastAPI backend.
    
    Uses PUT /query/PUT/predictions with payload:
    {
        "transaction_id": <prediction_id>,
        "actual_label": <true/false>
    }
    """
    try:
        url = f"{API_BASE_URL}/query/PUT/predictions"
        payload = {
            "transaction_id": prediction_id,
            "actual_label": actual_label
        }
        response = requests.put(url, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error updating prediction label: {e}")
        return None


# ============================================================================
# Metrics Calculation
# ============================================================================

def calculate_metrics():
    """Calculate accuracy metrics from predictions."""
    predictions = fetch_all_predictions()
    
    if not predictions:
        return None
    
    # Filter predictions that have actual_label set
    labeled = [p for p in predictions if p.get('actual_label') is not None]
    
    total = len(predictions)
    fraud_predictions = sum(1 for p in predictions if p.get('prediction') == True)
    
    if not labeled:
        return {
            'total_transactions': total,
            'predicted_fraud_count': fraud_predictions,
            'predicted_non_fraud_count': total - fraud_predictions,
            'labeled_count': 0,
            'unlabeled_count': total,
            'accuracy': None,
            'precision': None,
            'recall': None,
            'f1_score': None,
            'confusion_matrix': None,
        }
    
    y_true = [1 if p.get('actual_label') else 0 for p in labeled]
    y_pred = [1 if p.get('prediction') else 0 for p in labeled]
    
    metrics = {
        'total_transactions': total,
        'labeled_count': len(labeled),
        'unlabeled_count': total - len(labeled),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'true_fraud_count': sum(y_true),
        'predicted_fraud_count': sum(y_pred),
    }
    
    return metrics


def get_false_cases():
    """Get false positive and false negative cases from predictions."""
    predictions = fetch_all_predictions()
    
    false_positives = []
    false_negatives = []
    
    for p in predictions:
        actual = p.get('actual_label')
        predicted = p.get('prediction')
        
        # Skip unlabeled predictions
        if actual is None:
            continue
        
        # False Positive: predicted fraud but actually not fraud
        if predicted == True and actual == False:
            false_positives.append(p)
        
        # False Negative: predicted non-fraud but actually fraud
        elif predicted == False and actual == True:
            false_negatives.append(p)
    
    # Sort by probability
    false_positives.sort(key=lambda x: x.get('predict_proba', 0), reverse=True)
    false_negatives.sort(key=lambda x: x.get('predict_proba', 0))
    
    return false_positives[:50], false_negatives[:50]


def get_unlabeled_predictions(limit=20):
    """Get predictions that haven't been labeled (actual_label is NULL)."""
    predictions = fetch_all_predictions()
    unlabeled = [p for p in predictions if p.get('actual_label') is None]
    # Sort by probability (highest first for more interesting cases)
    unlabeled.sort(key=lambda x: x.get('predict_proba', 0), reverse=True)
    return unlabeled[:limit]


# ============================================================================
# Web Routes
# ============================================================================

@app.route('/')
def index():
    """Main dashboard page."""
    return redirect(url_for('dashboard'))


@app.route('/dashboard')
def dashboard():
    """Dashboard with accuracy metrics and overview."""
    metrics = calculate_metrics()
    false_positives, false_negatives = get_false_cases()
    stats = fetch_stats()
    
    return render_template('dashboard.html', 
                         metrics=metrics, 
                         stats=stats,
                         false_positives=false_positives[:10],
                         false_negatives=false_negatives[:10])


@app.route('/false_cases')
def false_cases():
    """Page showing all false positive and false negative cases."""
    false_positives, false_negatives = get_false_cases()
    
    return render_template('false_cases.html',
                         false_positives=false_positives,
                         false_negatives=false_negatives)


def filter_by_label_status(predictions, label_filter):
    """Filter predictions by label status (all, labeled, pending)."""
    if label_filter == 'labeled':
        return [p for p in predictions if p.get('actual_label') is not None]
    elif label_filter == 'pending':
        return [p for p in predictions if p.get('actual_label') is None]
    return predictions  # 'all' or no filter


@app.route('/frauds')
def frauds_list():
    """Page showing all fraud predictions with filter and link to manual labeling."""
    label_filter = request.args.get('filter', 'all')
    frauds = fetch_frauds()
    filtered_frauds = filter_by_label_status(frauds, label_filter)
    
    # Add index numbers for display (matching the original list position)
    for i, fraud in enumerate(filtered_frauds, 1):
        fraud['display_index'] = i
    
    return render_template('frauds_list.html',
                         predictions=filtered_frauds,
                         title='Fraud Predictions',
                         prediction_type='fraud',
                         current_filter=label_filter)


@app.route('/non_frauds')
def non_frauds_list():
    """Page showing all non-fraud predictions with inline labeling."""
    label_filter = request.args.get('filter', 'all')
    non_frauds = fetch_non_frauds()
    filtered_non_frauds = filter_by_label_status(non_frauds, label_filter)
    
    return render_template('non_frauds_list.html',
                         predictions=filtered_non_frauds,
                         title='Non-Fraud Predictions',
                         prediction_type='non_fraud',
                         current_filter=label_filter)


@app.route('/manual_labeling')
def manual_labeling():
    """Manual labeling interface for FRAUD predictions only."""
    # Get all fraud predictions that are pending labeling
    frauds = fetch_frauds()
    unlabeled_frauds = [f for f in frauds if f.get('actual_label') is None]
    # Sort by probability (highest first for most interesting cases)
    unlabeled_frauds.sort(key=lambda x: x.get('predict_proba', 0), reverse=True)
    
    # Add original index (position in the full fraud list) for reference
    all_frauds = fetch_frauds()
    fraud_id_to_index = {f.get('id') or f.get('transaction_id'): i + 1 for i, f in enumerate(all_frauds)}
    
    for fraud in unlabeled_frauds:
        fraud_id = fraud.get('id') or fraud.get('transaction_id')
        fraud['original_index'] = fraud_id_to_index.get(fraud_id, 0)
    
    return render_template('manual_labeling.html', transactions=unlabeled_frauds[:50])


@app.route('/label_transaction', methods=['POST'])
def label_transaction():
    """Endpoint to save manual labels via the FastAPI backend."""
    try:
        prediction_id = request.form.get('prediction_id') or request.form.get('transaction_id')
        manual_label = request.form.get('manual_label') or request.form.get('actual_label')
        return_to = request.form.get('return_to', 'manual_labeling')
        
        if not prediction_id or manual_label not in ['0', '1', 'true', 'false', 'True', 'False']:
            flash('Invalid data provided', 'error')
            return redirect(url_for('manual_labeling'))
        
        # Convert to boolean
        if manual_label in ['1', 'true', 'True']:
            actual_label = True
        else:
            actual_label = False
        
        result = update_prediction_label(int(prediction_id), actual_label)
        
        if result:
            flash('Transaction labeled successfully!', 'success')
        else:
            flash('Error saving label to backend', 'error')
            
    except Exception as e:
        print(f"Error in label_transaction: {e}")
        flash('Error processing request', 'error')
    
    # Redirect based on return_to parameter
    if return_to == 'non_frauds':
        return redirect(url_for('non_frauds_list'))
    elif return_to == 'frauds':
        return redirect(url_for('frauds_list'))
    else:
        return redirect(url_for('manual_labeling'))


# ============================================================================
# API Endpoints
# ============================================================================

@app.route('/api/metrics')
def api_metrics():
    """API endpoint to get current metrics."""
    metrics = calculate_metrics()
    if metrics:
        return jsonify(metrics)
    else:
        return jsonify({'error': 'No data available'}), 404


@app.route('/api/stats')
def api_stats():
    """API endpoint to get prediction statistics."""
    stats = fetch_stats()
    return jsonify(stats)


@app.route('/api/predictions')
def api_predictions():
    """API endpoint to get all predictions."""
    limit = request.args.get('limit', type=int)
    predictions = fetch_all_predictions(limit=limit)
    return jsonify(predictions)


@app.route('/api/frauds')
def api_frauds():
    """API endpoint to get fraud predictions."""
    limit = request.args.get('limit', type=int)
    frauds = fetch_frauds(limit=limit)
    return jsonify(frauds)


@app.route('/api/non_frauds')
def api_non_frauds():
    """API endpoint to get non-fraud predictions."""
    limit = request.args.get('limit', type=int)
    non_frauds = fetch_non_frauds(limit=limit)
    return jsonify(non_frauds)


@app.route('/api/false_cases')
def api_false_cases():
    """API endpoint to get false positive and negative cases."""
    false_positives, false_negatives = get_false_cases()
    return jsonify({
        'false_positives': false_positives,
        'false_negatives': false_negatives
    })


@app.route('/api/unlabeled')
def api_unlabeled():
    """API endpoint to get unlabeled predictions."""
    limit = request.args.get('limit', default=20, type=int)
    unlabeled = get_unlabeled_predictions(limit=limit)
    return jsonify(unlabeled)


@app.route('/api/label', methods=['POST'])
def api_label():
    """API endpoint to label a prediction."""
    try:
        data = request.get_json()
        prediction_id = data.get('prediction_id') or data.get('id')
        actual_label = data.get('actual_label')
        
        if prediction_id is None:
            return jsonify({'error': 'prediction_id is required'}), 400
        
        if actual_label is None:
            return jsonify({'error': 'actual_label is required'}), 400
        
        # Convert to boolean
        if isinstance(actual_label, bool):
            actual_label_bool = actual_label
        elif isinstance(actual_label, int):
            actual_label_bool = bool(actual_label)
        elif isinstance(actual_label, str):
            actual_label_bool = actual_label.lower() in ['1', 'true']
        else:
            return jsonify({'error': 'Invalid actual_label type'}), 400
        
        result = update_prediction_label(int(prediction_id), actual_label_bool)
        
        if result:
            return jsonify({'success': True, 'result': result})
        else:
            return jsonify({'error': 'Failed to update label'}), 500
            
    except Exception as e:
        print(f"Error in api_label: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/transaction/<int:transaction_id>')
def api_transaction_details(transaction_id):
    """API endpoint to get transaction details."""
    predictions = fetch_all_predictions()
    
    for p in predictions:
        if p.get('id') == transaction_id or p.get('transaction_id') == transaction_id:
            return jsonify(p)
    
    return jsonify({'error': 'Transaction not found'}), 404


@app.route('/api/health')
def api_health():
    """Health check endpoint."""
    backend_status = "unknown"
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        backend_status = "healthy" if response.status_code == 200 else "unhealthy"
    except:
        backend_status = "unreachable"
    
    return jsonify({
        'status': 'running',
        'backend_url': API_BASE_URL,
        'backend_status': backend_status,
        'timestamp': datetime.now().isoformat()
    })


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    print("🔍 Starting Fraud Detection Audit Web App")
    print("=" * 60)
    print(f"📡 Backend API URL: {API_BASE_URL}")
    
    # Test backend connection
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        if response.status_code == 200:
            print(f"✅ Connected to FastAPI backend")
        else:
            print(f"⚠️  Backend returned status {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"⚠️  Warning: Cannot connect to backend at {API_BASE_URL}")
        print(f"   Make sure the model-serving container is running!")
    
    print(f"\n🚀 Server starting at: http://localhost:5001")
    print("\n📝 Web Pages:")
    print("   /dashboard         - Main dashboard with metrics")
    print("   /frauds            - View fraud predictions")
    print("   /non_frauds        - View non-fraud predictions")
    print("   /false_cases       - View false positives/negatives")
    print("   /manual_labeling   - Label predictions manually")
    print("\n📝 API Endpoints:")
    print("   GET  /api/predictions   - All predictions")
    print("   GET  /api/frauds        - Fraud predictions")
    print("   GET  /api/non_frauds    - Non-fraud predictions")
    print("   GET  /api/metrics       - Accuracy metrics")
    print("   GET  /api/stats         - Statistics")
    print("   GET  /api/false_cases   - False cases")
    print("   GET  /api/unlabeled     - Unlabeled predictions")
    print("   POST /api/label         - Label a prediction")
    print("   GET  /api/health        - Health check")
    print("-" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5001)
