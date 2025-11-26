"""
Fraud Detection Audit Web Application Backend

A Flask web application that serves as a frontend for the FastAPI model-serving backend.
Fetches predictions from http://localhost:8000 and provides audit/labeling interface.
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_from_directory
import requests
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json

app = Flask(__name__, static_folder='build/static', template_folder='build')
app.secret_key = 'fraud_audit_secret_key_2025'

# Backend API configuration
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')

# Add context processor for templates
@app.context_processor
def inject_current_date():
    return {'current_date': datetime.now().strftime('%b %d, %Y')}


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
    """Fetch fraud predictions from the FastAPI backend."""
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
    """Fetch non-fraud predictions from the FastAPI backend."""
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
    """Update a prediction's actual_label via the FastAPI backend."""
    try:
        url = f"{API_BASE_URL}/query/PUT/predictions"
        payload = {
            "transaction_id": prediction_id,  # API uses transaction_id field
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

def calculate_metrics_from_predictions(predictions):
    """Calculate accuracy metrics from predictions list."""
    if not predictions:
        return None
    
    # Filter predictions that have both prediction and actual_label
    labeled = [p for p in predictions if p.get('actual_label') is not None]
    
    if not labeled:
        # Return basic stats without accuracy metrics
        total = len(predictions)
        fraud_predictions = sum(1 for p in predictions if p.get('prediction') == True)
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
    
    total = len(predictions)
    
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


def get_false_cases_from_predictions(predictions):
    """Extract false positive and false negative cases from predictions."""
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
    
    return false_positives, false_negatives


def get_unlabeled_predictions(predictions, limit=20):
    """Get predictions that haven't been labeled yet."""
    unlabeled = [p for p in predictions if p.get('actual_label') is None]
    # Sort by probability (highest first - more interesting cases)
    unlabeled.sort(key=lambda x: x.get('predict_proba', 0), reverse=True)
    return unlabeled[:limit]


# ============================================================================
# Static File Routes (React Build)
# ============================================================================

@app.route('/')
def serve_react():
    """Serve React app."""
    return send_from_directory(app.template_folder, 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    """Serve static files."""
    if os.path.exists(os.path.join(app.template_folder, path)):
        return send_from_directory(app.template_folder, path)
    return send_from_directory(app.template_folder, 'index.html')


# ============================================================================
# API Routes - For React Frontend
# ============================================================================

@app.route('/api/predictions')
def api_get_predictions():
    """Get all predictions."""
    limit = request.args.get('limit', type=int)
    predictions = fetch_all_predictions(limit=limit)
    return jsonify(predictions)


@app.route('/api/frauds')
def api_get_frauds():
    """Get fraud predictions (prediction=True)."""
    limit = request.args.get('limit', type=int)
    frauds = fetch_frauds(limit=limit)
    return jsonify(frauds)


@app.route('/api/non_frauds')
def api_get_non_frauds():
    """Get non-fraud predictions (prediction=False)."""
    limit = request.args.get('limit', type=int)
    non_frauds = fetch_non_frauds(limit=limit)
    return jsonify(non_frauds)


@app.route('/api/stats')
def api_get_stats():
    """Get prediction statistics."""
    stats = fetch_stats()
    return jsonify(stats)


@app.route('/api/metrics')
def api_metrics():
    """Get calculated metrics from all predictions."""
    predictions = fetch_all_predictions()
    metrics = calculate_metrics_from_predictions(predictions)
    if metrics:
        return jsonify(metrics)
    return jsonify({'error': 'No data available'}), 404


@app.route('/api/false_cases')
def api_false_cases():
    """Get false positive and false negative cases."""
    predictions = fetch_all_predictions()
    false_positives, false_negatives = get_false_cases_from_predictions(predictions)
    return jsonify({
        'false_positives': false_positives[:50],
        'false_negatives': false_negatives[:50]
    })


@app.route('/api/unlabeled')
def api_unlabeled():
    """Get unlabeled predictions for manual review."""
    limit = request.args.get('limit', default=20, type=int)
    predictions = fetch_all_predictions()
    unlabeled = get_unlabeled_predictions(predictions, limit=limit)
    return jsonify(unlabeled)


@app.route('/api/label', methods=['POST'])
def api_label_prediction():
    """Label a prediction with actual_label."""
    try:
        data = request.get_json()
        prediction_id = data.get('prediction_id') or data.get('id')
        actual_label = data.get('actual_label')
        
        if prediction_id is None:
            return jsonify({'error': 'prediction_id is required'}), 400
        
        if actual_label is None or actual_label not in [True, False, 0, 1]:
            return jsonify({'error': 'actual_label must be true/false or 0/1'}), 400
        
        # Convert to boolean
        actual_label_bool = bool(actual_label) if isinstance(actual_label, int) else actual_label
        
        result = update_prediction_label(prediction_id, actual_label_bool)
        
        if result:
            return jsonify({'success': True, 'result': result})
        else:
            return jsonify({'error': 'Failed to update label'}), 500
            
    except Exception as e:
        print(f"Error labeling prediction: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/transaction/<int:transaction_id>')
def api_transaction_details(transaction_id):
    """Get details for a specific transaction/prediction."""
    predictions = fetch_all_predictions()
    
    for p in predictions:
        if p.get('id') == transaction_id or p.get('transaction_id') == transaction_id:
            return jsonify(p)
    
    return jsonify({'error': 'Transaction not found'}), 404


@app.route('/api/data/query')
def api_query_data():
    """Query data endpoint - returns paginated transaction data."""
    try:
        page = request.args.get('page', default=1, type=int)
        per_page = request.args.get('per_page', default=20, type=int)
        filter_type = request.args.get('type', 'all')  # all, fraud, non_fraud, unlabeled
        
        # Fetch data based on filter
        if filter_type == 'fraud':
            predictions = fetch_frauds()
        elif filter_type == 'non_fraud':
            predictions = fetch_non_frauds()
        else:
            predictions = fetch_all_predictions()
        
        # Additional filtering for unlabeled
        if filter_type == 'unlabeled':
            predictions = [p for p in predictions if p.get('actual_label') is None]
        
        # Pagination
        total = len(predictions)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        page_data = predictions[start_idx:end_idx]
        
        return jsonify({
            'transactions': page_data,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total,
                'pages': (total + per_page - 1) // per_page if total > 0 else 0
            }
        })
        
    except Exception as e:
        print(f"Error querying data: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/data/edit/<int:prediction_id>', methods=['PUT'])
def api_edit_prediction(prediction_id):
    """Edit prediction's actual_label."""
    try:
        data = request.get_json()
        actual_label = data.get('actual_label')
        
        if actual_label is not None and actual_label not in [True, False, 0, 1]:
            return jsonify({'error': 'Invalid actual_label value'}), 400
        
        if actual_label is not None:
            actual_label_bool = bool(actual_label) if isinstance(actual_label, int) else actual_label
            result = update_prediction_label(prediction_id, actual_label_bool)
            
            if result:
                return jsonify({'success': True, 'message': 'Prediction updated', 'result': result})
            else:
                return jsonify({'error': 'Failed to update'}), 500
        
        return jsonify({'success': True, 'message': 'No changes made'})
        
    except Exception as e:
        print(f"Error editing prediction: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Health Check
# ============================================================================

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
    print("🔍 Starting Fraud Detection Audit Backend")
    print("=" * 60)
    print(f"📡 Backend API URL: {API_BASE_URL}")
    print(f"\n🚀 Server starting at: http://localhost:5000")
    print("\n📝 Available API endpoints:")
    print("   GET  /api/predictions      - All predictions")
    print("   GET  /api/frauds           - Fraud predictions only")
    print("   GET  /api/non_frauds       - Non-fraud predictions only")
    print("   GET  /api/stats            - Prediction statistics")
    print("   GET  /api/metrics          - Calculated accuracy metrics")
    print("   GET  /api/false_cases      - False positives/negatives")
    print("   GET  /api/unlabeled        - Unlabeled predictions")
    print("   POST /api/label            - Label a prediction")
    print("   GET  /api/transaction/<id> - Transaction details")
    print("   GET  /api/data/query       - Query with pagination")
    print("   PUT  /api/data/edit/<id>   - Edit prediction label")
    print("   GET  /api/health           - Health check")
    print("-" * 60)
    
    # Test backend connection
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        if response.status_code == 200:
            print(f"✅ Connected to FastAPI backend at {API_BASE_URL}")
        else:
            print(f"⚠️  Backend returned status {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"⚠️  Warning: Cannot connect to backend at {API_BASE_URL}")
        print(f"   Error: {e}")
        print("   Make sure the model-serving container is running!")
    
    print("-" * 60)
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)
