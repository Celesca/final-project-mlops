"""
Demo script for the Fraud Detection Audit Web Application

This script tests the connection to the FastAPI backend and demonstrates
the key features of the audit web app.
"""

import requests
import os
from datetime import datetime

# Backend API configuration
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')


def test_backend_connection():
    """Test connection to the FastAPI backend."""
    print("🔍 Testing Backend Connection")
    print("=" * 50)
    
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        if response.status_code == 200:
            print(f"✅ Connected to FastAPI backend at {API_BASE_URL}")
            data = response.json()
            print(f"   Version: {data.get('version', 'unknown')}")
            return True
        else:
            print(f"❌ Backend returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to backend at {API_BASE_URL}")
        print(f"   Error: {e}")
        return False


def test_predictions_endpoints():
    """Test the prediction query endpoints."""
    print("\n📊 Testing Prediction Endpoints")
    print("-" * 50)
    
    endpoints = [
        ("/query/GET/predictions", "All Predictions"),
        ("/query/GET/frauds", "Fraud Predictions"),
        ("/query/GET/non_frauds", "Non-Fraud Predictions"),
        ("/query/GET/stats", "Statistics"),
    ]
    
    for endpoint, name in endpoints:
        try:
            response = requests.get(f"{API_BASE_URL}{endpoint}?limit=5", timeout=10)
            if response.status_code == 200:
                data = response.json()
                count = len(data) if isinstance(data, list) else "N/A"
                print(f"✅ {name}: {count} items")
            else:
                print(f"❌ {name}: Status {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"❌ {name}: {e}")


def show_sample_predictions():
    """Show sample prediction data."""
    print("\n📋 Sample Predictions")
    print("-" * 50)
    
    try:
        response = requests.get(f"{API_BASE_URL}/query/GET/predictions?limit=3", timeout=10)
        if response.status_code == 200:
            predictions = response.json()
            
            if not predictions:
                print("   No predictions found in database")
                return
            
            for i, p in enumerate(predictions[:3], 1):
                print(f"\n   Prediction #{i}:")
                print(f"   - ID: {p.get('id')}")
                print(f"   - Type: {p.get('type')}")
                print(f"   - Amount: ${p.get('amount', 0):,.2f}")
                print(f"   - Probability: {p.get('predict_proba', 0):.4f}")
                print(f"   - Prediction: {'FRAUD' if p.get('prediction') else 'LEGIT'}")
                print(f"   - Actual Label: {p.get('actual_label')}")
        else:
            print(f"   Failed to fetch predictions: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"   Error: {e}")


def show_demo_guide():
    """Show demonstration guide."""
    print("""
🎯 Demo Guide for Fraud Detection Audit Web App
===============================================

The Flask app connects to the FastAPI backend to:
- Fetch predictions (frauds, non-frauds, all)
- Update actual labels for predictions
- Calculate accuracy metrics

📈 DASHBOARD (/dashboard):
   • View accuracy metrics (F1, Precision, Recall, Accuracy)
   • See confusion matrix visualization
   • Check summary statistics
   • Quick links to other pages

🚨 FRAUDS (/frauds):
   • View all predictions marked as fraud
   • See transaction details
   • Label transactions

✅ NON-FRAUDS (/non_frauds):
   • View all predictions marked as legitimate
   • See transaction details
   • Label transactions

❌ FALSE CASES (/false_cases):
   • Review False Positives - legitimate flagged as fraud
   • Review False Negatives - fraud missed by model
   • Analyze prediction errors

🏷️ MANUAL LABELING (/manual_labeling):
   • Review unlabeled predictions
   • Use buttons to mark as Fraud or Legitimate
   • Labels are saved to the backend database

📡 API Endpoints:
   • GET  /api/predictions   - All predictions from backend
   • GET  /api/frauds        - Fraud predictions only
   • GET  /api/non_frauds    - Non-fraud predictions only
   • GET  /api/metrics       - Calculated accuracy metrics
   • GET  /api/stats         - Prediction statistics
   • POST /api/label         - Label a prediction
   • GET  /api/health        - Backend health check

🔧 Prerequisites:
   1. Start Docker containers: docker compose up -d
   2. Wait for model-serving to be ready
   3. Run some predictions through the DAG
   4. Start this Flask app: python app.py

💡 Tips:
   • The web app requires the FastAPI backend to be running
   • Labels are persisted in the PostgreSQL database
   • Refresh pages to see updated metrics
   • Use the /api/health endpoint to check backend status
    """)


def main():
    """Main demo function."""
    print("🎭 Fraud Detection Audit Web App Demo")
    print("=" * 60)
    print(f"📡 Backend URL: {API_BASE_URL}")
    print(f"📅 Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Test backend connection
    connected = test_backend_connection()
    
    if connected:
        # Test prediction endpoints
        test_predictions_endpoints()
        
        # Show sample predictions
        show_sample_predictions()
    
    # Show demo guide
    show_demo_guide()
    
    if connected:
        response = input("\n🚀 Start the web application now? (y/n): ")
        if response.lower() == 'y':
            print("\nStarting Flask app...")
            os.system("python app.py")
        else:
            print("\n📝 To start the application: python app.py")
            print("   Then visit: http://localhost:5000")
    else:
        print("\n⚠️  Backend is not running!")
        print("   Please start the Docker containers first:")
        print("   cd c:\\Users\\Sawit\\Desktop\\final-project-mlops")
        print("   docker compose up -d")
        print("\n   Then run this demo again or start the Flask app:")
        print("   python app.py")


if __name__ == "__main__":
    main()
