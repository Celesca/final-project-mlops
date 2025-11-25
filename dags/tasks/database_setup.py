"""
Database setup task for initializing prediction storage tables.
"""
from dags.utils.database import init_prediction_db


def setup_prediction_database(**kwargs):
    """
    Initialize the prediction database tables.
    This task should run once before predictions start being stored.
    """
    print("🔧 Setting up prediction database...")
    init_prediction_db()
    print("✅ Prediction database setup complete!")

