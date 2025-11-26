# Fraud Detection Audit Web Application

A comprehensive web-based audit dashboard for reviewing and validating fraud detection model predictions. This Flask application provides interfaces for monitoring model performance, reviewing false cases, and manually labeling transactions.

## Features

### 🎯 Dashboard
- **Accuracy Metrics**: F1 Score, Precision, Recall, and Accuracy
- **Confusion Matrix Visualization**: Interactive chart showing prediction breakdown
- **Summary Statistics**: Total transactions, fraud counts, and review status
- **False Cases Preview**: Quick view of top false positives and negatives

### 📊 False Cases Analysis
- **False Positives**: Transactions incorrectly flagged as fraud
- **False Negatives**: Fraudulent transactions missed by the model
- **Detailed Transaction Information**: Complete transaction data with account balances
- **Quick Labeling**: Direct labeling actions from the false cases view

### 🏷️ Manual Labeling Interface
- **Interactive Transaction Cards**: Easy-to-review transaction format
- **Risk Factor Analysis**: Automatic identification of suspicious patterns
- **Keyboard Shortcuts**: Fast labeling with 'F' (fraud) and 'L' (legitimate)
- **Progress Tracking**: Visual indicators of labeling progress

### 🛠️ Additional Features
- **Auto-refresh Metrics**: Real-time dashboard updates every 30 seconds
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Transaction Details Modal**: Detailed analysis with risk factors
- **Manual Review Tracking**: Database columns for audit trail

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- PostgreSQL database (from the main project setup)
- Virtual environment (recommended)

### Installation

1. **Navigate to the audit web app directory**:
   ```bash
   cd 5_Audit_Web_App
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Update database schema** (adds manual review columns):
   ```bash
   python update_db_schema.py
   ```

5. **Start the web application**:
   ```bash
   python app.py
   ```

6. **Access the application**:
   Open your browser and go to: `http://localhost:5000`

## Database Schema Updates

The audit application adds two new columns to the `financial_data` table:

- `manual_status` (Boolean): Whether the transaction has been manually reviewed
- `manual_label` (Integer): Manual fraud label (0=not fraud, 1=fraud, null=not reviewed)

## Usage Guide

### Dashboard Navigation
1. **Dashboard** (`/dashboard`): Main overview with metrics and charts
2. **False Cases** (`/false_cases`): Detailed view of prediction errors
3. **Manual Labeling** (`/manual_labeling`): Interface for reviewing transactions

### Manual Labeling Process
1. Navigate to the Manual Labeling page
2. Review transaction details including:
   - Transaction amount and type
   - Account balances before/after
   - Model prediction vs actual label
   - Risk factors analysis
3. Click "Not Fraud" (green) or "Fraud" (red) to label
4. Use keyboard shortcuts for faster labeling:
   - Press **F** to mark as fraud
   - Press **L** to mark as legitimate

### Reviewing False Cases
1. Go to False Cases page
2. Review False Positives (incorrectly flagged as fraud)
3. Review False Negatives (missed fraud cases)
4. Use the quick action buttons to label transactions
5. Click on transaction IDs for detailed analysis

## API Endpoints

- `GET /api/metrics` - Get current performance metrics
- `GET /api/transaction/<id>` - Get detailed transaction information
- `POST /label_transaction` - Submit manual label for a transaction

## Configuration

The application uses the same `.env` file as the main project for database connections:

```env
DB_NAME=mydatabase
DB_USER=admin
DB_PASS=password
DB_HOST=localhost
DB_PORT=5432
```

## Technical Architecture

### Backend (Flask)
- **Flask**: Web framework
- **psycopg2**: PostgreSQL database connectivity
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Metrics calculation

### Frontend
- **Bootstrap 5**: Responsive UI framework
- **Chart.js**: Interactive charts and visualizations
- **Font Awesome**: Icons and visual elements
- **Vanilla JavaScript**: Dynamic interactions

### Database Integration
- Connects to the same PostgreSQL database as the main fraud detection system
- Reads prediction results and actual labels
- Stores manual review status and labels
- Calculates real-time accuracy metrics

## Development Notes

### Adding New Features
1. Model performance trends over time
2. Batch labeling operations
3. Export functionality for labeled data
4. Integration with model retraining pipeline

### Security Considerations
- Input validation on all form submissions
- SQL injection prevention with parameterized queries
- Session management for multi-user support
- Rate limiting for API endpoints

### Performance Optimizations
- Database query optimization with proper indexing
- Pagination for large transaction sets
- Caching for frequently accessed metrics
- Asynchronous loading for better user experience

## Troubleshooting

### Common Issues

1. **Database Connection Error**:
   - Check if PostgreSQL is running
   - Verify `.env` file configuration
   - Ensure database contains the `financial_data` table

2. **No Data Showing**:
   - Run the database schema update script
   - Check if the main fraud detection system has populated data
   - Verify the `probability` column exists and has values

3. **Metrics Not Calculating**:
   - Ensure both `isFraud` and `probability` columns have data
   - Check for null values in critical columns
   - Verify the prediction threshold (0.5) matches your model

### Debug Mode
To run in debug mode for development:
```bash
export FLASK_DEBUG=1  # Linux/Mac
set FLASK_DEBUG=1     # Windows
python app.py
```

## Contributing

When adding new features or fixing bugs:

1. Test with sample data first
2. Ensure responsive design works on all screen sizes
3. Add appropriate error handling
4. Update this README if adding new functionality
5. Follow the existing code style and patterns

## Integration with MLOps Pipeline

This audit web app is designed to integrate with the complete MLOps pipeline:

- **Part 1 (DAG_Data)**: Consumes processed transaction data
- **Part 2 (Data_Drifts)**: Could integrate drift alerts in the dashboard
- **Part 3 (Model_Serving)**: Reviews predictions from the serving layer
- **Part 4 (Retraining_MLFlow)**: Manual labels can feed back into retraining

The audit results and manual labels can be used to:
- Trigger model retraining when accuracy drops
- Identify data drift patterns
- Improve model feature engineering
- Generate quality reports for stakeholders