# Stock Trend Prediction using MLflow on Databricks

End-to-end ML pipeline for predicting stock price movements using Delta Lake, Spark ML, and MLflow.

## 🚀 Quick Start

**New to this project?** Start here:
- **[QUICKSTART.md](QUICKSTART.md)** - Get running in 5 minutes
- **[INSTRUCTIONS.md](INSTRUCTIONS.md)** - Complete setup and troubleshooting guide
- **[JOB_SETUP.md](JOB_SETUP.md)** - Set up automated Databricks Jobs
- **[LOCAL_SETUP.md](LOCAL_SETUP.md)** - Run locally on your machine

## Overview

This project demonstrates a production-style ML pipeline on Databricks:
- **Bronze/Silver/Gold Architecture** using Delta Lake
- **Feature Engineering** with time-series financial indicators
- **Model Training** with Spark ML (Logistic Regression, Random Forest, GBT)
- **MLflow Integration** for experiment tracking and model registry
- **Mock Data Generation** (no external API calls required)

## Prerequisites

- Databricks workspace with access to:
  - Delta Lake
  - MLflow
  - Spark 3.5+ runtime
- Python 3.8+ (or use uv with the specified version)
- GitHub account (for version control)

## Setup

**For detailed setup instructions, see [INSTRUCTIONS.md](INSTRUCTIONS.md)**

### Quick Setup Steps

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/stock-trend-prediction.git
   git push -u origin main
   ```

2. **Connect to Databricks**
   - Go to **Repos** → **Add Repo**
   - Connect to your GitHub repository
   - All files will be synced automatically

3. **Create Cluster**
   - Runtime: Databricks Runtime 13.3 LTS or higher
   - Enable: Delta Lake, MLflow (default)

4. **Run Notebooks**
   - Execute notebooks 01-06 in sequence
   - Attach to your cluster before running

## Running the Pipeline

Execute the notebooks **sequentially** in order:

### Phase 1: Data Ingestion
```python
# Run: notebooks/01_ingest_bronze.py
```
- Generates realistic mock stock data (OHLCV)
- Creates `stocks_bronze` Delta table
- **Expected time**: < 1 minute

### Phase 2: Feature Engineering
```python
# Run: notebooks/02_feature_engineering.py
```
- Creates time-series features (returns, moving averages, volatility)
- Creates `stocks_silver` Delta table
- **Expected time**: 1-2 minutes

### Phase 3: Label Creation
```python
# Run: notebooks/03_label_creation.py
```
- Creates binary labels (up/down) for next-day prediction
- Creates `stocks_gold` Delta table
- **Expected time**: < 1 minute

### Phase 4: Model Training
```python
# Run: notebooks/04_model_training_mlflow.py
```
- Trains baseline Logistic Regression model
- Logs experiment to MLflow
- **Expected time**: 2-5 minutes

### Phase 5: Model Comparison
```python
# Run: notebooks/05_model_comparison.py
```
- Trains and compares 3 models (LR, RF, GBT)
- Logs all experiments to MLflow
- **Expected time**: 5-15 minutes

### Phase 6: Model Registry
```python
# Run: notebooks/06_model_registry.py
```
- Registers best model to MLflow Model Registry
- Promotes model to Staging
- **Expected time**: < 1 minute

## Quick Start (All at Once)

### Option 1: Single Notebook (Recommended for Real Data)

If you have stock data in a Databricks table (e.g., `coinbase`), use the complete pipeline notebook:

```python
# Run the complete pipeline in one notebook
%run ./complete_pipeline_coinbase
```

This notebook:
- Reads directly from your source table (e.g., `coinbase`)
- Skips data ingestion
- Runs all phases: transformation → features → labels → training → comparison → registry
- All in one execution!

### Option 2: Use Master Notebook

A master notebook (`00_master_pipeline.py`) runs all separate notebooks in sequence:

```python
# Run the master notebook
%run ./00_master_pipeline
```

### Option 3: Manual Sequence

Or run notebooks individually:

```python
%run ./01_ingest_bronze
%run ./02_feature_engineering
%run ./03_label_creation
%run ./04_model_training_mlflow
%run ./05_model_comparison
%run ./06_model_registry
```

### Option 4: Set Up Automated Job

For production, set up a Databricks Job to run automatically. See **[JOB_SETUP.md](JOB_SETUP.md)** for detailed instructions.

## Configuration

Edit `config.yaml` to customize:

- **Stock symbols**: Change the `symbols` list
- **Date range**: Modify `start_date` and `end_date`
- **Delta table names**: Update table paths if needed
- **MLflow experiment**: Change experiment name
- **Model hyperparameters**: Adjust model settings

## Viewing Results

### MLflow UI

1. In Databricks, go to **Experiments** in the left sidebar
2. Find experiment: `stock_trend_prediction`
3. Compare runs, view metrics, and inspect models

### Model Registry

1. Go to **Models** in the left sidebar
2. Find model: `StockTrendPrediction`
3. View registered versions and stages

### Delta Tables

Query the tables using Databricks SQL or notebooks:

```python
# View bronze data
spark.read.format("delta").table("stocks_bronze").show()

# View silver data with features
spark.read.format("delta").table("stocks_silver").show()

# View gold data with labels
spark.read.format("delta").table("stocks_gold").show()
```

## Troubleshooting

### Import Errors

If notebooks can't find `utils`:

1. Make sure `utils/` directory is in the same path as notebooks
2. Or update the `sys.path.append()` in each notebook to point to the correct location

### Config File Not Found

1. Upload `config.yaml` to Databricks workspace
2. Update the path in `load_config()` calls, or use absolute path:
   ```python
   config = load_config("/Workspace/Shared/config.yaml")
   ```

### MLflow Experiment Issues

- On Databricks, MLflow is automatically configured
- If using local MLflow, update `tracking_uri` in `config.yaml`

### Data Generation

- Mock data is generated using realistic price movements (random walk with drift)
- Each symbol has a unique seed for reproducibility
- Data includes proper OHLC relationships and volume patterns

## Project Structure

```
databricks/
├── config.yaml              # Configuration file
├── pyproject.toml           # uv project config
├── requirements.txt         # pip dependencies (backup)
├── notebooks/
│   ├── complete_pipeline_coinbase.py  # Single notebook for complete pipeline
│   ├── 00_master_pipeline.py         # Master notebook to run all steps
│   ├── 01_ingest_bronze.py
│   ├── 02_feature_engineering.py
│   ├── 03_label_creation.py
│   ├── 04_model_training_mlflow.py
│   ├── 05_model_comparison.py
│   └── 06_model_registry.py
└── utils/
    ├── __init__.py
    └── helpers.py           # Utility functions
```

## Next Steps

- Tune hyperparameters using MLflow
- Add more features (technical indicators, market data)
- Implement walk-forward validation
- Create Databricks SQL dashboards
- Set up automated retraining pipelines

