# Databricks notebook source
# MAGIC %md
# MAGIC # Complete Stock Price Prediction Pipeline
# MAGIC 
# MAGIC This notebook runs the complete consolidated pipeline that:
# MAGIC - Reads directly from the `coinbase` table (skips ingestion)
# MAGIC - Performs feature engineering
# MAGIC - Creates labels for next-day price movement
# MAGIC - Trains and compares multiple ML models
# MAGIC - Registers the best model to MLflow Model Registry
# MAGIC 
# MAGIC **All steps are executed in a single consolidated script.**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline Execution
# MAGIC 
# MAGIC Click "Run All" to execute the entire pipeline.

# COMMAND ----------

print("=" * 80)
print("COMPLETE STOCK PRICE PREDICTION PIPELINE")
print("=" * 80)
print("Reading from: coinbase table")
print("Pipeline: Feature Engineering → Label Creation → Model Training → Model Comparison → Model Registry")
print("=" * 80)

# COMMAND ----------
%run ./complete_pipeline_coinbase

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline Complete!
# MAGIC 
# MAGIC The pipeline has finished executing. Check:
# MAGIC - **MLflow Experiments** for model metrics and comparisons
# MAGIC - **Model Registry** for the registered best model
# MAGIC - **Delta Tables** (stocks_silver, stocks_gold) for processed data
# MAGIC 
# MAGIC ### Accessing MLflow Results
# MAGIC 
# MAGIC 1. **In Databricks UI**: Click on "MLflow" in the left sidebar
# MAGIC 2. **Experiments**: Navigate to `/Shared/stock_trend_prediction`
# MAGIC 3. **Model Registry**: Look for model named `StockTrendPrediction`

# COMMAND ----------

# Display MLflow experiment information
import mlflow

try:
    experiment_path = "/Shared/stock_trend_prediction"
    experiment = mlflow.get_experiment_by_name(experiment_path)
    if experiment:
        print("=" * 80)
        print("MLFLOW EXPERIMENT INFORMATION")
        print("=" * 80)
        print(f"Experiment Name: {experiment.name}")
        print(f"Experiment ID: {experiment.experiment_id}")
        print(f"Artifact Location: {experiment.artifact_location}")
        print(f"Tracking URI: {mlflow.get_tracking_uri()}")
        print("\nTo view experiments:")
        print("  1. Go to Databricks UI → MLflow (left sidebar)")
        print(f"  2. Navigate to: {experiment_path}")
        print("=" * 80)
    else:
        print(f"⚠ Could not find experiment at: {experiment_path}")
except Exception as e:
    print(f"⚠ Could not retrieve experiment info: {e}")

# COMMAND ----------

# Display Model Registry information
from mlflow.tracking import MlflowClient

try:
    client = MlflowClient()
    model_name = "StockTrendPrediction"
    registered_model = client.get_registered_model(model_name)
    
    print("=" * 80)
    print("MODEL REGISTRY INFORMATION")
    print("=" * 80)
    print(f"Model Name: {registered_model.name}")
    print(f"Total Versions: {len(registered_model.latest_versions)}")
    print("\nVersions:")
    for version in registered_model.latest_versions:
        print(f"  - Version {version.version}: {version.current_stage}")
        print(f"    Run ID: {version.run_id}")
    print("\nTo view in Model Registry:")
    print("  1. Go to Databricks UI → MLflow → Models")
    print(f"  2. Click on: {model_name}")
    print("=" * 80)
except Exception as e:
    print(f"⚠ Model not found in registry or error: {e}")
    print("This is normal if the model registration failed. Check the pipeline output above.")

