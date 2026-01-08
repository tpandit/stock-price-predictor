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

# Run the complete consolidated pipeline
%run ./complete_pipeline_coinbase

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline Complete!
# MAGIC 
# MAGIC The pipeline has finished executing. Check:
# MAGIC - **MLflow Experiments** for model metrics and comparisons
# MAGIC - **Model Registry** for the registered best model
# MAGIC - **Delta Tables** (stocks_silver, stocks_gold) for processed data

