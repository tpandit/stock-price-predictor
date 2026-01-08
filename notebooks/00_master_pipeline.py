# Databricks notebook source
# MAGIC %md
# MAGIC # Stock Price Prediction Pipeline - Master Notebook
# MAGIC 
# MAGIC This notebook runs the entire ML pipeline in sequence:
# MAGIC 1. Data Ingestion (Bronze)
# MAGIC 2. Feature Engineering (Silver)
# MAGIC 3. Label Creation (Gold)
# MAGIC 4. Model Training
# MAGIC 5. Model Comparison
# MAGIC 6. Model Registry

# COMMAND ----------
# MAGIC %md
# MAGIC ## Pipeline Execution

# COMMAND ----------

print("=" * 80)
print("STOCK PRICE PREDICTION PIPELINE - MASTER RUN")
print("=" * 80)
print(f"Start Time: {dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()}")

# COMMAND ----------
# MAGIC %md
# MAGIC ### Phase 1: Data Ingestion

# COMMAND ----------

print("\n[1/6] Running Data Ingestion...")
print("-" * 80)
%run ./01_ingest_bronze
print("✓ Data Ingestion Complete\n")

# COMMAND ----------
# MAGIC %md
# MAGIC ### Phase 2: Feature Engineering

# COMMAND ----------

print("\n[2/6] Running Feature Engineering...")
print("-" * 80)
%run ./02_feature_engineering
print("✓ Feature Engineering Complete\n")

# COMMAND ----------
# MAGIC %md
# MAGIC ### Phase 3: Label Creation

# COMMAND ----------

print("\n[3/6] Running Label Creation...")
print("-" * 80)
%run ./03_label_creation
print("✓ Label Creation Complete\n")

# COMMAND ----------
# MAGIC %md
# MAGIC ### Phase 4: Model Training

# COMMAND ----------

print("\n[4/6] Running Model Training...")
print("-" * 80)
%run ./04_model_training_mlflow
print("✓ Model Training Complete\n")

# COMMAND ----------
# MAGIC %md
# MAGIC ### Phase 5: Model Comparison

# COMMAND ----------

print("\n[5/6] Running Model Comparison...")
print("-" * 80)
%run ./05_model_comparison
print("✓ Model Comparison Complete\n")

# COMMAND ----------
# MAGIC %md
# MAGIC ### Phase 6: Model Registry

# COMMAND ----------

print("\n[6/6] Running Model Registry...")
print("-" * 80)
%run ./06_model_registry
print("✓ Model Registry Complete\n")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Pipeline Complete

# COMMAND ----------

print("=" * 80)
print("PIPELINE COMPLETE - ALL TASKS FINISHED SUCCESSFULLY")
print("=" * 80)
print("\nNext Steps:")
print("1. Check MLflow Experiments for model metrics")
print("2. View Model Registry for registered models")
print("3. Query Delta tables: stocks_bronze, stocks_silver, stocks_gold")

