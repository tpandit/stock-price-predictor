"""
Complete Stock Price Prediction Pipeline
Reads from coinbase table and runs entire ML pipeline in one notebook
"""
import sys
import os

# Get the notebook path and add parent directory to Python path
try:
    notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    path_parts = notebook_path.split("/")
    if len(path_parts) > 1 and path_parts[-2] == "notebooks":
        workspace_path = "/".join(path_parts[:-2])
    else:
        workspace_path = "/".join(path_parts[:-1])
    
    if workspace_path not in sys.path:
        sys.path.insert(0, workspace_path)
    print(f"Notebook path: {notebook_path}")
    print(f"Workspace path: {workspace_path}")
except Exception as e:
    print(f"Warning: Could not get notebook path: {e}")
    current_dir = os.getcwd()
    if "notebooks" in current_dir:
        workspace_path = current_dir.replace("/notebooks", "")
        sys.path.insert(0, workspace_path)
    else:
        workspace_path = current_dir
        sys.path.insert(0, workspace_path)

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lag, lead, when, to_date, lit,
    avg, stddev
)
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    GBTClassifier
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient
from utils.helpers import load_config, get_spark_session, setup_mlflow_experiment

# Load configuration
try:
    config_path = f"{workspace_path}/config.yaml"
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)
except Exception as e:
    print(f"Warning: Could not use absolute path, trying relative: {e}")
    config = load_config("config.yaml")

# Configuration
source_table = config.get('data_source', {}).get('source_table', 'coinbase')
default_symbol = config.get('data_source', {}).get('default_symbol', 'TATAGLOBAL')
experiment_name = config['mlflow']['experiment_name']
tracking_uri = config['mlflow']['tracking_uri']
train_test_split = config['model']['train_test_split']
random_seed = config['model']['random_seed']
feature_columns = config['model']['features']
lr_params = config['hyperparameters']['logistic_regression']
rf_params = config['hyperparameters']['random_forest']
gbt_params = config['hyperparameters']['gbt']
model_name = "StockTrendPrediction"

# Initialize Spark session
spark = get_spark_session("CompletePipeline")

print("="*80)
print("COMPLETE STOCK PRICE PREDICTION PIPELINE")
print("="*80)
print(f"Source table: {source_table}")
print(f"Default symbol: {default_symbol}")
print(f"MLflow experiment: {experiment_name}")

# ============================================================================
# PHASE 1: DATA INGESTION & TRANSFORMATION
# ============================================================================
print("\n" + "="*80)
print("PHASE 1: Data Ingestion & Transformation")
print("="*80)

# Read from source table
print(f"\nReading from source table: {source_table}")
try:
    source_df = spark.read.table(source_table)
    print(f"Successfully read {source_df.count()} rows from {source_table}")
    print("\nSource table schema:")
    source_df.printSchema()
    print("\nSample of source data:")
    source_df.show(5)
except Exception as e:
    raise ValueError(f"Error reading from table {source_table}: {e}")

# Map column names (case-insensitive matching)
column_mapping = {}
for col_name in source_df.columns:
    col_lower = col_name.lower()
    if 'date' in col_lower:
        column_mapping['date'] = col_name
    elif 'open' in col_lower and 'date' not in col_lower:
        column_mapping['open'] = col_name
    elif 'high' in col_lower:
        column_mapping['high'] = col_name
    elif 'low' in col_lower:
        column_mapping['low'] = col_name
    elif 'close' in col_lower:
        column_mapping['close'] = col_name
    elif 'volume' in col_lower or 'quantity' in col_lower or 'trade' in col_lower:
        column_mapping['volume'] = col_name
    elif 'symbol' in col_lower:
        column_mapping['symbol'] = col_name

print(f"\nDetected column mapping: {column_mapping}")

# Transform to bronze schema
print("\nTransforming data to bronze schema...")
bronze_df = source_df.select(
    to_date(col(column_mapping.get('date', 'Date')), format='mixed', errors='coerce').alias('date'),
    when(col(column_mapping.get('symbol', 'symbol')).isNotNull(), 
         col(column_mapping.get('symbol', 'symbol'))).otherwise(lit(default_symbol)).alias('symbol'),
    col(column_mapping.get('open', 'Open')).cast('double').alias('open'),
    col(column_mapping.get('high', 'High')).cast('double').alias('high'),
    col(column_mapping.get('low', 'Low')).cast('double').alias('low'),
    col(column_mapping.get('close', 'Close')).cast('double').alias('close'),
    col(column_mapping.get('volume', 'Total Trade Quantity')).cast('long').alias('volume')
)

# Remove rows with null dates or critical price data
bronze_df = bronze_df.filter(
    col('date').isNotNull() &
    col('open').isNotNull() &
    col('high').isNotNull() &
    col('low').isNotNull() &
    col('close').isNotNull() &
    col('volume').isNotNull()
)

# Validate and fix OHLC relationships
invalid_ohlc = bronze_df.filter(
    (col('high') < col('low')) |
    (col('high') < col('open')) |
    (col('high') < col('close')) |
    (col('low') > col('open')) |
    (col('low') > col('close'))
)

invalid_count = invalid_ohlc.count()
if invalid_count > 0:
    print(f"Warning: Found {invalid_count} rows with invalid OHLC relationships. Removing...")
    bronze_df = bronze_df.filter(
        (col('high') >= col('low')) &
        (col('high') >= col('open')) &
        (col('high') >= col('close')) &
        (col('low') <= col('open')) &
        (col('low') <= col('close'))
    )

bronze_df = bronze_df.orderBy('date', 'symbol')
print(f"Bronze data: {bronze_df.count()} rows after transformation")

# ============================================================================
# PHASE 2: FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*80)
print("PHASE 2: Feature Engineering")
print("="*80)

# Define window specification
window_spec = Window.partitionBy("symbol").orderBy("date")

# Calculate daily return
silver_df = bronze_df.withColumn(
    "prev_close",
    lag("close", 1).over(window_spec)
).withColumn(
    "daily_return",
    when(col("prev_close").isNotNull() & (col("prev_close") != 0),
         (col("close") - col("prev_close")) / col("prev_close")
    ).otherwise(None)
)

# Calculate moving averages
silver_df = silver_df.withColumn(
    "ma_5",
    avg("close").over(window_spec.rowsBetween(-4, 0))
).withColumn(
    "ma_10",
    avg("close").over(window_spec.rowsBetween(-9, 0))
).withColumn(
    "ma_20",
    avg("close").over(window_spec.rowsBetween(-19, 0))
)

# Calculate volatility
silver_df = silver_df.withColumn(
    "volatility",
    stddev("daily_return").over(window_spec.rowsBetween(-9, 0))
)

# Calculate volume change
silver_df = silver_df.withColumn(
    "prev_volume",
    lag("volume", 1).over(window_spec)
).withColumn(
    "volume_change",
    when(col("prev_volume").isNotNull() & (col("prev_volume") != 0),
         (col("volume") - col("prev_volume")).cast("double") / col("prev_volume")
    ).otherwise(None)
)

# Select final columns and remove rows where essential features are null
silver_df = silver_df.select(
    "date", "symbol", "open", "high", "low", "close", "volume",
    "daily_return", "ma_5", "ma_10", "ma_20", "volatility", "volume_change"
).filter(
    col("ma_20").isNotNull() & 
    col("daily_return").isNotNull() &
    col("volatility").isNotNull()
)

print(f"Silver data: {silver_df.count()} rows after feature engineering")

# ============================================================================
# PHASE 3: LABEL CREATION
# ============================================================================
print("\n" + "="*80)
print("PHASE 3: Label Creation")
print("="*80)

# Get next day's close price and create label
gold_df = silver_df.withColumn(
    "next_day_close",
    lead("close", 1).over(window_spec)
).withColumn(
    "label",
    when(col("next_day_close").isNotNull(),
         when(col("next_day_close") > col("close"), 1).otherwise(0)
    ).otherwise(None)
).filter(col("label").isNotNull())

print(f"Gold data: {gold_df.count()} rows after label creation")

# Show label distribution
print("\nLabel Distribution:")
label_dist = gold_df.groupBy("label").count().orderBy("label")
label_dist.show()

total_rows = gold_df.count()
label_1_count = gold_df.filter(col("label") == 1).count()
label_0_count = gold_df.filter(col("label") == 0).count()
print(f"Label = 1 (price goes up): {label_1_count} ({label_1_count/total_rows*100:.2f}%)")
print(f"Label = 0 (price goes down/stays same): {label_0_count} ({label_0_count/total_rows*100:.2f}%)")

# ============================================================================
# PHASE 4: MODEL TRAINING & COMPARISON
# ============================================================================
print("\n" + "="*80)
print("PHASE 4: Model Training & Comparison")
print("="*80)

# Set up MLflow experiment
experiment_id = setup_mlflow_experiment(experiment_name, tracking_uri)
mlflow.set_experiment(experiment_id=experiment_id)

# Prepare data for modeling
df = gold_df.dropna(subset=feature_columns + ["label"])

# Assemble features
assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features"
)
feature_vector_df = assembler.transform(df.select(feature_columns + ["label"]))

# Split data
train_df, test_df = feature_vector_df.randomSplit(
    [train_test_split, 1 - train_test_split],
    seed=random_seed
)

print(f"Train set: {train_df.count()} rows")
print(f"Test set: {test_df.count()} rows")

# Initialize evaluators
binary_evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)
multiclass_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction"
)

# Dictionary to store model results
model_results = {}

# 1. Logistic Regression
print("\n" + "-"*60)
print("Training Logistic Regression...")
print("-"*60)

with mlflow.start_run(run_name="logistic_regression") as run:
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=lr_params["maxIter"],
        regParam=lr_params["regParam"]
    )
    
    model = lr.fit(train_df)
    predictions = model.transform(test_df)
    
    auc = binary_evaluator.evaluate(predictions)
    accuracy = multiclass_evaluator.setMetricName("accuracy").evaluate(predictions)
    precision = multiclass_evaluator.setMetricName("weightedPrecision").evaluate(predictions)
    recall = multiclass_evaluator.setMetricName("weightedRecall").evaluate(predictions)
    
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("maxIter", lr_params["maxIter"])
    mlflow.log_param("regParam", lr_params["regParam"])
    mlflow.log_metric("auc", auc)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    
    try:
        mlflow.spark.log_model(model, "model")
    except Exception as e:
        print(f"Warning: Could not log model: {e}")
    
    mlflow.set_tag("model_type", "LogisticRegression")
    
    model_results["LogisticRegression"] = {
        "run_id": run.info.run_id,
        "auc": auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }
    
    print(f"AUC: {auc:.4f}, Accuracy: {accuracy:.4f}")

# 2. Random Forest
print("\n" + "-"*60)
print("Training Random Forest...")
print("-"*60)

with mlflow.start_run(run_name="random_forest") as run:
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        numTrees=rf_params["numTrees"],
        maxDepth=rf_params["maxDepth"],
        seed=random_seed
    )
    
    model = rf.fit(train_df)
    predictions = model.transform(test_df)
    
    auc = binary_evaluator.evaluate(predictions)
    accuracy = multiclass_evaluator.setMetricName("accuracy").evaluate(predictions)
    precision = multiclass_evaluator.setMetricName("weightedPrecision").evaluate(predictions)
    recall = multiclass_evaluator.setMetricName("weightedRecall").evaluate(predictions)
    
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("numTrees", rf_params["numTrees"])
    mlflow.log_param("maxDepth", rf_params["maxDepth"])
    mlflow.log_metric("auc", auc)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    
    try:
        mlflow.spark.log_model(model, "model")
    except Exception as e:
        print(f"Warning: Could not log model: {e}")
    
    mlflow.set_tag("model_type", "RandomForest")
    
    model_results["RandomForest"] = {
        "run_id": run.info.run_id,
        "auc": auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }
    
    print(f"AUC: {auc:.4f}, Accuracy: {accuracy:.4f}")

# 3. Gradient-Boosted Trees
print("\n" + "-"*60)
print("Training Gradient-Boosted Trees...")
print("-"*60)

with mlflow.start_run(run_name="gbt") as run:
    gbt = GBTClassifier(
        featuresCol="features",
        labelCol="label",
        maxIter=gbt_params["maxIter"],
        maxDepth=gbt_params["maxDepth"],
        seed=random_seed
    )
    
    model = gbt.fit(train_df)
    predictions = model.transform(test_df)
    
    auc = binary_evaluator.evaluate(predictions)
    accuracy = multiclass_evaluator.setMetricName("accuracy").evaluate(predictions)
    precision = multiclass_evaluator.setMetricName("weightedPrecision").evaluate(predictions)
    recall = multiclass_evaluator.setMetricName("weightedRecall").evaluate(predictions)
    
    mlflow.log_param("model_type", "GBT")
    mlflow.log_param("maxIter", gbt_params["maxIter"])
    mlflow.log_param("maxDepth", gbt_params["maxDepth"])
    mlflow.log_metric("auc", auc)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    
    try:
        mlflow.spark.log_model(model, "model")
    except Exception as e:
        print(f"Warning: Could not log model: {e}")
    
    mlflow.set_tag("model_type", "GBT")
    
    model_results["GBT"] = {
        "run_id": run.info.run_id,
        "auc": auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }
    
    print(f"AUC: {auc:.4f}, Accuracy: {accuracy:.4f}")

# Compare models
print("\n" + "="*60)
print("Model Comparison Summary")
print("="*60)

best_model = max(model_results.items(), key=lambda x: x[1]["auc"])
best_model_name = best_model[0]
best_model_metrics = best_model[1]

print(f"\nBest Model (by AUC): {best_model_name}")
print(f"  Run ID: {best_model_metrics['run_id']}")
print(f"  AUC: {best_model_metrics['auc']:.4f}")
print(f"  Accuracy: {best_model_metrics['accuracy']:.4f}")
print(f"  Precision: {best_model_metrics['precision']:.4f}")
print(f"  Recall: {best_model_metrics['recall']:.4f}")

print("\nAll Models Comparison:")
print(f"{'Model':<20} {'AUC':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10}")
print("-" * 60)
for model_name, metrics in model_results.items():
    print(f"{model_name:<20} {metrics['auc']:<10.4f} {metrics['accuracy']:<10.4f} "
          f"{metrics['precision']:<10.4f} {metrics['recall']:<10.4f}")

# ============================================================================
# PHASE 5: MODEL REGISTRY
# ============================================================================
print("\n" + "="*80)
print("PHASE 5: Model Registry")
print("="*80)

best_run_id = best_model_metrics['run_id']

print(f"\nRegistering best model '{model_name}' from run {best_run_id}...")

try:
    # Register model
    model_version = mlflow.register_model(
        f"runs:/{best_run_id}/model",
        model_name
    )
    
    print(f"Successfully registered model:")
    print(f"  Model Name: {model_name}")
    print(f"  Version: {model_version.version}")
    print(f"  Stage: {model_version.current_stage}")
    
    # Add description and tags
    client = MlflowClient()
    client.update_model_version(
        name=model_name,
        version=model_version.version,
        description=f"Stock trend prediction model - {best_model_name}. AUC: {best_model_metrics['auc']:.4f}"
    )
    
    client.set_model_version_tag(
        name=model_name,
        version=model_version.version,
        key="model_type",
        value=str(best_model_name)
    )
    
    # Transition to Staging
    print(f"\nTransitioning model to 'Staging' stage...")
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Staging"
    )
    
    print(f"Model version {model_version.version} promoted to Staging")
    
    # Display registered model information
    registered_model = client.get_registered_model(model_name)
    print(f"\nModel Name: {registered_model.name}")
    print(f"Latest Versions: {len(registered_model.latest_versions)}")
    
except Exception as e:
    print(f"Error registering model: {str(e)}")
    print("Note: Model may already be registered. Check MLflow UI.")

# ============================================================================
# PIPELINE COMPLETE
# ============================================================================
print("\n" + "="*80)
print("PIPELINE COMPLETE!")
print("="*80)
print(f"\nSummary:")
print(f"  - Source data: {source_df.count()} rows from {source_table}")
print(f"  - Bronze data: {bronze_df.count()} rows")
print(f"  - Silver data: {silver_df.count()} rows")
print(f"  - Gold data: {gold_df.count()} rows")
print(f"  - Best model: {best_model_name} (AUC: {best_model_metrics['auc']:.4f})")
print(f"\nView results in MLflow UI:")
print(f"  - Experiment: {experiment_name}")
print(f"  - Model Registry: {model_name}")
print("\n" + "="*80)

