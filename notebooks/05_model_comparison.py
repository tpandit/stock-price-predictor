"""
Phase 5: Model Comparison
Train and compare multiple models: Logistic Regression, Random Forest, GBT
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    GBTClassifier
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import col
import mlflow
import mlflow.spark
from utils.helpers import load_config, get_spark_session, setup_mlflow_experiment

# Load configuration
config = load_config()
gold_table = config['delta_tables']['gold']
experiment_name = config['mlflow']['experiment_name']
tracking_uri = config['mlflow']['tracking_uri']
train_test_split = config['model']['train_test_split']
random_seed = config['model']['random_seed']
feature_columns = config['model']['features']
lr_params = config['hyperparameters']['logistic_regression']
rf_params = config['hyperparameters']['random_forest']
gbt_params = config['hyperparameters']['gbt']

# Initialize Spark session
spark = get_spark_session("ModelComparison")

# Set up MLflow experiment
experiment_id = setup_mlflow_experiment(experiment_name, tracking_uri)
mlflow.set_experiment(experiment_id=experiment_id)

print(f"Reading from Gold table: {gold_table}")

# Read from gold table
df = spark.read.format("delta").table(gold_table)

# Remove rows with null features
df = df.dropna(subset=feature_columns + ["label"])

# Select feature columns and label
feature_df = df.select(feature_columns + ["label"])

# Assemble features into a vector
assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features"
)

# Transform data
feature_vector_df = assembler.transform(feature_df)

# Split data into train and test sets (use same split as baseline)
train_df, test_df = feature_vector_df.randomSplit(
    [train_test_split, 1 - train_test_split],
    seed=random_seed
)

print(f"Train set size: {train_df.count()} rows")
print(f"Test set size: {test_df.count()} rows")

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

# 1. Train Logistic Regression
print("\n" + "="*50)
print("Training Logistic Regression...")
print("="*50)

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
    
    mlflow.spark.log_model(model, "model")
    mlflow.set_tag("model_type", "LogisticRegression")
    
    model_results["LogisticRegression"] = {
        "run_id": run.info.run_id,
        "auc": auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }
    
    print(f"AUC: {auc:.4f}, Accuracy: {accuracy:.4f}")

# 2. Train Random Forest
print("\n" + "="*50)
print("Training Random Forest Classifier...")
print("="*50)

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
    
    mlflow.spark.log_model(model, "model")
    mlflow.set_tag("model_type", "RandomForest")
    
    model_results["RandomForest"] = {
        "run_id": run.info.run_id,
        "auc": auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }
    
    print(f"AUC: {auc:.4f}, Accuracy: {accuracy:.4f}")

# 3. Train Gradient-Boosted Trees
print("\n" + "="*50)
print("Training Gradient-Boosted Trees...")
print("="*50)

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
    
    mlflow.spark.log_model(model, "model")
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
print("\n" + "="*50)
print("Model Comparison Summary")
print("="*50)

# Find best model by AUC
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

# Store best model info for next phase
print(f"\nBest model run_id saved: {best_model_metrics['run_id']}")
print(f"View results in MLflow UI: Experiment '{experiment_name}'")

print("\nModel comparison completed successfully!")



