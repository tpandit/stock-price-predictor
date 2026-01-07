"""
Phase 4: Model Training with MLflow
Train baseline Logistic Regression model and track experiment
"""
import sys
import os

# Get the notebook path and add parent directory to Python path
try:
    notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    # Remove notebook name and 'notebooks' directory to get workspace root
    path_parts = notebook_path.split("/")
    if len(path_parts) > 1 and path_parts[-2] == "notebooks":
        # Go up two levels: remove notebook name and 'notebooks' directory
        workspace_path = "/".join(path_parts[:-2])
    else:
        # Fallback: just remove notebook name
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
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import col
import mlflow
import mlflow.spark
from utils.helpers import load_config, get_spark_session, setup_mlflow_experiment

# Load configuration - use absolute path
try:
    config_path = f"{workspace_path}/config.yaml"
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)
except Exception as e:
    print(f"Warning: Could not use absolute path, trying relative: {e}")
    config = load_config("config.yaml")
gold_table = config['delta_tables']['gold']
experiment_name = config['mlflow']['experiment_name']
tracking_uri = config['mlflow']['tracking_uri']
train_test_split = config['model']['train_test_split']
random_seed = config['model']['random_seed']
feature_columns = config['model']['features']
lr_params = config['hyperparameters']['logistic_regression']

# Initialize Spark session
spark = get_spark_session("ModelTraining")

# Set up MLflow experiment
experiment_id = setup_mlflow_experiment(experiment_name, tracking_uri)
mlflow.set_experiment(experiment_id=experiment_id)

print(f"Reading from Gold table: {gold_table}")

# Read from gold table
df = spark.read.format("delta").table(gold_table)

print(f"Loaded {df.count()} rows from {gold_table}")

# Check for null values in feature columns
print("\nChecking for null values in features:")
for feature in feature_columns:
    null_count = df.filter(col(feature).isNull()).count()
    print(f"  {feature}: {null_count} nulls")

# Remove rows with null features
df = df.dropna(subset=feature_columns + ["label"])

print(f"\nRows after removing nulls: {df.count()}")

# Select feature columns and label
feature_df = df.select(feature_columns + ["label"])

# Assemble features into a vector
assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features"
)

# Transform data
feature_vector_df = assembler.transform(feature_df)

# Split data into train and test sets
train_df, test_df = feature_vector_df.randomSplit(
    [train_test_split, 1 - train_test_split],
    seed=random_seed
)

print(f"\nTrain set size: {train_df.count()} rows")
print(f"Test set size: {test_df.count()} rows")

# Start MLflow run
with mlflow.start_run(run_name="logistic_regression_baseline") as run:
    run_id = run.info.run_id
    print(f"\nMLflow Run ID: {run_id}")
    
    # Log parameters
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("maxIter", lr_params["maxIter"])
    mlflow.log_param("regParam", lr_params["regParam"])
    mlflow.log_param("train_test_split", train_test_split)
    mlflow.log_param("random_seed", random_seed)
    mlflow.log_param("features", ",".join(feature_columns))
    
    # Train Logistic Regression model
    print("\nTraining Logistic Regression model...")
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=lr_params["maxIter"],
        regParam=lr_params["regParam"]
    )
    
    model = lr.fit(train_df)
    
    # Make predictions
    print("Making predictions on test set...")
    predictions = model.transform(test_df)
    
    # Evaluate model
    # Binary classification evaluator for AUC
    binary_evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    
    auc = binary_evaluator.evaluate(predictions)
    
    # Multiclass evaluator for accuracy, precision, recall
    multiclass_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )
    
    accuracy = multiclass_evaluator.evaluate(predictions)
    
    # Precision
    precision_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="weightedPrecision"
    )
    precision = precision_evaluator.evaluate(predictions)
    
    # Recall
    recall_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="weightedRecall"
    )
    recall = recall_evaluator.evaluate(predictions)
    
    # Log metrics
    mlflow.log_metric("auc", auc)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    
    # Log model (don't register directly to avoid registry URI issues)
    try:
        mlflow.spark.log_model(
            model,
            "model"
        )
    except Exception as e:
        print(f"Warning: Could not log model with mlflow.spark: {e}")
        print("Trying alternative logging method...")
        # Alternative: log as generic model
        mlflow.pyfunc.log_model(
            "model",
            python_model=model,
            artifacts={}
        )
    
    # Tag run
    mlflow.set_tag("model_type", "LogisticRegression")
    mlflow.set_tag("phase", "baseline")
    
    print("\nModel Training Results:")
    print(f"  AUC: {auc:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    
    # Show prediction distribution
    print("\nPrediction Distribution:")
    predictions.groupBy("label", "prediction").count().orderBy("label", "prediction").show()
    
    print(f"\nModel logged to MLflow with run_id: {run_id}")
    print(f"Experiment: {experiment_name}")

print("\nTraining completed successfully!")



