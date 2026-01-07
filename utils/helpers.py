"""
Utility functions for Stock Trend Prediction pipeline
"""
import yaml
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType
import mlflow


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_spark_session(app_name: str = "StockTrendPrediction") -> SparkSession:
    """
    Get or create Spark session with Delta Lake support
    
    Args:
        app_name: Name of the Spark application
        
    Returns:
        SparkSession instance
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()
    return spark


def validate_delta_table_schema(spark: SparkSession, table_name: str, expected_schema: StructType) -> bool:
    """
    Validate that a Delta table exists and has the expected schema
    
    Args:
        spark: SparkSession instance
        table_name: Name of the Delta table
        expected_schema: Expected schema as StructType
        
    Returns:
        True if schema matches, False otherwise
    """
    try:
        df = spark.read.format("delta").table(table_name)
        actual_schema = df.schema
        
        # Compare field names and types
        if len(actual_schema.fields) != len(expected_schema.fields):
            return False
        
        for actual_field, expected_field in zip(actual_schema.fields, expected_schema.fields):
            if actual_field.name != expected_field.name or actual_field.dataType != expected_field.dataType:
                return False
        
        return True
    except Exception as e:
        print(f"Error validating table {table_name}: {str(e)}")
        return False


def setup_mlflow_experiment(experiment_name: str, tracking_uri: str = "databricks") -> str:
    """
    Set up MLflow experiment and return experiment ID
    
    Args:
        experiment_name: Name of the MLflow experiment
        tracking_uri: MLflow tracking URI (default: "databricks" for Databricks)
        
    Returns:
        Experiment ID
    """
    if tracking_uri != "databricks":
        mlflow.set_tracking_uri(tracking_uri)
    
    # Construct experiment path
    if tracking_uri == "databricks":
        experiment_path = f"/Shared/{experiment_name}"
    else:
        experiment_path = experiment_name
    
    # Set the experiment (this will create if it doesn't exist)
    mlflow.set_experiment(experiment_path)
    
    # Get the experiment ID
    experiment = mlflow.get_experiment_by_name(experiment_path)
    if experiment:
        return experiment.experiment_id
    else:
        # Fallback: try to create explicitly
        try:
            return mlflow.create_experiment(experiment_path)
        except Exception:
            # If creation fails, experiment might already exist, try setting again
            mlflow.set_experiment(experiment_path)
            experiment = mlflow.get_experiment_by_name(experiment_path)
            if experiment:
                return experiment.experiment_id
            else:
                raise ValueError(f"Failed to create or access experiment: {experiment_path}")

