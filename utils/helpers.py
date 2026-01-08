"""
Utility functions for Stock Trend Prediction pipeline
"""
import yaml
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType
import mlflow


def load_config(config_path: str = None) -> dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config.yaml file (if None, tries to find it automatically)
        
    Returns:
        Dictionary containing configuration
    """
    import os
    
    def read_file_databricks(file_path: str) -> str:
        """Read file content in Databricks, handling both DBFS and workspace paths"""
        # Try multiple methods to read the file
        methods = []
        
        # Method 1: Try DBFS mount path (if workspace path)
        if file_path.startswith("/Users/") or file_path.startswith("/Workspace/"):
            # Convert workspace path to DBFS path
            if file_path.startswith("/Users/"):
                dbfs_path = f"/dbfs/Workspace{file_path}"
            else:
                dbfs_path = f"/dbfs{file_path}"
            methods.append(("DBFS mount", lambda: open(dbfs_path, 'r').read(), dbfs_path))
        
        # Method 2: Try dbutils.fs.head (for workspace files)
        if file_path.startswith("/Users/") or file_path.startswith("/Workspace/"):
            try:
                methods.append(("dbutils.fs", lambda: dbutils.fs.head(file_path), file_path))
            except NameError:
                pass
        
        # Method 3: Try direct file open (for local/relative paths)
        methods.append(("direct open", lambda: open(file_path, 'r').read(), file_path))
        
        # Try each method
        last_error = None
        for method_name, read_func, path in methods:
            try:
                if method_name == "DBFS mount" and not os.path.exists(path):
                    continue
                content = read_func()
                print(f"Successfully read config using {method_name} from: {path}")
                return content
            except Exception as e:
                last_error = e
                continue
        
        # If all methods failed, raise error
        raise FileNotFoundError(f"Could not read file at {file_path}. Last error: {last_error}")
    
    if config_path is None:
        # Try to find config.yaml automatically
        try:
            # Try Databricks notebook context
            try:
                notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
                path_parts = notebook_path.split("/")
                if len(path_parts) > 1 and path_parts[-2] == "notebooks":
                    workspace_path = "/".join(path_parts[:-2])
                else:
                    workspace_path = "/".join(path_parts[:-1])
                config_path = f"{workspace_path}/config.yaml"
            except NameError:
                # dbutils not available
                config_path = "config.yaml"
        except Exception as e:
            config_path = "config.yaml"
    elif not os.path.isabs(config_path):
        # If relative path, try to find it relative to workspace
        try:
            try:
                notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
                path_parts = notebook_path.split("/")
                if len(path_parts) > 1 and path_parts[-2] == "notebooks":
                    workspace_path = "/".join(path_parts[:-2])
                else:
                    workspace_path = "/".join(path_parts[:-1])
                abs_config_path = f"{workspace_path}/{config_path}"
                config_path = abs_config_path
            except NameError:
                pass  # Use original path
        except Exception:
            pass  # Use original path
    
    # Read the file using Databricks-compatible method
    try:
        content = read_file_databricks(config_path)
        config = yaml.safe_load(content)
        return config
    except Exception as e:
        # Try alternative: relative path in current directory
        if config_path != "config.yaml":
            try:
                print(f"Trying relative path 'config.yaml' as fallback...")
                with open("config.yaml", 'r') as f:
                    config = yaml.safe_load(f)
                print(f"Successfully loaded config from: config.yaml (relative)")
                return config
            except:
                pass
        
        raise FileNotFoundError(f"Could not find config.yaml. Tried: {config_path}. Error: {e}")


def get_spark_session(app_name: str = "StockTrendPrediction") -> SparkSession:
    """
    Get or create Spark session with Delta Lake support
    
    Args:
        app_name: Name of the Spark application
        
    Returns:
        SparkSession instance
    """
    # Check if we're in Databricks (dbutils available)
    try:
        # In Databricks, use existing Spark session
        from pyspark.sql import SparkSession
        spark = SparkSession.getActiveSession()
        if spark is None:
            spark = SparkSession.builder \
                .appName(app_name) \
                .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
                .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
                .getOrCreate()
        else:
            # Update app name if possible
            spark.sparkContext.setJobGroup(app_name, app_name)
    except NameError:
        # Not in Databricks, create new session
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


def is_local_path(path: str) -> bool:
    """
    Check if a path is a local file path (starts with file:// or is a local path)
    
    Args:
        path: Path to check
        
    Returns:
        True if local path, False if table name
    """
    return path.startswith("file://") or path.startswith("/") or "://" not in path and "/" in path


def read_delta_table(spark: SparkSession, table_path: str):
    """
    Read from Delta table, supporting both Databricks table names and local file paths
    
    Args:
        spark: SparkSession instance
        table_path: Either a table name (e.g., "stocks_bronze") or file path (e.g., "file:///tmp/delta/stocks_bronze")
        
    Returns:
        DataFrame
    """
    if is_local_path(table_path):
        return spark.read.format("delta").load(table_path)
    else:
        return spark.read.format("delta").table(table_path)


def write_delta_table(df, table_path: str, mode: str = "overwrite"):
    """
    Write to Delta table, supporting both Databricks table names and local file paths
    
    Args:
        df: DataFrame to write
        table_path: Either a table name (e.g., "stocks_bronze") or file path (e.g., "file:///tmp/delta/stocks_bronze")
        mode: Write mode (default: "overwrite")
        
    Returns:
        None
    """
    if is_local_path(table_path):
        # For local paths, ensure directory exists
        import os
        if table_path.startswith("file://"):
            actual_path = table_path[7:]  # Remove file:// prefix
        else:
            actual_path = table_path
        os.makedirs(actual_path, exist_ok=True)
        df.write.format("delta").mode(mode).save(table_path)
    else:
        df.write.format("delta").mode(mode).saveAsTable(table_path)


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

