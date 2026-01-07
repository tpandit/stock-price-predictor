"""
Phase 6: Model Registry
Register best model to MLflow Model Registry and promote to staging/production
"""
import sys
import os

# Get the notebook path and add parent directory to Python path
try:
    notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    workspace_path = "/".join(notebook_path.split("/")[:-1])
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
experiment_name = config['mlflow']['experiment_name']
tracking_uri = config['mlflow']['tracking_uri']
model_name = "StockTrendPrediction"

# Initialize Spark session
spark = get_spark_session("ModelRegistry")

# Set up MLflow experiment
experiment_id = setup_mlflow_experiment(experiment_name, tracking_uri)
mlflow.set_experiment(experiment_id=experiment_id)

print("="*50)
print("MLflow Model Registry")
print("="*50)

# Get all runs from the experiment
experiment = mlflow.get_experiment(experiment_id)
print(f"\nExperiment: {experiment.name}")
print(f"Experiment ID: {experiment_id}")

# Search for runs in the experiment
runs = mlflow.search_runs(experiment_ids=[experiment_id], order_by=["metrics.auc DESC"])

if runs.empty:
    print("No runs found in experiment. Please run model training notebooks first.")
    sys.exit(1)

print(f"\nFound {len(runs)} runs in experiment")

# Display top runs by AUC
print("\nTop runs by AUC:")
print(runs[["run_id", "tags.model_type", "metrics.auc", "metrics.accuracy"]].head(10))

# Get the best run (highest AUC)
best_run = runs.iloc[0]
best_run_id = best_run["run_id"]
best_model_type = best_run["tags.model_type"] if "tags.model_type" in best_run else "Unknown"
best_auc = best_run["metrics.auc"]

print(f"\nBest Run:")
print(f"  Run ID: {best_run_id}")
print(f"  Model Type: {best_model_type}")
print(f"  AUC: {best_auc:.4f}")

# Register the model
print(f"\nRegistering model '{model_name}' from run {best_run_id}...")

try:
    # Register model from the best run
    model_version = mlflow.register_model(
        f"runs:/{best_run_id}/model",
        model_name
    )
    
    print(f"Successfully registered model:")
    print(f"  Model Name: {model_name}")
    print(f"  Version: {model_version.version}")
    print(f"  Stage: {model_version.current_stage}")
    
    # Add model description
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    
    client.update_model_version(
        name=model_name,
        version=model_version.version,
        description=f"Stock trend prediction model - {best_model_type}. AUC: {best_auc:.4f}"
    )
    
    # Add tags to model version
    client.set_model_version_tag(
        name=model_name,
        version=model_version.version,
        key="model_type",
        value=str(best_model_type)
    )
    
    client.set_model_version_tag(
        name=model_name,
        version=model_version.version,
        key="best_run_id",
        value=str(best_run_id)
    )
    
    print(f"\nModel description and tags added")
    
    # Transition model to Staging
    print(f"\nTransitioning model to 'Staging' stage...")
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Staging"
    )
    
    print(f"Model version {model_version.version} promoted to Staging")
    
    # Optionally, transition to Production (commented out by default)
    # Uncomment the following lines to promote directly to Production
    # print(f"\nTransitioning model to 'Production' stage...")
    # client.transition_model_version_stage(
    #     name=model_name,
    #     version=model_version.version,
    #     stage="Production"
    # )
    # print(f"Model version {model_version.version} promoted to Production")
    
    # Display registered model information
    print("\n" + "="*50)
    print("Registered Model Information")
    print("="*50)
    
    registered_model = client.get_registered_model(model_name)
    print(f"Model Name: {registered_model.name}")
    print(f"Description: {registered_model.description}")
    print(f"Latest Versions: {len(registered_model.latest_versions)}")
    
    print("\nModel Versions:")
    for version in registered_model.latest_versions:
        print(f"  Version {version.version}: {version.current_stage} (created: {version.creation_timestamp})")
    
    print(f"\nModel successfully registered and available in MLflow Model Registry!")
    print(f"Access it via: MLflow UI > Models > {model_name}")
    
except Exception as e:
    print(f"Error registering model: {str(e)}")
    print("Note: Model may already be registered. Check MLflow UI for existing versions.")
    
    # Try to get existing model versions
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        registered_model = client.get_registered_model(model_name)
        print(f"\nExisting model '{model_name}' found with {len(registered_model.latest_versions)} versions")
    except:
        pass

print("\nModel registry phase completed!")



