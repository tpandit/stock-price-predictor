"""
Phase 3: Label Creation (Gold Layer)
Create supervised learning labels for next-day price movement prediction
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
from pyspark.sql.functions import col, lead, when
from pyspark.sql.window import Window
from utils.helpers import load_config, get_spark_session

# Load configuration - use absolute path
try:
    config_path = f"{workspace_path}/config.yaml"
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)
except Exception as e:
    print(f"Warning: Could not use absolute path, trying relative: {e}")
    config = load_config("config.yaml")
silver_table = config['delta_tables']['silver']
gold_table = config['delta_tables']['gold']

# Initialize Spark session
spark = get_spark_session("LabelCreation")

print(f"Reading from Silver table: {silver_table}")

# Read from silver table
df = spark.read.format("delta").table(silver_table)

print(f"Loaded {df.count()} rows from {silver_table}")

# Define window specification: partition by symbol, order by date
window_spec = Window.partitionBy("symbol").orderBy("date")

# Get next day's close price using lead function
df = df.withColumn(
    "next_day_close",
    lead("close", 1).over(window_spec)
)

# Create label: 1 if next_day_close > current_close, else 0
df = df.withColumn(
    "label",
    when(col("next_day_close").isNotNull(),
         when(col("next_day_close") > col("close"), 1).otherwise(0)
    ).otherwise(None)
)

# Drop rows with null labels (last day for each symbol)
gold_df = df.filter(col("label").isNotNull())

# Select final columns for gold table (model-ready dataset)
gold_df = gold_df.select(
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "daily_return",
    "ma_5",
    "ma_10",
    "ma_20",
    "volatility",
    "volume_change",
    "label"
)

print(f"\nLabel Creation Summary:")
print(f"Rows before label creation: {df.count()}")
print(f"Rows after label creation (dropping nulls): {gold_df.count()}")
print(f"Rows dropped: {df.count() - gold_df.count()}")

# Show label distribution
print("\nLabel Distribution:")
label_dist = gold_df.groupBy("label").count().orderBy("label")
label_dist.show()

# Show label distribution by symbol
print("\nLabel Distribution by Symbol:")
label_dist_by_symbol = gold_df.groupBy("symbol", "label").count().orderBy("symbol", "label")
label_dist_by_symbol.show()

# Calculate label percentages
total_rows = gold_df.count()
label_1_count = gold_df.filter(col("label") == 1).count()
label_0_count = gold_df.filter(col("label") == 0).count()

print(f"\nLabel Statistics:")
print(f"Total rows: {total_rows}")
print(f"Label = 1 (price goes up): {label_1_count} ({label_1_count/total_rows*100:.2f}%)")
print(f"Label = 0 (price goes down/stays same): {label_0_count} ({label_0_count/total_rows*100:.2f}%)")

# Show sample data
print("\nSample data with labels:")
gold_df.orderBy("symbol", "date").show(10)

# Write to Delta table
print(f"\nWriting to Gold table: {gold_table}")
gold_df.write.format("delta").mode("overwrite").saveAsTable(gold_table)

print(f"Successfully wrote {gold_df.count()} rows to {gold_table}")

# Verify table creation
verify_df = spark.read.format("delta").table(gold_table)
print(f"\nVerification: Table {gold_table} contains {verify_df.count()} rows")
print("\nTable schema:")
verify_df.printSchema()



