"""
Phase 3: Label Creation (Gold Layer)
Create supervised learning labels for next-day price movement prediction
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lead, when
from pyspark.sql.window import Window
from utils.helpers import load_config, get_spark_session

# Load configuration
config = load_config()
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



