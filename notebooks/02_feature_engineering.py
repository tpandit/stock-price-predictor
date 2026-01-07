"""
Phase 2: Feature Engineering (Silver Layer)
Create time-series features from raw stock data
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lag, when, isnan, isnull, 
    avg, stddev, lead, window
)
from pyspark.sql.window import Window
from utils.helpers import load_config, get_spark_session

# Load configuration
config = load_config()
bronze_table = config['delta_tables']['bronze']
silver_table = config['delta_tables']['silver']

# Initialize Spark session
spark = get_spark_session("FeatureEngineering")

print(f"Reading from Bronze table: {bronze_table}")

# Read from bronze table
df = spark.read.format("delta").table(bronze_table)

print(f"Loaded {df.count()} rows from {bronze_table}")

# Define window specification: partition by symbol, order by date
window_spec = Window.partitionBy("symbol").orderBy("date")

# Calculate daily return: (close - previous_close) / previous_close
df = df.withColumn(
    "prev_close",
    lag("close", 1).over(window_spec)
).withColumn(
    "daily_return",
    when(col("prev_close").isNotNull() & (col("prev_close") != 0),
         (col("close") - col("prev_close")) / col("prev_close")
    ).otherwise(None)
)

# Calculate moving averages
# MA_5: 5-day moving average of close price
df = df.withColumn(
    "ma_5",
    avg("close").over(window_spec.rowsBetween(-4, 0))
)

# MA_10: 10-day moving average of close price
df = df.withColumn(
    "ma_10",
    avg("close").over(window_spec.rowsBetween(-9, 0))
)

# MA_20: 20-day moving average of close price
df = df.withColumn(
    "ma_20",
    avg("close").over(window_spec.rowsBetween(-19, 0))
)

# Calculate volatility: 10-day rolling standard deviation of daily returns
df = df.withColumn(
    "volatility",
    stddev("daily_return").over(window_spec.rowsBetween(-9, 0))
)

# Calculate volume change: (volume - previous_volume) / previous_volume
df = df.withColumn(
    "prev_volume",
    lag("volume", 1).over(window_spec)
).withColumn(
    "volume_change",
    when(col("prev_volume").isNotNull() & (col("prev_volume") != 0),
         (col("volume") - col("prev_volume")).cast("double") / col("prev_volume")
    ).otherwise(None)
)

# Select final columns for silver table
silver_df = df.select(
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
    "volume_change"
)

# Remove rows where essential features are null (first few days for each symbol)
# Keep rows where at least ma_20 is calculated (need 20 days of history)
silver_df = silver_df.filter(
    col("ma_20").isNotNull() & 
    col("daily_return").isNotNull() &
    col("volatility").isNotNull()
)

print(f"\nFeature Engineering Summary:")
print(f"Rows after feature engineering: {silver_df.count()}")
print(f"Rows with null daily_return: {silver_df.filter(col('daily_return').isNull()).count()}")
print(f"Rows with null volatility: {silver_df.filter(col('volatility').isNull()).count()}")

# Show sample data
print("\nSample data with features:")
silver_df.orderBy("symbol", "date").show(10)

# Write to Delta table
print(f"\nWriting to Silver table: {silver_table}")
silver_df.write.format("delta").mode("overwrite").saveAsTable(silver_table)

print(f"Successfully wrote {silver_df.count()} rows to {silver_table}")

# Verify table creation
verify_df = spark.read.format("delta").table(silver_table)
print(f"\nVerification: Table {silver_table} contains {verify_df.count()} rows")
print("\nTable schema:")
verify_df.printSchema()

# Show feature statistics
print("\nFeature Statistics:")
silver_df.select("daily_return", "ma_5", "ma_10", "ma_20", "volatility", "volume_change").describe().show()



