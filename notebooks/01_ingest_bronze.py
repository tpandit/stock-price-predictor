"""
Phase 1: Data Ingestion (Bronze Layer)
Read stock data from source table and write to Delta bronze table
"""
import sys
import os

# Get the notebook path and add parent directory to Python path
try:
    # Get current notebook path
    notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    # Remove notebook name and 'notebooks' directory to get workspace root
    path_parts = notebook_path.split("/")
    if len(path_parts) > 1 and path_parts[-2] == "notebooks":
        # Go up two levels: remove notebook name and 'notebooks' directory
        workspace_path = "/".join(path_parts[:-2])
    else:
        # Fallback: just remove notebook name
        workspace_path = "/".join(path_parts[:-1])
    
    # Add to Python path so we can import utils
    if workspace_path not in sys.path:
        sys.path.insert(0, workspace_path)
    print(f"Notebook path: {notebook_path}")
    print(f"Workspace path: {workspace_path}")
except Exception as e:
    print(f"Warning: Could not get notebook path: {e}")
    # Fallback: try to find parent directory
    current_dir = os.getcwd()
    if "notebooks" in current_dir:
        parent_dir = current_dir.replace("/notebooks", "")
        sys.path.insert(0, parent_dir)
        workspace_path = parent_dir
    else:
        sys.path.insert(0, current_dir)
        workspace_path = current_dir

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, DateType
from pyspark.sql.functions import col, to_date, lit, when
from utils.helpers import load_config, get_spark_session

# Load configuration - use absolute path
try:
    config_path = f"{workspace_path}/config.yaml"
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)
except Exception as e:
    print(f"Warning: Could not use absolute path, trying relative: {e}")
    config = load_config("config.yaml")

bronze_table = config['delta_tables']['bronze']
start_date = config['date_range']['start_date']
end_date = config['date_range']['end_date']

# Get source table configuration (defaults to coinbase if not specified)
source_table = config.get('data_source', {}).get('source_table', 'coinbase')
default_symbol = config.get('data_source', {}).get('default_symbol', 'TATAGLOBAL')

# Initialize Spark session
spark = get_spark_session("StockIngestion")

print(f"Reading from source table: {source_table}")
print(f"Date range filter: {start_date} to {end_date}")
print(f"Default symbol: {default_symbol}")

# Read from source table
try:
    print(f"\nAttempting to read from table: {source_table}")
    source_df = spark.read.table(source_table)
    print(f"Successfully read {source_df.count()} rows from {source_table}")
    print("\nSource table schema:")
    source_df.printSchema()
    print("\nSample of source data:")
    source_df.show(5)
except Exception as e:
    print(f"Error reading from table {source_table}: {e}")
    print("Falling back to mock data generation...")
    # Fallback to mock data generation (keep original code as backup)
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    symbols = config.get('symbols', ['TATAGLOBAL'])
    # ... (mock data generation code would go here as fallback)
    raise ValueError(f"Could not read from source table {source_table}. Please ensure the table exists.")

# Transform the data to match expected bronze schema
# Expected columns: date, symbol, open, high, low, close, volume
# Source columns (from CSV): Date, Open, High, Low, Close, Total Trade Quantity

print("\nTransforming data to bronze schema...")

# Map column names (case-insensitive matching)
# Try different possible column name variations
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

print(f"Detected column mapping: {column_mapping}")

# Select and rename columns
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

# Filter by date range if specified
if start_date and end_date:
    bronze_df = bronze_df.filter(
        (col('date') >= to_date(lit(start_date))) & 
        (col('date') <= to_date(lit(end_date)))
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

# Validate OHLC relationships
print("\nValidating OHLC relationships...")
invalid_ohlc = bronze_df.filter(
    (col('high') < col('low')) |
    (col('high') < col('open')) |
    (col('high') < col('close')) |
    (col('low') > col('open')) |
    (col('low') > col('close'))
)

invalid_count = invalid_ohlc.count()
if invalid_count > 0:
    print(f"Warning: Found {invalid_count} rows with invalid OHLC relationships. These will be removed.")
    print("Sample of invalid rows:")
    invalid_ohlc.show(5)
    # Remove invalid rows
    bronze_df = bronze_df.filter(
        (col('high') >= col('low')) &
        (col('high') >= col('open')) &
        (col('high') >= col('close')) &
        (col('low') <= col('open')) &
        (col('low') <= col('close'))
    )

# Sort by date for consistency
bronze_df = bronze_df.orderBy('date', 'symbol')

spark_df = bronze_df

# Data quality validation
print("\n" + "="*60)
print("Data Quality Checks:")
print("="*60)
total_rows = spark_df.count()
print(f"Total rows: {total_rows}")
print(f"Rows with nulls: {spark_df.filter(col('date').isNull() | 
                                          col('symbol').isNull() | 
                                          col('close').isNull()).count()}")
print(f"Unique symbols: {spark_df.select('symbol').distinct().count()}")

# Date range statistics
date_stats = spark_df.agg(
    {'date': 'min', 'date': 'max'}
).collect()[0]
print(f"Date range: {date_stats[0]} to {date_stats[1]}")

# Price statistics
price_stats = spark_df.agg(
    {'open': 'min', 'open': 'max', 'open': 'avg',
     'close': 'min', 'close': 'max', 'close': 'avg',
     'volume': 'min', 'volume': 'max', 'volume': 'avg'}
).collect()[0]
print(f"\nPrice Statistics:")
print(f"  Open: min={price_stats[0]:.2f}, max={price_stats[1]:.2f}, avg={price_stats[2]:.2f}")
print(f"  Close: min={price_stats[3]:.2f}, max={price_stats[4]:.2f}, avg={price_stats[5]:.2f}")
print(f"  Volume: min={price_stats[6]:,}, max={price_stats[7]:,}, avg={price_stats[8]:,.0f}")

# Show sample data
print("\nSample data (first 10 rows):")
spark_df.show(10, truncate=False)

# Write to Delta table
print(f"\nWriting to Delta table: {bronze_table}")
spark_df.write.format("delta").mode("overwrite").saveAsTable(bronze_table)

print(f"Successfully wrote {spark_df.count()} rows to {bronze_table}")

# Verify table creation
verify_df = spark.read.format("delta").table(bronze_table)
print(f"\nVerification: Table {bronze_table} contains {verify_df.count()} rows")
print("\nTable schema:")
verify_df.printSchema()



