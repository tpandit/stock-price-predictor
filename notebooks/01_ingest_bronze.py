"""
Phase 1: Data Ingestion (Bronze Layer)
Generate mock stock data and write to Delta table
"""
import sys
import os

# Get the notebook path and add parent directory to Python path
try:
    # Get current notebook path
    notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    # Extract workspace path (everything before /notebooks/)
    workspace_path = "/".join(notebook_path.split("/")[:-1])
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

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, DateType
from pyspark.sql.functions import col, to_date
from utils.helpers import load_config, get_spark_session

# Load configuration - use absolute path
try:
    config_path = f"{workspace_path}/config.yaml"
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)
except Exception as e:
    print(f"Warning: Could not use absolute path, trying relative: {e}")
    config = load_config("config.yaml")
symbols = config['symbols']
start_date = config['date_range']['start_date']
end_date = config['date_range']['end_date']
bronze_table = config['delta_tables']['bronze']

# Initialize Spark session
spark = get_spark_session("StockIngestion")

print(f"Starting mock data generation for symbols: {symbols}")
print(f"Date range: {start_date} to {end_date}")
print("Note: Using mock data generation (no API calls required)")

# Define expected schema
bronze_schema = StructType([
    StructField("date", DateType(), True),
    StructField("symbol", StringType(), True),
    StructField("open", DoubleType(), True),
    StructField("high", DoubleType(), True),
    StructField("low", DoubleType(), True),
    StructField("close", DoubleType(), True),
    StructField("volume", LongType(), True)
])


def generate_mock_stock_data(symbol, start_date, end_date, seed=None):
    """
    Generate realistic mock stock price data (OHLCV) for a given symbol and date range.
    
    Args:
        symbol: Stock symbol
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        seed: Random seed for reproducibility (uses symbol hash if None)
    
    Returns:
        pandas DataFrame with OHLCV data
    """
    # Use symbol to set seed for reproducibility
    if seed is None:
        seed = hash(symbol) % 10000
    np.random.seed(seed)
    
    # Create date range (business days only)
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    dates = pd.bdate_range(start=start, end=end, freq='B')  # B = business days
    
    if len(dates) == 0:
        raise ValueError(f"No business days in date range {start_date} to {end_date}")
    
    # Base prices for different symbols (realistic starting points)
    base_prices = {
        'AAPL': 150.0,
        'MSFT': 250.0,
        'AMZN': 100.0,
        'GOOGL': 120.0,
        'TSLA': 200.0,
    }
    
    # Default base price if symbol not in dict
    base_price = base_prices.get(symbol, 100.0)
    
    # Generate price data using random walk with drift
    n_days = len(dates)
    
    # Daily returns (normal distribution with slight positive drift)
    daily_returns = np.random.normal(0.0005, 0.02, n_days)  # ~0.05% daily drift, 2% volatility
    
    # Generate close prices using cumulative returns
    close_prices = base_price * np.exp(np.cumsum(daily_returns))
    
    # Generate OHLC from close prices
    data = []
    prev_close = base_price
    
    for i, date in enumerate(dates):
        close = close_prices[i]
        
        # Open price: close to previous close with some gap
        gap = np.random.normal(0, 0.005) * prev_close  # Small gap
        open_price = prev_close * (1 + gap)
        
        # High and Low: based on open and close with realistic ranges
        price_range = abs(close - open_price) + np.random.uniform(0.01, 0.03) * open_price
        high = max(open_price, close) + np.random.uniform(0, price_range * 0.5)
        low = min(open_price, close) - np.random.uniform(0, price_range * 0.5)
        
        # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # Volume: base volume with some randomness and correlation to price movement
        base_volume = 10_000_000  # 10M shares base
        volatility_factor = abs(close - open_price) / open_price
        volume = int(base_volume * (1 + volatility_factor * 2) * np.random.uniform(0.5, 2.0))
        
        data.append({
            'date': date,
            'symbol': symbol,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': volume
        })
        
        prev_close = close
    
    df = pd.DataFrame(data)
    return df


# Generate mock data for each symbol
all_data = []

for symbol in symbols:
    try:
        print(f"Generating mock data for {symbol}...")
        df = generate_mock_stock_data(symbol, start_date, end_date)
        
        # Reorder columns
        df = df[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
        
        # Data quality checks
        # Remove rows with null values in critical columns
        df = df.dropna(subset=['date', 'open', 'high', 'low', 'close', 'volume'])
        
        # Validate OHLC relationships
        assert (df['high'] >= df['low']).all(), f"High < Low found for {symbol}"
        assert (df['high'] >= df['open']).all(), f"High < Open found for {symbol}"
        assert (df['high'] >= df['close']).all(), f"High < Close found for {symbol}"
        assert (df['low'] <= df['open']).all(), f"Low > Open found for {symbol}"
        assert (df['low'] <= df['close']).all(), f"Low > Close found for {symbol}"
        
        all_data.append(df)
        print(f"Successfully generated {len(df)} rows for {symbol}")
        
    except Exception as e:
        print(f"Error generating data for {symbol}: {str(e)}")
        continue

if not all_data:
    raise ValueError("No data was successfully generated for any symbol")

# Combine all dataframes
combined_df = pd.concat(all_data, ignore_index=True)

# Convert to Spark DataFrame
spark_df = spark.createDataFrame(combined_df, schema=bronze_schema)

# Ensure date column is properly typed
spark_df = spark_df.withColumn("date", to_date(col("date")))

# Data quality validation
print("\nData Quality Checks:")
print(f"Total rows: {spark_df.count()}")
print(f"Rows with nulls: {spark_df.filter(col('date').isNull() | 
                                          col('symbol').isNull() | 
                                          col('close').isNull()).count()}")
print(f"Unique symbols: {spark_df.select('symbol').distinct().count()}")

# Show sample data
print("\nSample data:")
spark_df.show(10)

# Write to Delta table
print(f"\nWriting to Delta table: {bronze_table}")
spark_df.write.format("delta").mode("overwrite").saveAsTable(bronze_table)

print(f"Successfully wrote {spark_df.count()} rows to {bronze_table}")

# Verify table creation
verify_df = spark.read.format("delta").table(bronze_table)
print(f"\nVerification: Table {bronze_table} contains {verify_df.count()} rows")
print("\nTable schema:")
verify_df.printSchema()



