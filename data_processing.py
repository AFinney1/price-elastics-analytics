import polars as pl
import numpy as np

def clean_and_merge_data(transactions_df, traffic_df, marketing_df, sku_df):
    """Clean and merge all data sources"""
    
    # Remove outliers in transactions
    q1 = transactions_df['price'].quantile(0.01)
    q99 = transactions_df['price'].quantile(0.99)
    
    transactions_clean = transactions_df.filter(
        (pl.col('price') >= q1) & 
        (pl.col('price') <= q99) &
        (pl.col('quantity') > 0)
    )
    
    # Aggregate daily data by SKU
    daily_sku_agg = transactions_clean.group_by(['date', 'sku_id']).agg([
        pl.col('price').mean().alias('avg_price'),
        pl.col('quantity').sum().alias('total_quantity'),
        pl.col('revenue').sum().alias('total_revenue')
    ])
    
    # Merge with traffic data
    merged_df = daily_sku_agg.join(traffic_df, on='date', how='left')
    
    # Merge with marketing data
    merged_df = merged_df.join(marketing_df, on='date', how='left')
    
    # Merge with SKU metadata
    merged_df = merged_df.join(sku_df, on='sku_id', how='left')
    
    # Add derived features
    merged_df = merged_df.with_columns([
        # Log transformations for elasticity models
        pl.col('avg_price').log().alias('log_price'),
        pl.col('total_quantity').log().alias('log_quantity'),
        
        # Total marketing spend
        (pl.col('display_spend') + pl.col('search_spend') + pl.col('social_spend')).alias('total_marketing_spend'),
        
        # Day of week
        pl.col('date').dt.weekday().alias('day_of_week'),
        
        # Month
        pl.col('date').dt.month().alias('month'),
        
        # Quarter
        pl.col('date').dt.quarter().alias('quarter')
    ])
    
    return merged_df

def prepare_model_data(merged_df, sku_id=None):
    """Prepare data for modeling"""
    
    if sku_id:
        model_df = merged_df.filter(pl.col('sku_id') == sku_id)
    else:
        model_df = merged_df
    
    # Remove rows with null values in key columns
    model_df = model_df.drop_nulls(subset=['log_price', 'log_quantity'])
    
    # Sort by date
    model_df = model_df.sort('date')
    
    return model_df

def validate_uploaded_data(transactions_df, skus_df, traffic_df, marketing_df):
    """Validate that uploaded data has required columns and correct data types"""
    
    errors = []
    
    # Check transactions columns
    required_transaction_cols = ['date', 'sku_id', 'price', 'quantity', 'revenue']
    missing_trans_cols = set(required_transaction_cols) - set(transactions_df.columns)
    if missing_trans_cols:
        errors.append(f"Transactions file missing columns: {missing_trans_cols}")
    
    # Check SKU columns
    required_sku_cols = ['sku_id', 'category', 'sub_category', 'base_price']
    missing_sku_cols = set(required_sku_cols) - set(skus_df.columns)
    if missing_sku_cols:
        errors.append(f"SKUs file missing columns: {missing_sku_cols}")
    
    # Check traffic columns
    required_traffic_cols = ['date', 'total_traffic', 'unique_visitors', 'conversion_rate']
    missing_traffic_cols = set(required_traffic_cols) - set(traffic_df.columns)
    if missing_traffic_cols:
        errors.append(f"Traffic file missing columns: {missing_traffic_cols}")
    
    # Check marketing columns
    required_marketing_cols = ['date', 'display_spend', 'search_spend', 'social_spend']
    missing_marketing_cols = set(required_marketing_cols) - set(marketing_df.columns)
    if missing_marketing_cols:
        errors.append(f"Marketing file missing columns: {missing_marketing_cols}")
    
    return errors

def process_uploaded_files(transactions_file, skus_file, traffic_file, marketing_file):
    """Process uploaded CSV files and convert to Polars DataFrames"""
    
    try:
        # Read CSV files
        transactions_df = pl.read_csv(transactions_file)
        skus_df = pl.read_csv(skus_file)
        traffic_df = pl.read_csv(traffic_file)
        marketing_df = pl.read_csv(marketing_file)
        
        # Convert date columns to datetime
        transactions_df = transactions_df.with_columns(
            pl.col('date').str.to_datetime().cast(pl.Date)
        )
        traffic_df = traffic_df.with_columns(
            pl.col('date').str.to_datetime().cast(pl.Date)
        )
        marketing_df = marketing_df.with_columns(
            pl.col('date').str.to_datetime().cast(pl.Date)
        )
        
        # Ensure numeric columns are properly typed
        transactions_df = transactions_df.with_columns([
            pl.col('price').cast(pl.Float64),
            pl.col('quantity').cast(pl.Int64),
            pl.col('revenue').cast(pl.Float64)
        ])
        
        skus_df = skus_df.with_columns([
            pl.col('base_price').cast(pl.Float64)
        ])
        
        traffic_df = traffic_df.with_columns([
            pl.col('total_traffic').cast(pl.Int64),
            pl.col('unique_visitors').cast(pl.Int64),
            pl.col('conversion_rate').cast(pl.Float64)
        ])
        
        marketing_df = marketing_df.with_columns([
            pl.col('display_spend').cast(pl.Float64),
            pl.col('search_spend').cast(pl.Float64),
            pl.col('social_spend').cast(pl.Float64)
        ])
        
        # Add optional columns if missing
        if 'email_campaigns' not in marketing_df.columns:
            marketing_df = marketing_df.with_columns(
                pl.lit(0).alias('email_campaigns')
            )
        
        if 'base_elasticity' not in skus_df.columns:
            skus_df = skus_df.with_columns(
                pl.lit(-1.5).alias('base_elasticity')
            )
        
        if 'seasonality_strength' not in skus_df.columns:
            skus_df = skus_df.with_columns(
                pl.lit(0.1).alias('seasonality_strength')
            )
        
        if 'trend_strength' not in skus_df.columns:
            skus_df = skus_df.with_columns(
                pl.lit(0.0001).alias('trend_strength')
            )
        
        # Validate data
        errors = validate_uploaded_data(transactions_df, skus_df, traffic_df, marketing_df)
        
        if errors:
            return None, None, None, None, errors
        
        return transactions_df, skus_df, traffic_df, marketing_df, []
        
    except Exception as e:
        return None, None, None, None, [f"Error processing files: {str(e)}"]