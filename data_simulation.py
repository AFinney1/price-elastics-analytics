import polars as pl
import numpy as np
from datetime import datetime, timedelta
import random

def simulate_ecommerce_data(n_skus=1000, n_days=365*3, n_categories=20):
    """Simulate multi-year ecommerce transaction data"""
    
    np.random.seed(42)
    random.seed(42)
    
    # Generate SKU metadata
    categories = [f"Category_{i}" for i in range(n_categories)]
    sub_categories = [f"SubCat_{i}_{j}" for i in range(n_categories) for j in range(3)]
    
    sku_data = []
    for i in range(n_skus):
        category = random.choice(categories)
        # Fix: Extract category number properly
        cat_num = category.split('_')[1]
        # Get subcategories for this category
        matching_subcats = [sc for sc in sub_categories if sc.startswith(f"SubCat_{cat_num}_")]
        sub_category = random.choice(matching_subcats)
        
        # Base price and elasticity vary by category
        base_price = np.random.lognormal(3.5, 0.8)  # Log-normal price distribution
        base_elasticity = -np.random.uniform(0.5, 2.5)  # Negative elasticity
        
        sku_data.append({
            'sku_id': f'SKU_{i:04d}',
            'category': category,
            'sub_category': sub_category,
            'base_price': base_price,
            'base_elasticity': base_elasticity,
            'seasonality_strength': np.random.uniform(0, 0.3),
            'trend_strength': np.random.uniform(-0.0002, 0.0002)
        })
    
    sku_df = pl.DataFrame(sku_data)
    
    # Generate daily transaction data
    start_date = datetime.now() - timedelta(days=n_days)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    all_transactions = []
    
    for date_idx, date in enumerate(dates):
        daily_transactions = []
        
        # Sample active SKUs for this day (not all SKUs sell every day)
        n_active_skus = int(n_skus * np.random.uniform(0.3, 0.7))
        active_skus = np.random.choice(sku_df['sku_id'], n_active_skus, replace=False)
        
        for sku_id in active_skus:
            sku_info = sku_df.filter(pl.col('sku_id') == sku_id)[0]
            
            # Dynamic pricing with random variations
            price_multiplier = 1 + np.random.uniform(-0.2, 0.2)  # ±20% price variation
            
            # Add seasonality
            seasonality = sku_info['seasonality_strength'][0] * np.sin(2 * np.pi * date_idx / 365)
            
            # Add trend
            trend = sku_info['trend_strength'][0] * date_idx
            
            current_price = sku_info['base_price'][0] * price_multiplier * (1 + seasonality + trend)
            
            # Calculate demand based on price elasticity
            price_ratio = current_price / sku_info['base_price'][0]
            elasticity = sku_info['base_elasticity'][0]
            
            # Base demand with price elasticity effect
            base_demand = np.random.poisson(50)
            demand_multiplier = price_ratio ** elasticity
            quantity = max(1, int(base_demand * demand_multiplier * np.random.uniform(0.8, 1.2)))
            
            # Revenue
            revenue = quantity * current_price
            
            daily_transactions.append({
                'date': date,
                'sku_id': sku_id,
                'price': current_price,
                'quantity': quantity,
                'revenue': revenue
            })
        
        all_transactions.extend(daily_transactions)
    
    transactions_df = pl.DataFrame(all_transactions)
    
    # Generate traffic data
    traffic_data = []
    for date in dates:
        # Weekend effect
        is_weekend = date.weekday() in [5, 6]
        base_traffic = 10000 if not is_weekend else 15000
        
        traffic = int(base_traffic * np.random.uniform(0.7, 1.3))
        
        traffic_data.append({
            'date': date,
            'total_traffic': traffic,
            'unique_visitors': int(traffic * 0.7),
            'conversion_rate': np.random.uniform(0.02, 0.05)
        })
    
    traffic_df = pl.DataFrame(traffic_data)
    
    # Generate marketing spend data
    marketing_data = []
    for date in dates:
        marketing_data.append({
            'date': date,
            'display_spend': np.random.uniform(100, 1000),
            'search_spend': np.random.uniform(200, 1500),
            'social_spend': np.random.uniform(50, 500),
            'email_campaigns': np.random.randint(0, 3)
        })
    
    marketing_df = pl.DataFrame(marketing_data)
    
    return transactions_df, sku_df, traffic_df, marketing_df