import numpy as np
import polars as pl
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
try:
    from linearmodels.panel import PanelOLS
    PANEL_AVAILABLE = True
except ImportError:
    PANEL_AVAILABLE = False
    print("Warning: linearmodels not available. Fixed effects model will use alternative implementation.")
import pymc as pm
import arviz as az
from scipy import stats

class PriceElasticityModels:
    
    @staticmethod
    def ols_log_log(data_df, confidence_level=0.95):
        """Simple OLS log-log model for price elasticity"""
        results = {}
        
        # Convert to pandas for statsmodels
        if isinstance(data_df, pl.DataFrame):
            df = data_df.to_pandas()
        else:
            df = data_df
        
        # Group by SKU
        for sku in df['sku_id'].unique():
            sku_data = df[df['sku_id'] == sku].copy()
            
            if len(sku_data) < 30:  # Minimum observations
                continue
            
            # Add controls
            X = sku_data[['log_price', 'total_marketing_spend', 'total_traffic']]
            X = sm.add_constant(X)
            y = sku_data['log_quantity']
            
            try:
                model = OLS(y, X)
                res = model.fit()
                
                # Extract elasticity (coefficient of log_price)
                elasticity = res.params['log_price']
                std_error = res.bse['log_price']
                conf_int = res.conf_int(alpha=1-confidence_level).loc['log_price']
                
                results[sku] = {
                    'elasticity': elasticity,
                    'std_error': std_error,
                    'conf_int_lower': conf_int[0],
                    'conf_int_upper': conf_int[1],
                    'r_squared': res.rsquared,
                    'n_obs': len(sku_data),
                    'p_value': res.pvalues['log_price']
                }
            except Exception as e:
                print(f"Error for SKU {sku}: {e}")
                continue
        
        return results
    
    @staticmethod
    def fixed_effects_panel(data_df, confidence_level=0.95):
        """Fixed effects panel model"""
        # Convert to pandas
        if isinstance(data_df, pl.DataFrame):
            df = data_df.to_pandas()
        else:
            df = data_df.copy()
        
        # Ensure we have the required columns
        required_cols = ['sku_id', 'date', 'log_quantity', 'log_price', 'total_marketing_spend', 'total_traffic']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Missing columns for Fixed Effects model: {missing_cols}")
            return {}
        
        # Remove any completely null rows
        df = df.dropna(subset=['log_quantity', 'log_price'], how='all')
        
        if len(df) == 0:
            print("No valid data for Fixed Effects model after removing nulls")
            return {}

        
        if PANEL_AVAILABLE:
            try:
                # Use linearmodels PanelOLS
                # Set multi-index
                df = df.set_index(['sku_id', 'date'])
                df = df.sort_index()
                
                # Dependent and independent variables
                y = df['log_quantity']
                X = df[['log_price', 'total_marketing_spend', 'total_traffic']]
                
                # Fit panel model with entity effects
                model = PanelOLS(y, X, entity_effects=True)
                res = model.fit()
                
                # Extract results for each SKU
                results = {}
                
                # Global elasticity
                global_elasticity = res.params['log_price']
                
                # Get confidence intervals - using level not alpha
                conf_int = res.conf_int(level=confidence_level)
                
                for sku in df.index.get_level_values(0).unique():
                    results[sku] = {
                        'elasticity': global_elasticity,  # Same for all in FE model
                        'std_error': res.std_errors['log_price'],
                        'conf_int_lower': conf_int.loc['log_price', 'lower'],
                        'conf_int_upper': conf_int.loc['log_price', 'upper'],
                        'r_squared': res.rsquared,
                        'n_obs': len(df.loc[sku]),
                        'p_value': res.pvalues['log_price']
                    }
                
                return results
                    
            except Exception as e:
                print(f"Error in panel model: {e}")
                # Fall back to alternative implementation
                df = df.reset_index() if isinstance(df.index, pd.MultiIndex) else df
                return PriceElasticityModels._fixed_effects_alternative(df, confidence_level)
                
        else:
            # Alternative implementation using statsmodels with dummy variables
            return PriceElasticityModels._fixed_effects_alternative(df, confidence_level)

    @staticmethod
    def _fixed_effects_alternative(df, confidence_level=0.95):
        """Alternative fixed effects implementation using dummy variables"""
        results = {}
        
        # Reset index if needed
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
        
        # Ensure numeric types for the columns we'll use
        numeric_cols = ['log_price', 'log_quantity', 'total_marketing_spend', 'total_traffic']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with NaN in critical columns
        df = df.dropna(subset=['log_price', 'log_quantity'])
        
        if len(df) == 0:
            print("No valid data after removing NaN values")
            return {}
        
        # Create dummy variables for SKUs (fixed effects)
        # Limit number of dummies to avoid memory issues
        unique_skus = df['sku_id'].unique()
        if len(unique_skus) > 100:
            # For large number of SKUs, use demeaning approach instead
            return PriceElasticityModels._fixed_effects_demeaned(df, confidence_level)
        
        sku_dummies = pd.get_dummies(df['sku_id'], prefix='sku', drop_first=True)
        
        # Combine features
        X = pd.concat([
            df[['log_price', 'total_marketing_spend', 'total_traffic']],
            sku_dummies
        ], axis=1)
        
        # Ensure all columns are numeric
        X = X.apply(pd.to_numeric, errors='coerce')
        X = sm.add_constant(X)
        y = df['log_quantity']
        
        # Remove any rows with NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        df_clean = df[mask]
        
        if len(X) == 0:
            print("No valid data after cleaning")
            return {}
        
        try:
            model = OLS(y, X)
            res = model.fit()
            
            # Extract elasticity
            elasticity = res.params['log_price']
            std_error = res.bse['log_price']
            conf_int = res.conf_int(alpha=1-confidence_level).loc['log_price']
            
            # Apply same elasticity to all SKUs (as in fixed effects)
            for sku in df_clean['sku_id'].unique():
                sku_data = df_clean[df_clean['sku_id'] == sku]
                
                # Get SKU-specific effect if available
                sku_effect = 0
                sku_col = f'sku_{sku}'
                if sku_col in res.params:
                    sku_effect = res.params[sku_col]
                
                results[sku] = {
                    'elasticity': elasticity,
                    'std_error': std_error,
                    'conf_int_lower': conf_int[0],
                    'conf_int_upper': conf_int[1],
                    'entity_effect': sku_effect,
                    'r_squared': res.rsquared,
                    'n_obs': len(sku_data),
                    'p_value': res.pvalues['log_price']
                }
        except Exception as e:
            print(f"Error in fixed effects model: {e}")
            return {}
        
        return results
    
    @staticmethod
    def _fixed_effects_demeaned(df, confidence_level=0.95):
        """Fixed effects using demeaning approach for large number of entities"""
        results = {}
        
        # Ensure numeric types
        numeric_cols = ['log_price', 'log_quantity', 'total_marketing_spend', 'total_traffic']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove NaN values
        df = df.dropna(subset=['log_price', 'log_quantity'])
        
        # Demean variables by SKU
        df_demeaned = df.copy()
        
        # Group by SKU and demean
        for col in ['log_price', 'log_quantity', 'total_marketing_spend', 'total_traffic']:
            df_demeaned[f'{col}_demeaned'] = df.groupby('sku_id')[col].transform(lambda x: x - x.mean())
        
        # Run regression on demeaned variables
        X = df_demeaned[['log_price_demeaned', 'total_marketing_spend_demeaned', 'total_traffic_demeaned']]
        X = sm.add_constant(X)
        y = df_demeaned['log_quantity_demeaned']
        
        # Remove rows with NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        df_clean = df[mask]
        
        try:
            model = OLS(y, X)
            res = model.fit()
            
            # Extract elasticity
            elasticity = res.params['log_price_demeaned']
            std_error = res.bse['log_price_demeaned']
            conf_int = res.conf_int(alpha=1-confidence_level).loc['log_price_demeaned']
            
            # Apply same elasticity to all SKUs
            for sku in df_clean['sku_id'].unique():
                sku_data = df_clean[df_clean['sku_id'] == sku]
                
                results[sku] = {
                    'elasticity': elasticity,
                    'std_error': std_error,
                    'conf_int_lower': conf_int[0],
                    'conf_int_upper': conf_int[1],
                    'r_squared': res.rsquared,
                    'n_obs': len(sku_data),
                    'p_value': res.pvalues['log_price_demeaned']
                }
                
        except Exception as e:
            print(f"Error in demeaned fixed effects: {e}")
            return {}
        
        return results
    
    @staticmethod
    def hierarchical_bayes(data_df, n_samples=2000):
        """Hierarchical Bayesian model for price elasticity"""
        # Convert to pandas
        if isinstance(data_df, pl.DataFrame):
            df = data_df.to_pandas()
        else:
            df = data_df.copy()
        
        # Remove NaN values
        df = df.dropna(subset=['log_price', 'log_quantity', 'category', 'sku_id'])
        
        # Encode SKUs as integers
        sku_encoder = {sku: i for i, sku in enumerate(df['sku_id'].unique())}
        df['sku_idx'] = df['sku_id'].map(sku_encoder)
        
        # Encode categories
        cat_encoder = {cat: i for i, cat in enumerate(df['category'].unique())}
        df['cat_idx'] = df['category'].map(cat_encoder)
        
        n_skus = len(sku_encoder)
        n_cats = len(cat_encoder)
        n_obs = len(df)
        
        # Standardize features for better sampling
        df['log_price_std'] = (df['log_price'] - df['log_price'].mean()) / df['log_price'].std()
        df['marketing_std'] = (df['total_marketing_spend'] - df['total_marketing_spend'].mean()) / df['total_marketing_spend'].std()
        df['traffic_std'] = (df['total_traffic'] - df['total_traffic'].mean()) / df['total_traffic'].std()
        
        with pm.Model() as model:
            # Hyperpriors
            mu_elasticity = pm.Normal('mu_elasticity', mu=-1.5, sigma=1)
            sigma_elasticity = pm.HalfNormal('sigma_elasticity', sigma=0.5)
            
            # Category-level effects
            category_effect = pm.Normal('category_effect', mu=0, sigma=0.5, shape=n_cats)
            
            # SKU-specific elasticities - one per SKU
            elasticity_by_sku = pm.Normal('elasticity_by_sku', 
                                        mu=mu_elasticity,
                                        sigma=sigma_elasticity,
                                        shape=n_skus)
            
            # Add category effect to SKU elasticities
            # We need to map each SKU to its category
            sku_to_cat = np.zeros(n_skus, dtype=int)
            for sku, sku_idx in sku_encoder.items():
                cat = df[df['sku_id'] == sku]['category'].iloc[0]
                sku_to_cat[sku_idx] = cat_encoder[cat]
            
            # SKU elasticities with category effects
            elasticity_with_cat = elasticity_by_sku + category_effect[sku_to_cat]
            
            # Other coefficients
            beta_marketing = pm.Normal('beta_marketing', mu=0, sigma=0.1)
            beta_traffic = pm.Normal('beta_traffic', mu=0, sigma=0.1)
            
            # Model - map each observation to its SKU's elasticity
            sku_indices = df['sku_idx'].values
            
            mu_quantity = (elasticity_with_cat[sku_indices] * df['log_price_std'].values +
                        beta_marketing * df['marketing_std'].values +
                        beta_traffic * df['traffic_std'].values)
            
            # Likelihood
            sigma_obs = pm.HalfNormal('sigma_obs', sigma=1)
            quantity = pm.Normal('quantity', mu=mu_quantity, sigma=sigma_obs, 
                            observed=df['log_quantity'].values)
            
            # Sample
            trace = pm.sample(n_samples, return_inferencedata=True, 
                            progressbar=True, chains=2, target_accept=0.9)
        
        # Extract results
        results = {}
        posterior = trace.posterior
        
        for sku, idx in sku_encoder.items():
            # Get samples for this SKU including category effect
            cat_idx = sku_to_cat[idx]
            sku_base_samples = posterior['elasticity_by_sku'].isel(elasticity_by_sku_dim_0=idx).values.flatten()
            cat_effect_samples = posterior['category_effect'].isel(category_effect_dim_0=cat_idx).values.flatten()
            
            # Total elasticity for the SKU
            sku_samples = sku_base_samples + cat_effect_samples
            
            results[sku] = {
                'elasticity': np.mean(sku_samples),
                'std_error': np.std(sku_samples),
                'conf_int_lower': np.percentile(sku_samples, 2.5),
                'conf_int_upper': np.percentile(sku_samples, 97.5),
                'samples': sku_samples
            }
        
        return results, trace