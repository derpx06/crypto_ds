import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit

def parse_and_join_data(historical_path, sentiment_path):
    """
    Load datasets and merge using IST->UTC aligned date key.
    """
    hd = pd.read_csv(historical_path)
    fg = pd.read_csv(sentiment_path)
    
    # Parse IST timestamps to UTC
    hd['timestamp_ist'] = pd.to_datetime(hd['Timestamp IST'], format='%d-%m-%Y %H:%M', errors='coerce')
    hd['timestamp_utc'] = hd['timestamp_ist'] - pd.Timedelta(hours=5, minutes=30)
    hd['date_utc'] = hd['timestamp_utc'].dt.strftime('%Y-%m-%d')
    
    # Parse sentiment date
    fg['date'] = pd.to_datetime(fg['date'], errors='coerce').dt.strftime('%Y-%m-%d')
    
    # Merge on UTC date
    merged = hd.merge(
        fg[['date', 'timestamp', 'value', 'classification']].rename(columns={
            'timestamp': 'fg_timestamp',
            'value': 'fg_value',
            'classification': 'fg_classification',
        }),
        left_on='date_utc',
        right_on='date',
        how='left'
    )
    return merged

def clean_and_winsorize(df, p_low=0.01, p_high=0.99):
    """
    Clean Side and Crossed indicators, winsorize heavy-tailed variables, and define targets.
    """
    df = df.copy()
    
    # Parse Crossed as boolean
    if df['Crossed'].dtype == object:
        df['Crossed'] = df['Crossed'].astype(str).str.upper().map({'TRUE': True, 'FALSE': False})
    
    # Filter administrative transactions if they do not represent trading intent
    df = df[~df['Direction'].eq('Spot Dust Conversion')].copy()
    
    # Sort chronologically to prevent leakages
    df = df[df['timestamp_utc'].notna()].sort_values('timestamp_utc').reset_index(drop=True)
    
    # Winsorize key variables
    for col in ['Closed PnL', 'Size USD', 'Fee']:
        lo, hi = df[col].quantile([p_low, p_high])
        df[col + '_w'] = df[col].clip(lo, hi)
        
    # Define binary target
    df['y'] = (df['Closed PnL'] > 0).astype(int)
    
    # Log transforms
    df['size_usd_log'] = np.log1p(df['Size USD'].clip(lower=0))
    df['fee_abs_log'] = np.log1p(np.abs(df['Fee']))
    df['exec_price_log'] = np.log1p(df['Execution Price'].clip(lower=0))
    
    # Sentiment regime ordinal mapping
    sent_map = {'Extreme Fear': 0, 'Fear': 1, 'Neutral': 2, 'Greed': 3, 'Extreme Greed': 4}
    df['sent_regime_ord'] = df['fg_classification'].map(sent_map).fillna(2) # Default to neutral if missing
    
    # Time parts
    df['hour'] = df['timestamp_utc'].dt.hour
    df['dow'] = df['timestamp_utc'].dt.dayofweek
    df['month'] = df['timestamp_utc'].dt.month
    
    return df

def generate_features(df):
    """
    Generate >100 features including rolling account stats, coin stats,
    sentiment velocity/lags, and cross-feature interactions.
    All operations use shift(1) to avoid lookahead leakage.
    """
    df = df.copy()
    
    # 1. ACCOUNT-LEVEL ROLLING WINDOW FEATURES
    g = df.groupby('Account', sort=False)
    
    # Helper to clean indexing
    def get_rolling_stat(group_obj, col, window, stat, shift=1):
        rolled = group_obj[col].shift(shift)
        if stat == 'mean':
            return rolled.rolling(window, min_periods=max(2, window//4)).mean().reset_index(level=0, drop=True)
        elif stat == 'std':
            return rolled.rolling(window, min_periods=max(2, window//4)).std().reset_index(level=0, drop=True)
        elif stat == 'sum':
            return rolled.rolling(window, min_periods=max(2, window//4)).sum().reset_index(level=0, drop=True)
        elif stat == 'max':
            return rolled.rolling(window, min_periods=max(2, window//4)).max().reset_index(level=0, drop=True)
        elif stat == 'min':
            return rolled.rolling(window, min_periods=max(2, window//4)).min().reset_index(level=0, drop=True)
        elif stat == 'ewm':
            return rolled.ewm(span=window, adjust=False).mean().reset_index(level=0, drop=True)

    # Rolling win rates (prior account performance)
    for w in [5, 10, 20, 50, 100]:
        df[f'acc_wr{w}'] = get_rolling_stat(g, 'y', w, 'mean')
        
    # Rolling PnL context
    for w in [10, 20, 50]:
        df[f'acc_pnl_mean_{w}'] = get_rolling_stat(g, 'Closed PnL_w', w, 'mean')
        df[f'acc_pnl_std_{w}'] = get_rolling_stat(g, 'Closed PnL_w', w, 'std')
        df[f'acc_pnl_max_{w}'] = get_rolling_stat(g, 'Closed PnL_w', w, 'max')
        df[f'acc_pnl_min_{w}'] = get_rolling_stat(g, 'Closed PnL_w', w, 'min')
        df[f'acc_pnl_velocity_{w}'] = get_rolling_stat(g, 'Closed PnL_w', w, 'ewm')
        
    # Rolling size and execution context
    df['crossed_int'] = df['Crossed'].astype(float)
    for w in [10, 20, 50]:
        df[f'acc_size_mean_{w}'] = get_rolling_stat(g, 'Size USD', w, 'mean')
        df[f'acc_size_std_{w}'] = get_rolling_stat(g, 'Size USD', w, 'std')
        df[f'acc_fee_mean_{w}'] = get_rolling_stat(g, 'Fee', w, 'mean')
        df[f'acc_taker_rate_{w}'] = get_rolling_stat(g, 'crossed_int', w, 'mean')
        
    # Consecutive trade streaks (win/loss streaks)
    def calc_streak(x):
        # x is a pandas Series of targets
        shifted = x.shift(1).fillna(0)
        streak = np.zeros(len(x))
        curr = 0
        for idx, val in enumerate(shifted):
            if val == 1:
                curr = curr + 1 if curr > 0 else 1
            else:
                curr = curr - 1 if curr < 0 else -1
            streak[idx] = curr
        return pd.Series(streak, index=x.index)
        
    df['acc_streak'] = g['y'].apply(calc_streak).reset_index(level=0, drop=True)
    
    # 2. COIN-LEVEL ROLLING WINDOW FEATURES
    coin = df.groupby('Coin', sort=False)
    for w in [10, 30, 60]:
        df[f'coin_wr{w}'] = get_rolling_stat(coin, 'y', w, 'mean')
        df[f'coin_pnl{w}'] = get_rolling_stat(coin, 'Closed PnL_w', w, 'mean')
        df[f'coin_taker_rate_{w}'] = get_rolling_stat(coin, 'crossed_int', w, 'mean')
        
    # 3. SENTIMENT TIMELINE FEATURES
    daily = df[['date_utc', 'fg_value', 'sent_regime_ord']].drop_duplicates('date_utc').sort_values('date_utc').copy()
    
    # Lags (shift daily metrics)
    for lag in [1, 2, 3, 5, 7, 10, 14, 21, 30]:
        daily[f'fg_l{lag}'] = daily['fg_value'].shift(lag)
        daily[f'fg_regime_l{lag}'] = daily['sent_regime_ord'].shift(lag)
        
    # Delta (Velocity)
    for lag in [1, 3, 7, 14]:
        daily[f'fg_d{lag}'] = daily['fg_value'] - daily[f'fg_l{lag}']
        
    # Rolling Statistics
    for w in [7, 14, 30]:
        daily[f'fg_r{w}m'] = daily['fg_value'].rolling(w, min_periods=max(2, w//3)).mean().shift(1)
        daily[f'fg_r{w}s'] = daily['fg_value'].rolling(w, min_periods=max(2, w//3)).std().shift(1)
        
    # Merge daily timeline sentiment features back
    sent_cols = [c for c in daily.columns if c not in ['fg_value', 'sent_regime_ord', 'date_utc']]
    df = df.merge(daily[['date_utc'] + sent_cols], on='date_utc', how='left')
    
    # 4. MICROSTRUCTURE & DIRECTION FEATURES
    df['is_buy'] = (df['Side'].astype(str).str.upper() == 'BUY').astype(int)
    df['is_sell'] = (df['Side'].astype(str).str.upper() == 'SELL').astype(int)
    
    # 5. CROSS-PRODUCT INTERACTIONS
    df['sent_x_taker'] = df['sent_regime_ord'] * df['crossed_int'].fillna(0)
    df['sent_x_accwr'] = df['sent_regime_ord'] * df['acc_wr20'].fillna(0.5)
    df['sent_x_accsize'] = df['sent_regime_ord'] * df['acc_size_mean_20'].fillna(0.0)
    df['sent_x_coinwr'] = df['sent_regime_ord'] * df['coin_wr30'].fillna(0.5)
    
    df['size_x_sent'] = df['size_usd_log'] * df['fg_value'].fillna(50)
    df['fee_x_taker'] = df['fee_abs_log'] * df['crossed_int'].fillna(0)
    
    # Fill standard numeric NaNs to maintain dimension stability
    return df

def build_purged_walkforward_splits(data, n_splits=5, purge_days=1, embargo_days=1):
    """
    Build walk-forward training/validation splits with purging and embargoing.
    This strictly avoids overlaps to control label leakages in cross-validation.
    """
    data = data.sort_values('timestamp_utc').reset_index(drop=True)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    ts = data['timestamp_utc']
    folds = []
    
    for i, (tr, te) in enumerate(tscv.split(data), 1):
        test_start = ts.iloc[te].min()
        test_end = ts.iloc[te].max()
        
        # Purge window: Exclude training labels just before test start
        purge_start = test_start - pd.Timedelta(days=purge_days)
        # Embargo window: Exclude training labels just after test end
        embargo_end = test_end + pd.Timedelta(days=embargo_days)
        
        # Keep training samples that are outside the purge-embargo zone
        tr_mask = (ts.iloc[tr] < purge_start) | (ts.iloc[tr] > embargo_end)
        tr_clean = tr[tr_mask.values]
        
        if len(tr_clean) > 0:
            folds.append((i, tr_clean, te))
            
    return folds

def bootstrap_ci(arr, n_boot=1000, alpha=0.05, seed=42):
    """
    Compute Bootstrap Confidence Intervals for an array.
    """
    arr = np.asarray(arr)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return np.mean(arr), lo, hi

from sklearn.base import BaseEstimator, ClassifierMixin

class StackingClassifierPipeline(ClassifierMixin, BaseEstimator):
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model
        
    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        for m in self.base_models.values():
            m.fit(X, y)
        meta_features = np.column_stack([m.predict_proba(X)[:, 1] for m in self.base_models.values()])
        self.meta_model.fit(meta_features, y)
        return self
        
    def predict_proba(self, X):
        meta_features = np.column_stack([m.predict_proba(X)[:, 1] for m in self.base_models.values()])
        p = self.meta_model.predict_proba(meta_features)
        return p
        
    def predict(self, X):
        meta_features = np.column_stack([m.predict_proba(X)[:, 1] for m in self.base_models.values()])
        return self.meta_model.predict(meta_features)
