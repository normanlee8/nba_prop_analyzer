import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import logging
import os
import warnings
from prop_analyzer import config as cfg
from prop_analyzer.models import registry

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def get_feature_cols(prop_type):
    """
    Defines the standard feature set for training.
    CRITICAL UPDATE: Added Minutes Volatility and Trends.
    """
    # 1. Player Prop Trends (Mapped from {PROP}_SZN_AVG -> SZN Avg)
    cols = [
        'SZN Avg', 'L5 Avg', 'L10_STD_DEV', 'L5 EWMA',
        'L3 Avg' # Short term form
    ]
    
    # 2. Volatility & Role Features (Global)
    # These allow the model to widen confidence intervals for inconsistent players
    cols.extend([
        'MIN_SZN_AVG', 
        'MIN_L5_AVG', 
        'MIN_L5_STD'  # <--- KEY FEATURE: Minutes Standard Deviation
    ])
    
    # 3. Context Features
    cols.extend([
        'HOME_AWAY_CODE', 'REST_DAYS', 'GAME_PACE',
        'USG_PROXY', 'TS_PCT' # Advanced stats if available
    ])
    
    # 4. Opponent Context
    cols.extend([
        'OPP_DEF_RATING', 'OPP_PACE', 
        'OPP_ALLOWED_AVG', 'OPP_DVP_RANK'
    ])
    
    return cols

def prepare_training_data(df, prop_type, target_col='Actual'):
    """
    Prepares X and y matrices for training.
    """
    if df.empty:
        return None, None, None

    # Filter for rows where target exists
    df = df.dropna(subset=[target_col]).copy()
    
    # Map dynamic prop columns (e.g., PTS_SZN_AVG) to generic feature names (SZN Avg)
    # This allows us to use one feature list for all models
    rename_map = {
        f'{prop_type}_SZN_AVG': 'SZN Avg',
        f'{prop_type}_L5_AVG': 'L5 Avg',
        f'{prop_type}_L10_STD': 'L10_STD_DEV',
        f'{prop_type}_L10_STD_DEV': 'L10_STD_DEV', # Handle alias
        f'{prop_type}_L5_EWMA': 'L5 EWMA',
        f'{prop_type}_L3_AVG': 'L3 Avg',
        f'OPP_{prop_type}_ALLOWED': 'OPP_ALLOWED_AVG',
        f'DVP_RANK_{prop_type}': 'OPP_DVP_RANK'
    }
    
    df_renamed = df.rename(columns=rename_map)
    
    # Select features
    feature_cols = get_feature_cols(prop_type)
    
    # Ensure all columns exist (fill missing with NaN to be handled by Imputer)
    for c in feature_cols:
        if c not in df_renamed.columns:
            df_renamed[c] = np.nan
            
    X = df_renamed[feature_cols]
    y = df_renamed[target_col]
    
    return X, y, feature_cols

def train_model_for_prop(df, prop_type):
    """
    Trains an ensemble of Quantile Regressors (for Range/StdDev) 
    and a Classifier (for Trend/Sanity Check).
    """
    logging.info(f"Training models for {prop_type}...")
    
    X, y, features = prepare_training_data(df, prop_type)
    
    if X is None or len(X) < 100:
        logging.warning(f"Not enough data to train {prop_type} (n={len(X) if X is not None else 0})")
        return None

    # Preprocessing
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    
    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)
    
    # --- 1. QUANTILE REGRESSION (The "S-Tier" Engine) ---
    # We define the range of outcomes: q20 (Low), q50 (Median), q80 (High)
    models = {
        'q20': {},
        'q50': {},
        'q80': {}
    }
    
    quantiles = [0.20, 0.50, 0.80]
    
    for q in quantiles:
        q_key = f"q{int(q*100)}"
        
        # LGBM Quantile
        lgbm = lgb.LGBMRegressor(
            objective='quantile', 
            alpha=q, 
            n_estimators=200, 
            learning_rate=0.05,
            max_depth=5,
            verbose=-1
        )
        lgbm.fit(X_scaled, y)
        
        # XGBoost Quantile (Absolute Error proxy or true quantile if available)
        # Using standard regressor as secondary vote, but optimized for q-loss where supported
        xgb_mod = xgb.XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=q,
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            n_jobs=1
        )
        xgb_mod.fit(X_scaled, y)
        
        models[q_key]['lgbm'] = lgbm
        models[q_key]['xgb'] = xgb_mod

    # --- 2. CLASSIFIER (Trend Detection) ---
    # We create a synthetic binary target: Did player beat their L5 Average?
    # This helps detect players "heating up" vs "cooling down".
    # Note: We do NOT train on Vegas lines here because we don't have historical lines.
    # The S-Tier logic (inference.py) uses the Regression Z-Score for the actual pick.
    # This classifier is just a divergence check.
    
    thresholds = X['L5 Avg'].fillna(y.mean()).values
    # Add slight noise to prevent overfitting to exact averages
    noise = np.random.normal(0, 0.5, size=len(y))
    y_class = (y > (thresholds + noise)).astype(int)
    
    clf = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        eval_metric='logloss',
        use_label_encoder=False
    )
    clf.fit(X_scaled, y_class)

    # Bundle Artifacts
    artifacts = {
        'q20': models['q20'],
        'q50': models['q50'],
        'q80': models['q80'],
        'clf': clf,
        'scaler': scaler,
        'imputer': imputer,
        'features': features
    }
    
    return artifacts

def train_all_models():
    """
    Main entry point for retraining. Loads master dataset and iterates all props.
    """
    logging.info("Loading Master Training Dataset...")
    
    if not cfg.MASTER_TRAINING_FILE.exists():
        logging.error(f"Training data not found: {cfg.MASTER_TRAINING_FILE}")
        logging.error("Please run 'run_build_db.py' first to generate data.")
        return

    try:
        # Load data
        df = pd.read_parquet(cfg.MASTER_TRAINING_FILE)
        logging.info(f"Loaded {len(df)} rows of training data.")
        
        # Verify Volatility Features exist
        if 'MIN_L5_STD' not in df.columns:
            logging.warning(">>> CRITICAL: 'MIN_L5_STD' not found in training data.")
            logging.warning(">>> The models will not learn volatility penalties.")
            logging.warning(">>> ACTION: Run 'run_build_db.py' immediately after this.")
        
        # Define props to train
        props_to_train = [
            'PTS', 'REB', 'AST', 'PRA', 'PR', 'PA', 'RA',
            'FG3M', 'BLK', 'STL', 'TOV', 'FANTASY_PTS',
            # 1H and Q1 models
            'Q1_PTS', 'Q1_REB', 'Q1_AST',
            '1H_PTS', '1H_REB', '1H_AST', '1H_PRA'
        ]
        
        for prop in props_to_train:
            logging.info(f"--- Training {prop} ---")
            artifacts = train_model_for_prop(df, prop)
            
            if artifacts:
                registry.save_artifacts(prop, artifacts)
                logging.info(f"Saved model artifacts for {prop}")
                
        logging.info(">>> Training Complete. Models updated in /prop_models/ <<<")
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise e

if __name__ == "__main__":
    # Setup simple logging for standalone run
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    train_all_models()