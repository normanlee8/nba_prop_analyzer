import sys
import pandas as pd
import numpy as np
import logging
import xgboost as xgb
import lightgbm as lgb
import re
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from prop_analyzer import config as cfg
from prop_analyzer.config import Cols
from prop_analyzer.features import definitions as feat_defs
from prop_analyzer.models import registry
from prop_analyzer.utils import common

# Constants
TEST_SET_SIZE_PCT = 0.20
MIN_SAMPLES = 200

# Map Prop Categories to Data Column Prefixes
PROP_KEY_MAP = {
    'Points': 'PTS', 'Rebounds': 'REB', 'Assists': 'AST',
    'Threes': 'FG3M', 'Steals': 'STL', 'Blocks': 'BLK', 'Turnovers': 'TOV',
    
    'FG Attempted': 'FGA', '3s Attempted': 'FG3A',
    'Double Doubles': 'DD', 'Double Double': 'DD',
    'Triple Doubles': 'TD', 'Triple Double': 'TD',
    
    'PRA': 'PRA', 'Pts+Reb+Ast': 'PRA', 
    'Pts+Reb': 'PR', 'Pts+Ast': 'PA', 'Reb+Ast': 'RA',
    'Fantasy Points': 'FANTASY_PTS',
    
    # Q1 / 1H Specific Maps
    'Q1_PTS': 'Q1_PTS', 'Q1_REB': 'Q1_REB', 'Q1_AST': 'Q1_AST',
    'Q1_PRA': 'Q1_PRA', 'Q1_PR': 'Q1_PR', 'Q1_PA': 'Q1_PA', 'Q1_RA': 'Q1_RA',
    'Q1_FG3M': 'Q1_FG3M',
    
    '1H_PTS': '1H_PTS', '1H_REB': '1H_REB', '1H_AST': '1H_AST',
    '1H_PRA': '1H_PRA', '1H_PR': '1H_PR', '1H_PA': '1H_PA', '1H_RA': '1H_RA',
    '1H_FG3M': '1H_FG3M',

    # Direct mappings
    'PTS': 'PTS', 'REB': 'REB', 'AST': 'AST', 'FG3M': 'FG3M',
    'STL': 'STL', 'BLK': 'BLK', 'TOV': 'TOV', 'STK': 'STK',
    'FGA': 'FGA', 'FG3A': 'FG3A', 'DD': 'DD', 'TD': 'TD'
}

def calculate_time_decay_weights(df, date_col, decay_rate=0.002):
    """
    Calculates sample weights based on exponential time decay.
    Recent games get higher weight.
    decay_rate 0.002 -> Half-life of approx 350 days (1 year).
    """
    if date_col not in df.columns:
        return pd.Series(1.0, index=df.index)
        
    # Ensure datetime
    dates = pd.to_datetime(df[date_col])
    max_date = dates.max()
    
    # Calculate days diff
    days_diff = (max_date - dates).dt.days
    
    # Calculate weight: exp(-decay * days)
    weights = np.exp(-decay_rate * days_diff)
    return weights

def add_interaction_features(df):
    """
    Creates interaction features (e.g., Position * Vacancy) to help
    Tree models find the correlation between vacancy and position faster.
    """
    if 'Primary_Pos' in df.columns:
        # Vectorized string check
        is_guard = df['Primary_Pos'].astype(str).str.contains('G', case=False, regex=True).astype(int)
        is_forward = df['Primary_Pos'].astype(str).str.contains('F', case=False, regex=True).astype(int)
        
        if 'MISSING_USG_G' in df.columns:
            df['INT_GUARD_VACANCY'] = is_guard * df['MISSING_USG_G']
            
        if 'MISSING_USG_F' in df.columns:
            df['INT_FORWARD_VACANCY'] = is_forward * df['MISSING_USG_F']
            
    return df

def rename_features_for_model(df, prop_cat):
    """
    Maps specific column names (e.g., PTS_SZN_AVG) to generic definition names (e.g., SZN Avg).
    """
    prefix = PROP_KEY_MAP.get(prop_cat, prop_cat)
    
    mapping = {
        f'{prefix}_{Cols.SZN_AVG}': 'SZN Avg',
        f'{prefix}_{Cols.L5_AVG}': 'L5 Avg',  
        f'{prefix}_L5_EWMA': 'L5 EWMA',
        f'{prefix}_L3_AVG': 'L3 Avg',
        f'{prefix}_L10_STD': 'L10_STD_DEV',
        f'SZN_TS_PCT': 'SZN_TS_PCT',
        f'SZN_USG_PROXY': 'SZN_USG_PROXY'
    }
    
    actual_rename = {k: v for k, v in mapping.items() if k in df.columns}
    if actual_rename:
        df = df.rename(columns=actual_rename)
        
    return df

def get_feature_cols(prop_cat, all_columns):
    """
    Determines which columns to use for training based on definitions.
    Includes new Interaction Features.
    """
    # 1. Start with Base Features from definitions
    relevant = feat_defs.BASE_FEATURE_COLS.copy()
    
    # 2. Add Vacancy & Interaction Columns
    vacancy_cols = [
        'TEAM_MISSING_USG', 'TEAM_MISSING_MIN', 
        'MISSING_USG_G', 'MISSING_USG_F',
        'INT_GUARD_VACANCY', 'INT_FORWARD_VACANCY' # New
    ]
    for vc in vacancy_cols:
        if vc in all_columns and vc not in relevant:
            relevant.append(vc)

    # 3. Add Rank/Team Columns dynamically found in the CSV
    keywords = feat_defs.RELEVANT_KEYWORDS.get(prop_cat, [])
    
    rank_cols = [
        c for c in all_columns 
        if ('_RANK' in c or 'TEAM_' in c or 'OPP_' in c or 'DVP_' in c)
        and c not in relevant
        and 'NAME' not in c and 'ABBREV' not in c and Cols.DATE not in c
        and 'SEASON_ID' not in c and Cols.PLAYER_ID not in c
        and c not in vacancy_cols 
    ]

    if keywords:
        filtered_ranks = [
            c for c in rank_cols 
            if any(k in c for k in keywords) 
            or 'PACE' in c or 'EFF' in c or 'DVP_' in c
        ]
        relevant.extend(filtered_ranks)
    else:
        relevant.extend(rank_cols)
    
    # 4. Filter VS_OPP and HIST features
    allowed_suffixes = feat_defs.PROP_FEATURE_MAP.get(prop_cat, [])
    final_features = set(relevant)
    
    always_keep = ['VS_OPP_GAMES_PLAYED', 'VS_OPP_MIN']
    for f in feat_defs.VS_OPP_FEATURES:
        if f in always_keep: continue
        is_valid = any(f == f"VS_OPP_{s}" for s in allowed_suffixes)
        if not is_valid and f in final_features:
            final_features.remove(f)
            
    for f in feat_defs.HIST_FEATURES:
        if f == 'HIST_VS_OPP_GAMES': continue
        is_valid = any(f.startswith(f"HIST_VS_OPP_{s}_") for s in allowed_suffixes)
        if not is_valid and f in final_features:
            final_features.remove(f)
            
    return [c for c in list(final_features) if c in all_columns]

def backfill_missing_cols(df, cols):
    """Ensures all feature columns exist, setting to NaN for Imputer."""
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan 
    return df

def train_single_prop(df, prop_cat):
    """Trains models for a specific prop category."""
    logging.info(f"Training {prop_cat}...")
    
    # --- TIME SERIES SPLIT PROTECTION ---
    date_col = None
    if Cols.DATE in df.columns: date_col = Cols.DATE
    elif 'GAME_DATE' in df.columns: date_col = 'GAME_DATE'
    
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=date_col, ascending=True).reset_index(drop=True)
    else:
        logging.warning(f"[{prop_cat}] Date column missing. Shuffling might leak future data!")

    # --- FEATURE ENGINEERING (ON THE FLY) ---
    df = add_interaction_features(df)

    # --- SYNTHETIC LINE GENERATION (Refined with Noise) ---
    prop_prefix = PROP_KEY_MAP.get(prop_cat, prop_cat)
    szn_col = f'{prop_prefix}_{Cols.SZN_AVG}'
    l5_col = f'{prop_prefix}_{Cols.L5_AVG}'
    
    if Cols.PROP_LINE not in df.columns:
        if szn_col in df.columns and l5_col in df.columns:
            # Baseline: Weighted Average favoring recent form
            base_line = (0.4 * df[szn_col]) + (0.6 * df[l5_col])
            # Add Noise: +/- 5% to simulate market variance / shading
            # This prevents model from overfitting to an exact mathematical average
            noise = np.random.uniform(0.95, 1.05, size=len(df))
            df[Cols.PROP_LINE] = base_line * noise
        elif szn_col in df.columns:
            df[Cols.PROP_LINE] = df[szn_col]
        else:
            df[Cols.PROP_LINE] = df[prop_cat].rolling(window=5, min_periods=1).mean().shift(1)
            
        df = df.dropna(subset=[Cols.PROP_LINE]).copy()

    # --- RENAME COLUMNS ---
    df = rename_features_for_model(df, prop_cat)

    # --- SAMPLE WEIGHT CALCULATION (Time Decay) ---
    sample_weights = calculate_time_decay_weights(df, date_col)

    # Select and Prepare Features
    feature_list = get_feature_cols(prop_cat, df.columns)
    
    if len(feature_list) < 5:
        logging.warning(f"[{prop_cat}] Not enough matching features found ({len(feature_list)}). Skipping.")
        return

    df = backfill_missing_cols(df, feature_list)
    
    # Sanitize column names
    sanitized_cols = [re.sub(r'[^\w\s]', '_', str(col)).replace(' ', '_') for col in feature_list]
    
    # Prepare X (Features)
    X = df[feature_list].copy()
    X.columns = sanitized_cols
    
    # Prepare Y (Targets)
    target_col = 'Actual Value' 
    if target_col not in df.columns: df[target_col] = df[prop_cat] 
    y_reg = df[target_col]
    
    # --- PUSH HANDLING ---
    no_push_mask = df[target_col] != df[Cols.PROP_LINE]
    
    # Time-Series Split Index
    split_idx = int(len(X) * (1 - TEST_SET_SIZE_PCT))
    
    # REGRESSION SPLIT (All Data)
    X_train_reg, X_val_reg = X.iloc[:split_idx], X.iloc[split_idx:]
    y_reg_train, y_reg_val = y_reg.iloc[:split_idx], y_reg.iloc[split_idx:]
    w_train_reg = sample_weights.iloc[:split_idx]
    
    # CLASSIFICATION SPLIT (No Pushes)
    X_clf_full = X[no_push_mask]
    y_clf_full = (df.loc[no_push_mask, target_col] > df.loc[no_push_mask, Cols.PROP_LINE]).astype(int)
    w_clf_full = sample_weights[no_push_mask]
    
    split_idx_clf = int(len(X_clf_full) * (1 - TEST_SET_SIZE_PCT))
    
    X_train_clf, X_val_clf = X_clf_full.iloc[:split_idx_clf], X_clf_full.iloc[split_idx_clf:]
    y_clf_train, y_clf_val = y_clf_full.iloc[:split_idx_clf], y_clf_full.iloc[split_idx_clf:]
    w_train_clf = w_clf_full.iloc[:split_idx_clf]

    # Pipeline Setup
    zero_impute_keywords = ['HIST_', 'VS_OPP_', 'Q1_', 'Q2_', 'Q3_', 'Q4_', 'DVP_', 'MISSING', 'INT_']
    hist_cols = [c for c in X.columns if any(k in c for k in zero_impute_keywords)]
    base_cols = [c for c in X.columns if c not in hist_cols]
    
    preprocessor = ColumnTransformer([
        ('zero_fill', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0, keep_empty_features=True)), 
            ('scaler', StandardScaler())
        ]), hist_cols),
        ('median_fill', Pipeline([
            ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)), 
            ('scaler', StandardScaler())
        ]), base_cols)
    ], remainder='passthrough')
    
    try:
        X_train_proc_reg = preprocessor.fit_transform(X_train_reg)
        X_val_proc_reg = preprocessor.transform(X_val_reg)
        X_train_proc_clf = preprocessor.transform(X_train_clf)
        X_val_proc_clf = preprocessor.transform(X_val_clf)
    except Exception as e:
        logging.error(f"Preprocessing failed for {prop_cat}: {e}")
        return

    # --- MODEL 1: QUANTILE REGRESSION ---
    def train_q(alpha):
        lgbm = lgb.LGBMRegressor(objective='quantile', alpha=alpha, n_estimators=600, learning_rate=0.04, verbose=-1)
        lgbm.fit(
            X_train_proc_reg, y_reg_train, sample_weight=w_train_reg,
            eval_set=[(X_val_proc_reg, y_reg_val)], 
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        xgb_mod = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha, n_estimators=600, learning_rate=0.04)
        xgb_mod.fit(X_train_proc_reg, y_reg_train, sample_weight=w_train_reg, eval_set=[(X_val_proc_reg, y_reg_val)], verbose=False)
        return lgbm, xgb_mod

    lgbm_q20, xgb_q20 = train_q(0.20)
    lgbm_q80, xgb_q80 = train_q(0.80)
    
    # --- MODEL 2: CLASSIFIER ---
    # Poisson switch for Counting Stats
    is_counting_stat = prop_cat in ['STL', 'BLK', 'FG3M', 'TOV', 'DD', 'TD']
    
    # We stick to binary:logistic for the Probability of winning the bet (Over/Under)
    # Poisson is usually for the *Regressor* (predicting count), but since we use Quantile Regression
    # for the range, we will stick to Logistic for the Win/Loss probability.
    # However, we can use 'count:poisson' for an auxiliary Regressor if needed, but for now 
    # optimizing the Classifier with better weights is the priority.
    
    clf = xgb.XGBClassifier(
        objective='binary:logistic', 
        n_estimators=600, 
        learning_rate=0.03, 
        eval_metric='logloss',
        max_depth=4 # Slightly shallower trees to prevent overfitting on noise
    )
    
    clf.fit(
        X_train_proc_clf, y_clf_train, 
        sample_weight=w_train_clf, 
        eval_set=[(X_val_proc_clf, y_clf_val)], 
        verbose=False
    )
    
    preds = clf.predict_proba(X_val_proc_clf)[:, 1]
    acc = accuracy_score(y_clf_val, (preds > 0.5).astype(int))
    logging.info(f"[{prop_cat}] Validation Accuracy (Push-Free): {acc:.1%}")

    artifacts = {
        'scaler': preprocessor,
        'features': sanitized_cols,
        'q20': {'lgbm': lgbm_q20, 'xgb': xgb_q20},
        'q80': {'lgbm': lgbm_q80, 'xgb': xgb_q80},
        'clf': clf
    }
    registry.save_artifacts(prop_cat, artifacts)

def main():
    common.setup_logging(name="train_models")
    logging.info(">>> STARTING MODEL TRAINING PIPELINE")

    # 1. Load Training Data (FROM PARQUET)
    train_file = cfg.MASTER_TRAINING_FILE
    if not train_file.exists():
        logging.critical(f"Training dataset not found at {train_file}")
        logging.critical("Please run 'scripts/run_build_db.py' first.")
        return

    try:
        logging.info(f"Loading dataset: {train_file}")
        df = pd.read_parquet(train_file)
        
        if df.empty:
            logging.critical("Training dataset is empty.")
            return
            
        logging.info(f"Loaded {len(df)} rows of training data.")
        
    except Exception as e:
        logging.critical(f"Failed to load training data: {e}")
        return

    # 2. Filter Props based on Dataset Availability
    available_cols = set(df.columns)
    
    props_to_train = [p for p in cfg.SUPPORTED_PROPS if p in available_cols]
    skipped_props = [p for p in cfg.SUPPORTED_PROPS if p not in available_cols]

    if skipped_props:
        logging.info(f"Note: {len(skipped_props)} props excluded (Data not in dataset).")

    logging.info(f"Proceeding to train models for {len(props_to_train)} props...")

    # 3. Train Models
    successful = 0
    failed = 0
    
    for prop in props_to_train:
        logging.info(f"--- Training Model: {prop} ---")
        
        # Create specific dataframe for this prop
        prop_df = df.dropna(subset=[prop]).copy()
        prop_df['Actual Value'] = prop_df[prop]
        
        if prop_df.empty:
            logging.warning(f"Skipping {prop}: No valid rows after preprocessing.")
            failed += 1
            continue

        try:
            train_single_prop(prop_df, prop)
            successful += 1
        except Exception as e:
            logging.error(f"Failed to train {prop}: {e}", exc_info=True)
            failed += 1

    logging.info(f"<<< TRAINING COMPLETE. Success: {successful}, Failed: {failed}")

if __name__ == "__main__":
    main()