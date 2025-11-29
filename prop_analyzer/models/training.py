import pandas as pd
import numpy as np
import logging
import xgboost as xgb
import lightgbm as lgb
import re
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

from prop_analyzer import config as cfg
from prop_analyzer.features import definitions as feat_defs
from prop_analyzer.models import registry

# Constants
TEST_SET_SIZE_PCT = 0.20
MIN_SAMPLES = 200

# Map Prop Categories to Data Column Prefixes
PROP_KEY_MAP = {
    'Points': 'PTS', 'Rebounds': 'REB', 'Assists': 'AST',
    'Threes': 'FG3M', 'Steals': 'STL', 'Blocks': 'BLK', 'Turnovers': 'TOV',
    'PRA': 'PRA', 'Pts+Reb+Ast': 'PRA', 
    'Pts+Reb': 'PR', 'Pts+Ast': 'PA', 'Reb+Ast': 'RA',
    'Fantasy Points': 'FANTASY_PTS',
    # Direct mappings
    'PTS': 'PTS', 'REB': 'REB', 'AST': 'AST', 'FG3M': 'FG3M',
    'STL': 'STL', 'BLK': 'BLK', 'TOV': 'TOV', 'STK': 'STK'
}

def rename_features_for_model(df, prop_cat):
    """
    Maps specific column names (e.g., PTS_SZN_AVG) to generic definition names (e.g., SZN Avg).
    """
    prefix = PROP_KEY_MAP.get(prop_cat, prop_cat)
    
    # Define mappings based on dataset.py output
    mapping = {
        f'{prefix}_SZN_AVG': 'SZN Avg',
        f'{prefix}_L5_AVG': 'L5 Avg',  
        f'{prefix}_L5_EWMA': 'L5 EWMA',
        f'{prefix}_L3_AVG': 'L3 Avg',
        f'{prefix}_L10_STD': 'L10_STD_DEV',
        f'{prefix}_L10_STD_DEV': 'L10_STD_DEV',
        f'SZN_TS_PCT': 'SZN_TS_PCT',
        f'SZN_EFG_PCT': 'SZN_EFG_PCT',
        f'SZN_USG_PROXY': 'SZN_USG_PROXY'
        # REMOVED DANGEROUS FALLBACK to prevent duplicate columns and data leakage
    }
    
    # Only rename columns that actually exist in the DF
    actual_rename = {k: v for k, v in mapping.items() if k in df.columns}
    
    if actual_rename:
        df = df.rename(columns=actual_rename)
        
    return df

def get_feature_cols(prop_cat, all_columns):
    """
    Determines which columns to use for training based on definitions.
    """
    # 1. Start with Base Features from definitions
    relevant = feat_defs.BASE_FEATURE_COLS.copy()
    
    # 2. Explicitly add Vacancy Columns
    vacancy_cols = ['TEAM_MISSING_USG', 'TEAM_MISSING_MIN', 'MISSING_USG_G', 'MISSING_USG_F']
    for vc in vacancy_cols:
        if vc in all_columns and vc not in relevant:
            relevant.append(vc)

    # 3. Add Rank/Team Columns dynamically found in the CSV
    keywords = feat_defs.RELEVANT_KEYWORDS.get(prop_cat, [])
    
    rank_cols = [
        c for c in all_columns 
        if ('_RANK' in c or 'TEAM_' in c or 'OPP_' in c or 'DVP_' in c)
        and c not in relevant
        and 'NAME' not in c and 'ABBREV' not in c and 'DATE' not in c
        and 'SEASON_ID' not in c and 'PLAYER_ID' not in c # FIX: Exclude IDs explicitly
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
        # Fallback if prop not in map, take all context
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
            
    # Return intersection with actual available columns to avoid KeyErrors
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
    if 'GAME_DATE' in df.columns:
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df = df.sort_values(by='GAME_DATE', ascending=True).reset_index(drop=True)
    else:
        logging.warning(f"[{prop_cat}] 'GAME_DATE' missing. Shuffling might leak future data!")

    # --- RENAME COLUMNS ---
    df = rename_features_for_model(df, prop_cat)

    # --- SAMPLE WEIGHT CALCULATION ---
    if 'SEASON_ID' in df.columns:
        latest_season = df['SEASON_ID'].max()
        sample_weights = df['SEASON_ID'].apply(lambda x: 1.0 if x == latest_season else 0.6)
    else:
        sample_weights = pd.Series(1.0, index=df.index)

    # Select and Prepare Features
    feature_list = get_feature_cols(prop_cat, df.columns)
    
    # Check if we have enough features
    if len(feature_list) < 5:
        logging.warning(f"[{prop_cat}] Not enough matching features found ({len(feature_list)}). Skipping.")
        return

    df = backfill_missing_cols(df, feature_list)
    
    # Sanitize column names
    sanitized_cols = [re.sub(r'[^\w\s]', '_', str(col)).replace(' ', '_') for col in feature_list]
    
    X = df[feature_list].copy()
    X.columns = sanitized_cols
    
    # Targets
    y_reg = df['Actual Value']
    y_clf = (df['Actual Value'] > df['Prop Line']).astype(int)
    
    # Time-Series Split
    split_idx = int(len(X) * (1 - TEST_SET_SIZE_PCT))
    
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_reg_train, y_reg_val = y_reg.iloc[:split_idx], y_reg.iloc[split_idx:]
    y_clf_train, y_clf_val = y_clf.iloc[:split_idx], y_clf.iloc[split_idx:]
    
    w_train = sample_weights.iloc[:split_idx]
    
    # Pipeline Setup
    zero_impute_keywords = ['HIST_', 'VS_OPP_', 'Q1_', 'Q2_', 'Q3_', 'Q4_', 'DVP_', 'MISSING']
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
        X_train_proc = preprocessor.fit_transform(X_train)
        X_val_proc = preprocessor.transform(X_val)
    except Exception as e:
        logging.error(f"Preprocessing failed for {prop_cat}: {e}")
        return

    # --- MODEL 1: QUANTILE REGRESSION ---
    def train_q(alpha):
        lgbm = lgb.LGBMRegressor(objective='quantile', alpha=alpha, n_estimators=600, learning_rate=0.04, verbose=-1)
        lgbm.fit(
            X_train_proc, y_reg_train, sample_weight=w_train,
            eval_set=[(X_val_proc, y_reg_val)], 
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        xgb_mod = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha, n_estimators=600, learning_rate=0.04)
        xgb_mod.fit(X_train_proc, y_reg_train, sample_weight=w_train, eval_set=[(X_val_proc, y_reg_val)], verbose=False)
        return lgbm, xgb_mod

    lgbm_q20, xgb_q20 = train_q(0.20)
    lgbm_q80, xgb_q80 = train_q(0.80)
    
    # --- MODEL 2: CLASSIFIER ---
    clf = xgb.XGBClassifier(objective='binary:logistic', n_estimators=500, learning_rate=0.03, eval_metric='logloss')
    clf.fit(X_train_proc, y_clf_train, sample_weight=w_train, eval_set=[(X_val_proc, y_clf_val)], verbose=False)
    
    preds = clf.predict_proba(X_val_proc)[:, 1]
    acc = accuracy_score(y_clf_val, (preds > 0.5).astype(int))
    logging.info(f"[{prop_cat}] Validation Accuracy: {acc:.1%}")

    artifacts = {
        'scaler': preprocessor,
        'features': sanitized_cols,
        'q20': {'lgbm': lgbm_q20, 'xgb': xgb_q20},
        'q80': {'lgbm': lgbm_q80, 'xgb': xgb_q80},
        'clf': clf
    }
    registry.save_artifacts(prop_cat, artifacts)