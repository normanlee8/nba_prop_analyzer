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

def get_feature_cols(prop_cat, all_columns):
    """
    Determines which columns to use for training based on definitions.
    """
    # 1. Start with Base Features from definitions
    relevant = feat_defs.BASE_FEATURE_COLS.copy()
    
    # 2. Explicitly add Vacancy Columns (New Phase Feature)
    # These might not be in BASE_FEATURE_COLS yet, so we force add them if present in CSV
    vacancy_cols = ['TEAM_MISSING_USG', 'TEAM_MISSING_MIN', 'MISSING_USG_G', 'MISSING_USG_F']
    for vc in vacancy_cols:
        if vc in all_columns and vc not in relevant:
            relevant.append(vc)

    # 3. Add Rank/Team Columns dynamically found in the CSV
    # Only include rank/team columns that are statistically relevant to this prop
    keywords = feat_defs.RELEVANT_KEYWORDS.get(prop_cat, [])
    
    # Columns to search for context (Team Stats, Opponent Stats, DVP)
    rank_cols = [
        c for c in all_columns 
        if ('_RANK' in c or 'TEAM_' in c or 'OPP_' in c or 'DVP_' in c)
        and c not in relevant
        and 'NAME' not in c and 'ABBREV' not in c and 'DATE' not in c
        and c not in vacancy_cols # Already handled
    ]

    # Filter rank columns based on keywords (e.g. only keep 'REB' ranks for Rebound props)
    # But always keep global context like Pace, Efficiency, and DVP
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
    
    # 4. Filter VS_OPP and HIST features based on the specific prop to reduce noise
    allowed_suffixes = feat_defs.PROP_FEATURE_MAP.get(prop_cat, [])
    final_features = set(relevant)
    
    # Filter VS_OPP
    always_keep = ['VS_OPP_GAMES_PLAYED', 'VS_OPP_MIN']
    for f in feat_defs.VS_OPP_FEATURES:
        if f in always_keep: continue
        is_valid = any(f == f"VS_OPP_{s}" for s in allowed_suffixes)
        if not is_valid and f in final_features:
            final_features.remove(f)
            
    # Filter HIST
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

    # Select and Prepare Features
    feature_list = get_feature_cols(prop_cat, df.columns)
    df = backfill_missing_cols(df, feature_list)
    
    # Sanitize column names (Regex) for XGBoost/LightGBM compatibility
    sanitized_cols = [re.sub(r'[^\w\s]', '_', str(col)).replace(' ', '_') for col in feature_list]
    
    X = df[feature_list].copy()
    X.columns = sanitized_cols
    
    # Targets
    y_reg = df['Actual Value']
    y_clf = (df['Actual Value'] > df['Prop Line']).astype(int)
    
    # Time-Series Split (Last 20% is validation)
    split_idx = int(len(X) * (1 - TEST_SET_SIZE_PCT))
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_reg_train, y_reg_val = y_reg.iloc[:split_idx], y_reg.iloc[split_idx:]
    y_clf_train, y_clf_val = y_clf.iloc[:split_idx], y_clf.iloc[split_idx:]
    
    # Pipeline Setup
    # Group 1: Sparse/History data -> Impute with 0 (vacancies, opponent history)
    zero_impute_keywords = ['HIST_', 'VS_OPP_', 'Q1_', 'Q2_', 'Q3_', 'Q4_', 'DVP_', 'MISSING']
    hist_cols = [c for c in X.columns if any(k in c for k in zero_impute_keywords)]
    
    # Group 2: Base/Season Stats -> Impute with Median (average player behavior)
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
    
    # Fit Preprocessor
    try:
        X_train_proc = preprocessor.fit_transform(X_train)
        X_val_proc = preprocessor.transform(X_val)
    except Exception as e:
        logging.error(f"Preprocessing failed for {prop_cat}: {e}")
        return

    # --- MODEL 1: QUANTILE REGRESSION (EDGE) ---
    def train_q(alpha):
        # LightGBM
        lgbm = lgb.LGBMRegressor(objective='quantile', alpha=alpha, n_estimators=600, learning_rate=0.04, verbose=-1)
        lgbm.fit(X_train_proc, y_reg_train, eval_set=[(X_val_proc, y_reg_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        
        # XGBoost
        xgb_mod = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha, n_estimators=600, learning_rate=0.04)
        xgb_mod.fit(X_train_proc, y_reg_train, eval_set=[(X_val_proc, y_reg_val)], verbose=False)
        return lgbm, xgb_mod

    lgbm_q20, xgb_q20 = train_q(0.20)
    lgbm_q80, xgb_q80 = train_q(0.80)
    
    # --- MODEL 2: CLASSIFIER (WIN PROB) ---
    clf = xgb.XGBClassifier(objective='binary:logistic', n_estimators=500, learning_rate=0.03, eval_metric='logloss')
    clf.fit(X_train_proc, y_clf_train, eval_set=[(X_val_proc, y_clf_val)], verbose=False)
    
    # Quick Eval
    preds = clf.predict_proba(X_val_proc)[:, 1]
    acc = accuracy_score(y_clf_val, (preds > 0.5).astype(int))
    logging.info(f"[{prop_cat}] Validation Accuracy: {acc:.1%}")

    # Save Artifacts
    artifacts = {
        'scaler': preprocessor,
        'features': sanitized_cols,
        'q20': {'lgbm': lgbm_q20, 'xgb': xgb_q20},
        'q80': {'lgbm': lgbm_q80, 'xgb': xgb_q80},
        'clf': clf
    }
    registry.save_artifacts(prop_cat, artifacts)