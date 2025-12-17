import pandas as pd
import numpy as np
import re
import warnings
import logging
import scipy.stats as stats
from prop_analyzer import config as cfg
from prop_analyzer.models import registry

# --- MAPPING: Prop Text Names -> Internal Model Keys ---
PROP_MAP = {
    'Points': 'PTS', 'Player Points': 'PTS',
    'Rebounds': 'REB', 'Player Rebounds': 'REB',
    'Assists': 'AST', 'Player Assists': 'AST',
    'Threes': 'FG3M', '3-Pointers': 'FG3M', 'Player Threes': 'FG3M',
    'Steals': 'STL',
    'Blocks': 'BLK',
    'Turnovers': 'TOV',
    'PRA': 'PRA', 'Pts+Reb+Ast': 'PRA',
    'Pts+Reb': 'PR',
    'Pts+Ast': 'PA',
    'Reb+Ast': 'RA',
    'Fantasy Points': 'FANTASY_PTS',
    
    # Explicit Quarter/Half Mappings
    'Q1_PTS': 'Q1_PTS', 'Q1 Points': 'Q1_PTS',
    'Q1_REB': 'Q1_REB', 'Q1 Rebounds': 'Q1_REB',
    'Q1_AST': 'Q1_AST', 'Q1 Assists': 'Q1_AST',
    'Q1_PRA': 'Q1_PRA', 'Q1 Pts+Reb+Ast': 'Q1_PRA',
    'Q1_PR':  'Q1_PR',  'Q1 Pts+Reb': 'Q1_PR',
    'Q1_PA':  'Q1_PA',  'Q1 Pts+Ast': 'Q1_PA',
    'Q1_RA':  'Q1_RA',  'Q1 Reb+Ast': 'Q1_RA',
    
    '1H_PTS': '1H_PTS', '1H Points': '1H_PTS',
    '1H_REB': '1H_REB', '1H Rebounds': '1H_REB',
    '1H_AST': '1H_AST', '1H Assists': '1H_AST',
    '1H_PRA': '1H_PRA', '1H Pts+Reb+Ast': '1H_PRA',
    '1H_PR':  '1H_PR',  '1H Pts+Reb': '1H_PR',
    '1H_PA':  '1H_PA',  '1H Pts+Ast': '1H_PA',
    '1H_RA':  '1H_RA',  '1H Reb+Ast': '1H_RA'
}

def rename_features_for_inference(feature_dict, prop_cat):
    """
    Renames keys in the feature dictionary to match model expectations.
    """
    prefix = PROP_MAP.get(prop_cat, prop_cat)
    
    mapping = {
        f'{prefix}_SZN_AVG': 'SZN Avg',
        f'{prefix}_L5_AVG': 'L5 Avg',
        f'{prefix}_L5_EWMA': 'L5 EWMA',
        f'{prefix}_L3_AVG': 'L3 Avg',
        f'{prefix}_L10_STD': 'L10_STD_DEV',
        f'{prefix}_L10_STD_DEV': 'L10_STD_DEV'
    }
    
    new_dict = feature_dict.copy()
    for old_key, new_key in mapping.items():
        if old_key in new_dict:
            new_dict[new_key] = new_dict[old_key]
            
    return new_dict

def predict_props(features_df):
    results = []
    model_cache = {}
    
    logging.info(f"Starting batch inference on {len(features_df)} props...")

    for idx, row in features_df.iterrows():
        raw_type = row.get('Prop Category')
        
        if pd.isna(raw_type) or not isinstance(raw_type, str):
            continue 
            
        model_key = PROP_MAP.get(raw_type, raw_type)
        
        if model_key not in model_cache:
            try:
                loaded_artifact = registry.load_artifacts(model_key)
                model_cache[model_key] = loaded_artifact
            except Exception as e:
                # Silence model loading errors
                model_cache[model_key] = None
        
        # --- SAFETY CHECK ---
        szn_avg_key = f"{model_key}_SZN_AVG"
        szn_val = row.get(szn_avg_key)
        
        if pd.isna(szn_val) or (szn_val == 0.0 and model_key in ['PTS', 'PRA', 'PA', 'PR']):
            logging.warning(f"Skipping {row.get('Player Name')} ({raw_type}) - Missing History (SZN Avg is 0/NaN)")
            continue
            
        feature_vector = row.to_dict()
        feature_vector = rename_features_for_inference(feature_vector, raw_type)
        
        pred_out = predict_prop(model_cache, model_key, feature_vector)
        
        if pred_out:
            line = row.get('Prop Line', 0.0)
            injury_status = row.get('Status_Clean', 'ACTIVE')
            
            # Use the new probability logic
            analysis = determine_tier(
                line, 
                pred_out['q20'], 
                pred_out['q80'], 
                pred_out.get('prob_over_clf', 0.5),
                injury_status=injury_status
            )
            
            res = row.to_dict()
            res.update({
                'Model_Pred': round(analysis['Median_Proj'], 2),
                'Model_Conf': round(analysis['Win_Prob'], 3),
                'Win_Prob': round(analysis['Win_Prob'], 3), # <--- ADDED: Explicit key for analysis script
                'Edge_Type': analysis['Best Pick'],
                'Tier': analysis['Tier'],
                'Score': analysis['Score'],
                'Diff%': round((analysis['Edge'] / line) * 100, 1) if line > 0 else 0.0,
                'Spread_Pct': analysis['Spread_Pct'], # New Metric for Volatility filtering
                'Is_Divergent': analysis['Is_Divergent']
            })
            results.append(res)
            
    if not results:
        return pd.DataFrame()
        
    return pd.DataFrame(results)

def predict_prop(model_cache, prop_category, feature_vector_dict):
    models = model_cache.get(prop_category)
    if models is None:
        return None

    num_df = pd.DataFrame([feature_vector_dict])
    num_df.columns = [re.sub(r'[^\w\s]', '_', str(col)).replace(' ', '_') for col in num_df.columns]
    feature_cols = models['features']
    
    # Use np.nan for missing columns so Imputer handles them
    aligned_vector = num_df.reindex(columns=feature_cols, fill_value=np.nan)

    preprocessor = models['scaler']
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.filterwarnings("ignore", category=UserWarning, message=".*X does not have valid feature names.*")
            
            X_scaled = preprocessor.transform(aligned_vector)

            q20_lgbm = models['q20']['lgbm'].predict(X_scaled)[0]
            q20_xgb = models['q20']['xgb'].predict(X_scaled)[0]
            pred_lower = (q20_lgbm + q20_xgb) / 2

            q80_lgbm = models['q80']['lgbm'].predict(X_scaled)[0]
            q80_xgb = models['q80']['xgb'].predict(X_scaled)[0]
            pred_upper = (q80_lgbm + q80_xgb) / 2
            
            # Classifier is preserved but used differently now
            prob_over_clf = 0.5
            if models.get('clf'):
                try:
                    prob_over_clf = models['clf'].predict_proba(X_scaled)[0][1]
                except:
                    prob_over_clf = 0.5

        return {
            'q20': pred_lower,
            'q80': pred_upper,
            'prob_over_clf': prob_over_clf
        }
    except Exception as e:
        logging.error(f"Inference error for {prop_category}: {e}")
        return None

def determine_tier(prop_line, pred_lower, pred_upper, prob_over_clf, injury_status='ACTIVE'):
    """
    Calculates Tier based on the Implied Probability of the Vegas Line within
    the projected distribution (Z-Score method).
    """
    median_proj = (pred_lower + pred_upper) / 2
    regression_edge = median_proj - prop_line
    
    # --- DISTRIBUTION STATISTICS ---
    # q80 - q20 represents the middle 60% of the distribution.
    # In a normal distribution, the middle 60% covers approx 1.68 standard deviations (-0.84 to +0.84).
    # We estimate the standard deviation based on this spread to gauge volatility.
    spread = pred_upper - pred_lower
    if spread <= 0.01: spread = 0.01 # Prevent divide by zero
    
    implied_std_dev = spread / 1.68
    
    # Spread Pct: How wide is the range relative to the line?
    # High % means the model is unsure or the player is volatile.
    spread_pct = spread / prop_line if prop_line > 0 else 0.0
    
    # --- Z-SCORE CALCULATION ---
    # Z = (Target - Mean) / StdDev
    # We calculate the Z-Score of the VEGAS LINE within our projected distribution.
    z_score = (prop_line - median_proj) / implied_std_dev
    
    # Calculate Probability of going OVER the Vegas Line
    # Survival Function (1 - CDF) gives prob of value > x
    prob_over_implied = stats.norm.sf(z_score)
    prob_under_implied = 1.0 - prob_over_implied
    
    # --- SELECTION LOGIC ---
    if prob_over_implied >= 0.50:
        best_pick = 'Over'
        win_prob = prob_over_implied
    else:
        best_pick = 'Under'
        win_prob = prob_under_implied

    # --- TIERING SYSTEM ---
    # Use Config thresholds if available, else defaults
    S_TIER_PROB = getattr(cfg, 'MIN_PROB_FOR_S_TIER', 0.59)
    S_TIER_EDGE = getattr(cfg, 'MIN_EDGE_FOR_S_TIER', 1.5)
    A_TIER_PROB = 0.555
    A_TIER_EDGE = 1.0
    
    # Divergence check (Classifier vs Regression)
    # If Classifier strongly disagrees with Implied Prob, we downgrade.
    # e.g., Regression says 60% Over, Classifier says 40% Over -> Caution.
    clf_pick = 'Over' if prob_over_clf >= 0.5 else 'Under'
    is_divergent = (best_pick != clf_pick) and (abs(prob_over_clf - 0.5) > 0.05)

    score = abs(regression_edge)
    tier = 'C Tier' 
    
    if is_divergent:
        # If divergent, cap at B Tier maximum
        tier = 'C Tier'
        if win_prob >= A_TIER_PROB:
            tier = 'B Tier'
    else:
        if win_prob >= S_TIER_PROB and score >= S_TIER_EDGE:
            tier = 'S Tier'
        elif win_prob >= S_TIER_PROB and score >= A_TIER_EDGE:
            tier = 'A Tier'
        elif win_prob >= A_TIER_PROB and score >= S_TIER_EDGE:
            tier = 'A Tier'
        elif win_prob >= A_TIER_PROB:
            tier = 'B Tier'
        elif score >= S_TIER_EDGE:
            tier = 'B Tier'
        else:
            tier = 'C Tier'

    # Injury adjustments
    if injury_status == 'GTD' and tier in ['S Tier', 'A Tier']:
        tier = 'B Tier'
    elif injury_status in ['OUT', 'DOUBTFUL']:
        tier = 'Void'

    return {
        'Best Pick': best_pick,
        'Tier': tier,
        'Score': round(score, 2),
        'Edge': round(regression_edge, 2),
        'Win_Prob': win_prob,
        'Median_Proj': median_proj, # Fixed variable name (was median_pred in earlier broken version)
        'Spread_Pct': round(spread_pct, 3),
        'Is_Divergent': is_divergent
    }