import pandas as pd
import numpy as np
import re
import warnings
import logging
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
    'Fantasy Points': 'FANTASY_PTS'
}

def predict_props(features_df):
    results = []
    model_cache = {}
    
    logging.info(f"Starting batch inference on {len(features_df)} props...")

    for idx, row in features_df.iterrows():
        # FIX: Use 'Prop Category' to match parser and run_analysis.py
        raw_type = row.get('Prop Category')
        
        if pd.isna(raw_type) or not isinstance(raw_type, str):
            continue 
            
        model_key = PROP_MAP.get(raw_type, raw_type)
        
        if model_key not in model_cache:
            try:
                loaded_artifact = registry.load_artifacts(model_key)
                model_cache[model_key] = loaded_artifact
            except Exception as e:
                logging.warning(f"Could not load model for {model_key}: {e}")
                model_cache[model_key] = None
            
        feature_vector = row.to_dict()
        pred_out = predict_prop(model_cache, model_key, feature_vector)
        
        if pred_out:
            line = row.get('Prop Line', 0.0)
            injury_status = row.get('Status_Clean', 'ACTIVE')
            
            analysis = determine_tier(
                line, 
                pred_out['q20'], 
                pred_out['q80'], 
                pred_out['prob_over'], 
                injury_status=injury_status
            )
            
            res = row.to_dict()
            res.update({
                'Model_Pred': round(analysis['Median_Proj'], 2),
                'Model_Conf': round(analysis['Win_Prob'], 3),
                'Edge_Type': analysis['Best Pick'],
                'Tier': analysis['Tier'],
                'Score': analysis['Score'],
                'Diff%': round((analysis['Edge'] / line) * 100, 1) if line > 0 else 0.0,
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

    # Align Features
    num_df = pd.DataFrame([feature_vector_dict])
    num_df.columns = [re.sub(r'[^\w\s]', '_', str(col)).replace(' ', '_') for col in num_df.columns]
    feature_cols = models['features']
    aligned_vector = num_df.reindex(columns=feature_cols, fill_value=0.0)

    preprocessor = models['scaler']
    
    try:
        # Suppress warnings during transform AND prediction
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Specifically ignore the sklearn/LGBM feature name warning
            warnings.filterwarnings("ignore", category=UserWarning, message=".*X does not have valid feature names.*")
            
            X_scaled = preprocessor.transform(aligned_vector)

            # Predict Quantiles
            q20_lgbm = models['q20']['lgbm'].predict(X_scaled)[0]
            q20_xgb = models['q20']['xgb'].predict(X_scaled)[0]
            pred_lower = (q20_lgbm + q20_xgb) / 2

            q80_lgbm = models['q80']['lgbm'].predict(X_scaled)[0]
            q80_xgb = models['q80']['xgb'].predict(X_scaled)[0]
            pred_upper = (q80_lgbm + q80_xgb) / 2
            
            # Predict Probability
            prob_over = 0.5
            if models['clf']:
                prob_over = models['clf'].predict_proba(X_scaled)[0][1]

        return {
            'q20': pred_lower,
            'q80': pred_upper,
            'prob_over': prob_over
        }
    except Exception as e:
        logging.error(f"[{prop_category}] Prediction Failed: {e}")
        return None

def determine_tier(prop_line, pred_lower, pred_upper, prob_over, injury_status='ACTIVE'):
    median_pred = (pred_lower + pred_upper) / 2
    regression_edge = median_pred - prop_line
    
    if prob_over >= 0.50:
        model_pick = 'Over'
        win_prob = prob_over
    else:
        model_pick = 'Under'
        win_prob = 1.0 - prob_over

    if regression_edge > 0:
        reg_pick = 'Over'
    else:
        reg_pick = 'Under'

    is_divergent = (model_pick != reg_pick)
    score = abs(regression_edge)
    tier = 'C Tier' 
    
    S_TIER_PROB = getattr(cfg, 'MIN_PROB_FOR_S_TIER', 0.58)
    S_TIER_EDGE = getattr(cfg, 'MIN_EDGE_FOR_S_TIER', 1.5)
    A_TIER_PROB = 0.555
    A_TIER_EDGE = 1.0
    
    if is_divergent:
        tier = 'C Tier'
        best_pick = model_pick
    else:
        best_pick = model_pick
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
        'Median_Proj': median_pred,
        'Is_Divergent': is_divergent
    }