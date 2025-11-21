import pandas as pd
import numpy as np
import re
import warnings
from prop_analyzer import config as cfg

def predict_prop(model_cache, prop_category, feature_vector_dict):
    """
    Runs inference for a single prop.
    Returns: (projection_range, win_prob, metrics)
    """
    models = model_cache.get(prop_category)
    if models is None:
        return None

    # 1. Align Features
    # Convert dict to DataFrame
    num_df = pd.DataFrame([feature_vector_dict])
    
    # Sanitize column names to match training (e.g. replace spaces with _)
    num_df.columns = [re.sub(r'[^\w\s]', '_', str(col)).replace(' ', '_') for col in num_df.columns]
    
    # Reindex to ensure exact column match with training features
    feature_cols = models['features']
    aligned_vector = num_df.reindex(columns=feature_cols, fill_value=0.0)

    # 2. Scale
    preprocessor = models['scaler']
    
    # Suppress warnings during Transformation AND Prediction
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        X_scaled = preprocessor.transform(aligned_vector)

        # 3. Predict Quantiles (Edge)
        q20_lgbm = models['q20']['lgbm'].predict(X_scaled)[0]
        q20_xgb = models['q20']['xgb'].predict(X_scaled)[0]
        pred_lower = (q20_lgbm + q20_xgb) / 2

        q80_lgbm = models['q80']['lgbm'].predict(X_scaled)[0]
        q80_xgb = models['q80']['xgb'].predict(X_scaled)[0]
        pred_upper = (q80_lgbm + q80_xgb) / 2
        
        # 4. Predict Probability (Classifier)
        prob_over = 0.5
        if models['clf']:
            # [:, 1] is probability of class 1 (Over)
            prob_over = models['clf'].predict_proba(X_scaled)[0][1]

    return {
        'q20': pred_lower,
        'q80': pred_upper,
        'prob_over': prob_over
    }

def determine_tier(prop_line, pred_lower, pred_upper, prob_over, injury_status='ACTIVE'):
    """
    Calculates the Edge, Score, and Tier based on predictions.
    """
    median_pred = (pred_lower + pred_upper) / 2
    edge = median_pred - prop_line
    
    if edge > 0:
        best_pick = 'Over'
        win_prob = prob_over
    else:
        best_pick = 'Under'
        win_prob = 1.0 - prob_over

    score = abs(edge)
    
    # Tiering Logic
    is_high_confidence_prob = (win_prob >= cfg.MIN_PROB_FOR_S_TIER)
    
    if score >= cfg.MIN_EDGE_FOR_S_TIER:
        tier = 'S Tier' if is_high_confidence_prob else 'A Tier'
    elif score >= (prop_line * cfg.MIN_EDGE_FOR_A_TIER):
        tier = 'A Tier' if is_high_confidence_prob else 'B Tier'
    elif score > 0:
        tier = 'C Tier'
    else:
        tier = 'C Tier' # Negative edge case, though score is abs()
        
    # Low probability override
    if win_prob < 0.45:
         tier = 'C Tier'
         
    # Downgrade GTD (Game Time Decision)
    if injury_status == 'GTD' and tier == 'S Tier':
        tier = 'A Tier'

    return {
        'Best Pick': best_pick,
        'Tier': tier,
        'Score': round(score, 2),
        'Edge': round(edge, 2),
        'Win_Prob': win_prob,
        'Median_Proj': median_pred
    }