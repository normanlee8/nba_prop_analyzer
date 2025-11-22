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
        
        try:
            X_scaled = preprocessor.transform(aligned_vector)
        except Exception as e:
            # Fallback if scaling fails (e.g. feature mismatch)
            return None

        # 3. Predict Quantiles (Edge)
        q20_lgbm = models['q20']['lgbm'].predict(X_scaled)[0]
        q20_xgb = models['q20']['xgb'].predict(X_scaled)[0]
        pred_lower = (q20_lgbm + q20_xgb) / 2

        q80_lgbm = models['q80']['lgbm'].predict(X_scaled)[0]
        q80_xgb = models['q80']['xgb'].predict(X_scaled)[0]
        pred_upper = (q80_lgbm + q80_xgb) / 2
        
        # 4. Predict Probability (Classifier)
        # This is the most important metric for "Win %"
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
    Crucial Update: Now checks for Divergence between Classifier (Prob) and Regression (Edge).
    """
    median_pred = (pred_lower + pred_upper) / 2
    regression_edge = median_pred - prop_line
    
    # 1. Determine "Signal" from the Classifier
    if prob_over >= 0.50:
        model_pick = 'Over'
        win_prob = prob_over
    else:
        model_pick = 'Under'
        win_prob = 1.0 - prob_over

    # 2. Determine "Direction" from the Regression
    if regression_edge > 0:
        reg_pick = 'Over'
    else:
        reg_pick = 'Under'

    # 3. Divergence Check
    # If Classifier says Over (60%) but Regression says Under (Proj < Line), we have a problem.
    is_divergent = (model_pick != reg_pick)
    
    # 4. Scoring Logic
    # We use the Regression Edge magnitude for potential payout value, 
    # but the Classifier Probability for confidence.
    score = abs(regression_edge)
    
    # 5. Tiering Logic
    tier = 'C Tier' # Default
    
    # Thresholds
    S_TIER_PROB = cfg.MIN_PROB_FOR_S_TIER # e.g., 0.58
    S_TIER_EDGE = cfg.MIN_EDGE_FOR_S_TIER # e.g., 1.5
    
    A_TIER_PROB = 0.555
    A_TIER_EDGE = 1.0
    
    if is_divergent:
        # Penalize divergence heavily. 
        # Even if prob is high, if the math says negative edge, it's risky.
        tier = 'C Tier'
        # Override the pick to follow the Classifier (usually smarter), but mark it low confidence
        best_pick = model_pick
    else:
        best_pick = model_pick
        
        # Standard Tiering (Convergent Signals)
        if win_prob >= S_TIER_PROB and score >= S_TIER_EDGE:
            tier = 'S Tier'
        elif win_prob >= S_TIER_PROB and score >= A_TIER_EDGE:
            tier = 'A Tier' # High prob, decent edge
        elif win_prob >= A_TIER_PROB and score >= S_TIER_EDGE:
            tier = 'A Tier' # Good prob, huge edge
        elif win_prob >= A_TIER_PROB:
            tier = 'B Tier'
        elif score >= S_TIER_EDGE:
            tier = 'B Tier' # Low prob, massive edge (Value play)
        else:
            tier = 'C Tier'

    # 6. Injury Downgrades
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