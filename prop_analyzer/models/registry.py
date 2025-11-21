import os
import joblib
import logging
from pathlib import Path
from prop_analyzer import config as cfg

def get_model_paths(prop_cat):
    """Returns a dict of file paths for a specific prop category."""
    base = cfg.MODEL_DIR
    return {
        'scaler': base / f"scaler_{prop_cat}.pkl",
        'features': base / f"features_{prop_cat}.pkl",
        'lgbm_q20': base / f"model_{prop_cat}_lgbm_q20.pkl",
        'xgb_q20': base / f"model_{prop_cat}_xgb_q20.pkl",
        'lgbm_q80': base / f"model_{prop_cat}_lgbm_q80.pkl",
        'xgb_q80': base / f"model_{prop_cat}_xgb_q80.pkl",
        'clf': base / f"model_{prop_cat}_clf.pkl"
    }

def save_artifacts(prop_cat, artifacts):
    """
    Saves trained models and preprocessors to disk.
    artifacts: dict containing 'scaler', 'features', 'q20', 'q80', 'clf'
    """
    paths = get_model_paths(prop_cat)
    
    # Create directory if not exists
    cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        joblib.dump(artifacts['scaler'], paths['scaler'])
        joblib.dump(artifacts['features'], paths['features'])
        
        joblib.dump(artifacts['q20']['lgbm'], paths['lgbm_q20'])
        joblib.dump(artifacts['q20']['xgb'], paths['xgb_q20'])
        
        joblib.dump(artifacts['q80']['lgbm'], paths['lgbm_q80'])
        joblib.dump(artifacts['q80']['xgb'], paths['xgb_q80'])
        
        joblib.dump(artifacts['clf'], paths['clf'])
        logging.info(f"Saved models for {prop_cat}")
    except Exception as e:
        logging.error(f"Failed to save artifacts for {prop_cat}: {e}")

def load_model_cache(model_dir=None):
    """
    Loads ALL models into memory. 
    Returns a dictionary: { 'PTS': { ...models... }, 'REB': { ... } }
    """
    if model_dir is None:
        model_dir = cfg.MODEL_DIR
        
    model_cache = {}
    if not model_dir.exists():
        logging.error(f"Model directory not found: {model_dir}")
        return None

    # Find all categories based on feature files
    prop_categories = set()
    for f in os.listdir(model_dir):
        if f.startswith('features_') and f.endswith('.pkl'):
            cat = f.replace('features_', '').replace('.pkl', '')
            prop_categories.add(cat)

    if not prop_categories:
        logging.warning(f"No models found in {model_dir}")
        return None

    logging.info(f"Loading models for {len(prop_categories)} categories...")
    
    for prop_cat in prop_categories:
        paths = get_model_paths(prop_cat)
        try:
            # Check if essential files exist
            if not paths['features'].exists(): continue

            cache_entry = {
                'scaler': joblib.load(paths['scaler']),
                'features': joblib.load(paths['features']),
                'q20': {
                    'lgbm': joblib.load(paths['lgbm_q20']),
                    'xgb': joblib.load(paths['xgb_q20'])
                },
                'q80': {
                    'lgbm': joblib.load(paths['lgbm_q80']),
                    'xgb': joblib.load(paths['xgb_q80'])
                },
                'clf': joblib.load(paths['clf']) if paths['clf'].exists() else None
            }
            model_cache[prop_cat] = cache_entry
        except Exception as e:
            logging.error(f"Error loading {prop_cat}: {e}")

    return model_cache