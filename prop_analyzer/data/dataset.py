import pandas as pd
import numpy as np
import logging
from prop_analyzer import config as cfg
from prop_analyzer.config import Cols
from prop_analyzer.data import loader
from prop_analyzer.features import generator

def create_training_dataset():
    """
    Builds the final training dataset (master_training_dataset.parquet).
    Combines:
    1. Master Box Scores (Full Game)
    2. Master Q1 Stats (Targets for Q1 props)
    3. Master 1H Stats (Targets for 1H props)
    4. Rolling Features (SZN_AVG, L5, etc.) calculated by generator.py
    """
    logging.info("--- Building Final Training Dataset ---")
    
    # 1. Load Base Box Scores
    box_scores = loader.load_box_scores()
    if box_scores is None or box_scores.empty:
        logging.error("No box scores available. Cannot build training set.")
        return

    # 2. Load Quarter/Half Targets
    q1_df = loader.load_master_q1_history()
    h1_df = loader.load_master_1h_history()
    
    # 3. Merge Q1 Data (Target Columns)
    if not q1_df.empty:
        logging.info(f"Merging {len(q1_df)} Q1 records...")
        
        # Prepare for merge
        q1_df = q1_df.rename(columns={
            'PTS': 'Q1_PTS', 'REB': 'Q1_REB', 'AST': 'Q1_AST',
            'FG3M': 'Q1_FG3M', 'PRA': 'Q1_PRA'
        })
        
        # Select only necessary columns to avoid conflicts
        cols_to_merge = [Cols.PLAYER_ID, 'GAME_DATE', 'Q1_PTS', 'Q1_REB', 'Q1_AST', 'Q1_FG3M', 'Q1_PRA']
        q1_subset = q1_df[[c for c in cols_to_merge if c in q1_df.columns]]
        
        # Merge onto Box Scores
        box_scores = pd.merge(
            box_scores, 
            q1_subset,
            on=[Cols.PLAYER_ID, 'GAME_DATE'],
            how='left'
        )
    
    # 4. Merge 1H Data (Target Columns)
    if not h1_df.empty:
        logging.info(f"Merging {len(h1_df)} 1H records...")
        
        h1_df = h1_df.rename(columns={
            'PTS': '1H_PTS', 'REB': '1H_REB', 'AST': '1H_AST',
            'FG3M': '1H_FG3M', 'PRA': '1H_PRA'
        })
        
        cols_to_merge = [Cols.PLAYER_ID, 'GAME_DATE', '1H_PTS', '1H_REB', '1H_AST', '1H_FG3M', '1H_PRA']
        h1_subset = h1_df[[c for c in cols_to_merge if c in h1_df.columns]]
        
        box_scores = pd.merge(
            box_scores,
            h1_subset,
            on=[Cols.PLAYER_ID, 'GAME_DATE'],
            how='left'
        )

    # 5. Generate Features (Rolling Averages, etc.)
    # This adds the PRE-GAME context (e.g., L5_AVG) needed for training
    logging.info("Calculating features for training set...")
    
    # A. Full Game Rolling
    training_df = generator.add_rolling_stats_history(box_scores.copy())
    
    # B. Q1 Rolling (Calculate rolling stats of the Q1 targets)
    if 'Q1_PTS' in training_df.columns:
        training_df = generator.add_rolling_stats_history(
            training_df, 
            stats_to_roll=['Q1_PTS', 'Q1_REB', 'Q1_AST', 'Q1_PRA', 'Q1_FG3M']
        )
        
    # C. 1H Rolling
    if '1H_PTS' in training_df.columns:
        training_df = generator.add_rolling_stats_history(
            training_df, 
            stats_to_roll=['1H_PTS', '1H_REB', '1H_AST', '1H_PRA', '1H_FG3M']
        )

    # 6. Save Final Dataset
    logging.info(f"Saving training set with {training_df.shape[1]} columns...")
    training_df.to_parquet(cfg.MASTER_TRAINING_FILE, index=False)
    logging.info(f"Saved to {cfg.MASTER_TRAINING_FILE}")

if __name__ == "__main__":
    # Setup simple console logging if run directly
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    create_training_dataset()