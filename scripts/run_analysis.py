import sys
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from prop_analyzer import config as cfg
from prop_analyzer.features import generator
from prop_analyzer.models import inference
from prop_analyzer.utils import common

def print_pretty_table(df, title="TOP 20 DISCOVERED EDGES"):
    """
    Prints a DataFrame in a clean, grid-like format using | and =.
    """
    if df.empty:
        print("No results to display.")
        return

    # Convert all data to string to calculate widths
    df_str = df.astype(str)
    
    # Calculate column widths based on max length of data or header
    widths = []
    for col in df.columns:
        # max length of column data or column name
        max_len = max(df_str[col].apply(len).max(), len(col))
        widths.append(max_len + 2) # Add padding

    # Create format string (e.g., "| {:<10} | {:<5} | ...")
    fmt_parts = []
    for w in widths:
        fmt_parts.append(f"{{:<{w}}}")
    fmt = "| " + " | ".join(fmt_parts) + " |"

    # Create Separator Line
    try:
        header_str = fmt.format(*df.columns)
        sep_line = "=" * len(header_str)

        print(f"\n{title}")
        print(sep_line)
        print(header_str)
        print(sep_line)

        for _, row in df.iterrows():
            print(fmt.format(*row.values))

        print(sep_line + "\n")
    except Exception as e:
        logging.error(f"Error printing table: {e}")
        # Fallback to standard pandas print
        print(df.head(20))

def main():
    common.setup_logging(name="analysis_pregame")
    logging.info(">>> STARTING PRE-GAME PROP ANALYSIS <<<")
    
    # 1. Load Today's Props
    props_path = cfg.INPUT_DIR / "props_today.csv"
    if not props_path.exists():
        logging.critical(f"Props file not found: {props_path}")
        logging.critical("Please run 'scripts/run_scrape.py' or provide input.")
        return

    try:
        props_df = pd.read_csv(props_path)
        if props_df.empty:
            logging.warning("props_today.csv is empty.")
            return
            
        # --- ROBUST COLUMN HANDLING ---
        props_df.columns = props_df.columns.str.strip()
        
        # Use 'Prop Category' directly (standardized in parser.py)
        if 'Prop Category' not in props_df.columns:
            logging.critical(f"CRITICAL ERROR: 'Prop Category' column missing.")
            logging.critical(f"Available Columns: {list(props_df.columns)}")
            return

        logging.info(f"Loaded {len(props_df)} props. Columns verified.")
        
    except Exception as e:
        logging.critical(f"Failed to read props file: {e}")
        return

    # 2. Build Feature Vectors
    try:
        features_df = generator.build_feature_set(props_df)
        if features_df.empty:
            logging.critical("Feature generation returned empty dataset.")
            return
    except Exception as e:
        logging.critical(f"Feature generation failed: {e}", exc_info=True)
        return

    # 3. Run Inference
    logging.info("Running Machine Learning Inference...")
    try:
        results_df = inference.predict_props(features_df)
    except Exception as e:
        logging.critical(f"Inference process crashed: {e}", exc_info=True)
        return
    
    if results_df is None or results_df.empty:
        logging.warning("No predictions were generated.")
        return

    # 4. Filter & Format Output
    if 'Model_Conf' not in results_df.columns:
        results_df['Model_Conf'] = 0.0
    
    # --- SORTING LOGIC ---
    tier_map = {'S Tier': 0, 'A Tier': 1, 'B Tier': 2, 'C Tier': 3}
    results_df['Tier_Rank'] = results_df['Tier'].map(tier_map).fillna(99)
    results_df.sort_values(by=['Tier_Rank', 'Model_Conf'], ascending=[True, False], inplace=True)
    
    # --- FIX: Date Normalization & Formatting ---
    # 1. Standardize column name (Handle 'Game_date' vs 'GAME_DATE')
    if 'Game_date' in results_df.columns:
        results_df.rename(columns={'Game_date': 'GAME_DATE'}, inplace=True)
    
    # 2. Format Date to YYYY-MM-DD string to prevent table print issues
    if 'GAME_DATE' in results_df.columns:
        results_df['GAME_DATE'] = pd.to_datetime(results_df['GAME_DATE'], errors='coerce').dt.strftime('%Y-%m-%d')
        results_df['GAME_DATE'] = results_df['GAME_DATE'].fillna("N/A")

    # Rename Columns for Readability
    rename_map = {
        'Player Name': 'Player',
        'Prop Category': 'Prop',
        'Prop Line': 'Line',
        'Model_Pred': 'Proj',
        'Model_Conf': 'Prob',
        'Edge_Type': 'Pick',
        'GAME_DATE': 'Date'  # Now guaranteed to match
    }
    
    # Select columns to keep
    keep_cols = [
        'Player Name', 'Team', 'Opponent', 'Prop Category', 'Prop Line', 
        'GAME_DATE',
        'Model_Pred', 'Model_Conf', 'Edge_Type', 'Tier',
        'Last 5', 'Season Avg', 'Diff%'
    ]
    
    # Filter for existing columns
    existing_cols = [c for c in keep_cols if c in results_df.columns]
    
    # Create final output and rename
    final_output = results_df[existing_cols].copy()
    final_output.rename(columns=rename_map, inplace=True)

    # Format Prob as Percentage (e.g., 0.927 -> 92.7%)
    if 'Prob' in final_output.columns:
        final_output['Prob'] = final_output['Prob'].apply(lambda x: f"{x*100:.1f}%")

    # 5. Save Results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outfile = cfg.OUTPUT_DIR / "processed_props.csv" 
    record_file = cfg.INPUT_DIR / "records" / f"{timestamp}.csv"
    
    outfile.parent.mkdir(parents=True, exist_ok=True)
    record_file.parent.mkdir(parents=True, exist_ok=True)
    
    final_output.to_csv(outfile, index=False)
    final_output.to_csv(record_file, index=False)
    
    logging.info(f"Saved analysis to: {outfile}")
    
    # 6. Pretty Print to Console (Top 20)
    print_pretty_table(final_output.head(20))

    logging.info("<<< ANALYSIS COMPLETE >>>")

if __name__ == "__main__":
    main()