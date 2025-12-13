import sys
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from prop_analyzer import config as cfg
from prop_analyzer.config import Cols
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
    
    # Wrap execution in try/except for robustness
    try:
        # 1. Load Today's Props
        props_path = cfg.PROPS_FILE
        if not props_path.exists():
            logging.critical(f"Props file not found: {props_path}")
            logging.critical("Please run 'scripts/run_converter.py' or provide input.")
            return

        try:
            props_df = pd.read_csv(props_path)
            if props_df.empty:
                logging.warning("props_today.csv is empty.")
                return
                
            # Basic sanitization
            props_df.columns = props_df.columns.str.strip()
            
            # Validate Schema using Data Contract
            required = Cols.get_required_input_cols()
            missing = [c for c in required if c not in props_df.columns]
            
            if missing:
                logging.critical(f"CRITICAL ERROR: Input file missing required columns: {missing}")
                return

            logging.info(f"Loaded {len(props_df)} props. Schema verified.")
            
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
        if Cols.CONFIDENCE not in results_df.columns:
            results_df[Cols.CONFIDENCE] = 0.0
        
        # --- SORTING LOGIC ---
        tier_map = {'S Tier': 0, 'A Tier': 1, 'B Tier': 2, 'C Tier': 3}
        results_df['Tier_Rank'] = results_df[Cols.TIER].map(tier_map).fillna(99)
        results_df.sort_values(by=['Tier_Rank', Cols.CONFIDENCE], ascending=[True, False], inplace=True)
        
        # --- FORMATTING ---
        # Standardize Date format for display
        if Cols.DATE in results_df.columns:
            results_df[Cols.DATE] = pd.to_datetime(results_df[Cols.DATE], errors='coerce').dt.strftime('%Y-%m-%d')
            results_df[Cols.DATE] = results_df[Cols.DATE].fillna("N/A")

        # Select columns to keep (Using Cols constants)
        keep_cols = [
            Cols.PLAYER_NAME, Cols.TEAM, Cols.OPPONENT, Cols.PROP_TYPE, Cols.PROP_LINE, 
            Cols.DATE,
            Cols.PREDICTION, Cols.CONFIDENCE, Cols.EDGE_TYPE, Cols.TIER,
            f'Diff%', f'{Cols.L5_AVG}', f'{Cols.SZN_AVG}'
        ]
        
        # Filter for existing columns (handle case where L5/SZN avg might be named differently in raw output)
        # We try to find the matching columns dynamically if exact match fails
        final_cols = []
        for c in keep_cols:
            if c in results_df.columns:
                final_cols.append(c)
            # Try finding without prefix if needed, though training.py should be consistent now
            elif c == f'{Cols.SZN_AVG}' and 'SZN_AVG' in results_df.columns:
                 final_cols.append('SZN_AVG')

        final_output = results_df[final_cols].copy()

        # Rename Columns for Readability (Console Output)
        display_map = {
            Cols.PLAYER_NAME: 'Player',
            Cols.PROP_TYPE: 'Prop',
            Cols.PROP_LINE: 'Line',
            Cols.PREDICTION: 'Proj',
            Cols.CONFIDENCE: 'Prob',
            Cols.EDGE_TYPE: 'Pick',
            Cols.DATE: 'Date',
            f'{Cols.L5_AVG}': 'L5',
            f'{Cols.SZN_AVG}': 'SZN'
        }
        final_output.rename(columns=display_map, inplace=True)

        # Format Prob as Percentage
        if 'Prob' in final_output.columns:
            final_output['Prob'] = final_output['Prob'].apply(lambda x: f"{x*100:.1f}%")

        # 5. Save Results
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        outfile = cfg.PROCESSED_OUTPUT
        record_file = cfg.INPUT_DIR / "records" / f"analysis_{timestamp}.csv"
        
        outfile.parent.mkdir(parents=True, exist_ok=True)
        record_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save RAW standardized output to CSV (for Grading script)
        # We save results_df (with Cols schema) to file, NOT the pretty-printed version
        # This ensures Grading script can read it machine-reliably
        results_df.to_csv(outfile, index=False)
        results_df.to_csv(record_file, index=False)
        
        logging.info(f"Saved analysis to: {outfile}")
        
        # 6. Pretty Print to Console (Top 20)
        print_pretty_table(final_output.head(20))

        logging.info("<<< ANALYSIS COMPLETE >>>")
        
    except Exception as e:
        logging.critical(f"FATAL ERROR in Analysis Pipeline: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()