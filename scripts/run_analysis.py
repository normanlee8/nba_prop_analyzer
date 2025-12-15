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

def save_pretty_excel(df, output_path):
    """
    Saves the dataframe to Excel.
    """
    try:
        if df.empty: return

        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
        
        # Write data
        df.to_excel(writer, sheet_name='Picks', index=False)
        
        # Get workbook/worksheet objects
        workbook = writer.book
        worksheet = writer.sheets['Picks']
        
        # --- Formats ---
        pct_fmt = workbook.add_format({'num_format': '0.0%'})
        header_fmt = workbook.add_format({'bold': True, 'bottom': 1, 'bg_color': '#F0F0F0'})

        # Apply Header Format
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_fmt)

        # Apply Column Widths & Num Formats
        for i, col in enumerate(df.columns):
            max_len = max(df[col].astype(str).map(len).max(), len(str(col)))
            width = min(max_len + 2, 50)
            
            if col == 'Prob':
                worksheet.set_column(i, i, width, pct_fmt)
            else:
                worksheet.set_column(i, i, width)

        writer.close()
        logging.info(f"Saved Excel analysis to: {output_path}")
        
    except Exception as e:
        logging.error(f"Failed to save Excel file: {e}")

def print_pretty_table(df, title="TOP 20 DISCOVERED EDGES"):
    if df.empty:
        print("No results to display.")
        return

    df_str = df.astype(str)
    widths = [max(df_str[col].apply(len).max(), len(col)) + 2 for col in df.columns]
    fmt = "| " + " | ".join([f"{{:<{w}}}" for w in widths]) + " |"

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
    except Exception:
        print(df.head(20))

def main():
    common.setup_logging(name="analysis_pregame")
    logging.info(">>> STARTING PRE-GAME PROP ANALYSIS <<<")
    
    try:
        # 1. Load Today's Props
        props_path = cfg.PROPS_FILE
        if not props_path.exists():
            logging.critical(f"Props file not found: {props_path}")
            return

        try:
            props_df = pd.read_csv(props_path)
            if props_df.empty:
                logging.warning("props_today.csv is empty.")
                return
                
            props_df.columns = props_df.columns.str.strip()
            
            required = Cols.get_required_input_cols()
            missing = [c for c in required if c not in props_df.columns]
            
            if missing:
                logging.critical(f"CRITICAL ERROR: Input file missing required columns: {missing}")
                return

            logging.info(f"Loaded {len(props_df)} props.")
            
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
        
        tier_map = {'S Tier': 0, 'A Tier': 1, 'B Tier': 2, 'C Tier': 3}
        results_df['Tier_Rank'] = results_df[Cols.TIER].map(tier_map).fillna(99)
        results_df.sort_values(by=['Tier_Rank', Cols.CONFIDENCE], ascending=[True, False], inplace=True)
        
        if Cols.DATE in results_df.columns:
            results_df[Cols.DATE] = pd.to_datetime(results_df[Cols.DATE], errors='coerce').dt.strftime('%Y-%m-%d')
            results_df[Cols.DATE] = results_df[Cols.DATE].fillna("N/A")

        keep_cols = [
            Cols.PLAYER_NAME, Cols.TEAM, Cols.OPPONENT, Cols.PROP_TYPE, Cols.PROP_LINE, 
            Cols.DATE,
            Cols.PREDICTION, Cols.CONFIDENCE, Cols.EDGE_TYPE, Cols.TIER,
            f'Diff%', f'{Cols.L5_AVG}', f'{Cols.SZN_AVG}'
        ]
        
        final_cols = [c for c in keep_cols if c in results_df.columns]
        final_output = results_df[final_cols].copy()

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

        # 5. Save Results (FIXED)
        # A. Save System Parquet (Replaces CSV)
        results_df.to_parquet(cfg.PROCESSED_OUTPUT_SYSTEM, index=False)
        logging.info(f"Saved system results to {cfg.PROCESSED_OUTPUT_SYSTEM}")
        
        # B. Save Human Excel
        save_pretty_excel(final_output, cfg.PROCESSED_OUTPUT_XLSX)
        
        # C. Console
        console_output = final_output.copy()
        if 'Prob' in console_output.columns:
            console_output['Prob'] = console_output['Prob'].apply(lambda x: f"{x*100:.1f}%")
            
        print_pretty_table(console_output.head(20))

        logging.info("<<< ANALYSIS COMPLETE >>>")
        
    except Exception as e:
        logging.critical(f"FATAL ERROR in Analysis Pipeline: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()