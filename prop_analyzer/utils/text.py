import re
import logging
from unidecode import unidecode
from rapidfuzz import process, fuzz

def preprocess_name_for_fuzzy_match(name):
    """
    Standardizes player names for fuzzy matching (removes suffixes, accents, etc.).
    """
    if not name or str(name).lower() == 'nan':
        return name
        
    # Remove suffixes like Jr., Sr., II, III, IV
    name = re.sub(r'\s+(jr|sr|ii|iii|iv|v)\.?$', '', str(name), flags=re.IGNORECASE)
    name = unidecode(name).lower()
    name = re.sub(r'[^a-z0-9\s]', '', name)
    name = ' '.join(name.split())
    
    return name

def fuzzy_match_player(player_name, player_stats_df):
    """
    Matches a raw player name string to a row in the master player dataframe.
    Requires 'processed_name' column to exist in player_stats_df.
    """
    if player_stats_df is None or player_stats_df.empty:
        return None
        
    # Create processed list if not present (safety check)
    if 'processed_name' not in player_stats_df.columns:
        player_stats_df['processed_name'] = player_stats_df['clean_name'].apply(preprocess_name_for_fuzzy_match)

    choices_series = player_stats_df['processed_name'].dropna()
    choices_list = choices_series.tolist()
    
    clean_input = preprocess_name_for_fuzzy_match(player_name)
    
    if not choices_list:
        return None

    match = process.extractOne(clean_input, choices_list, scorer=fuzz.token_sort_ratio, score_cutoff=90)
    
    if match:
        matched_processed_name, score, _ = match
        # Find the index in the original DF
        original_index = choices_series[choices_series == matched_processed_name].index[0]
        player_data = player_stats_df.loc[original_index]
        return player_data
    else:
        logging.debug(f"No sufficient fuzzy match found for player: {player_name}")
        return None