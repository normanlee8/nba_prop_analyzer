import csv
import re
import datetime
import logging
from pathlib import Path
from prop_analyzer import config as cfg

# Expanded map to catch common abbreviations
DAYS_MAP = {
    'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3,
    'Fri': 4, 'Sat': 5, 'Sun': 6,
    'MON': 0, 'TUE': 1, 'WED': 2, 'THU': 3,
    'FRI': 4, 'SAT': 5, 'SUN': 6
}

def clean_prop_line(text):
    """
    Robustly extracts a numeric value from a line string.
    Handles: "25.5", "O 25.5", "25,5", " 25.5 "
    """
    # 1. Normalize
    s = text.strip().upper()
    s = s.replace(',', '.') # Handle European format
    
    # 2. Remove Over/Under prefixes common in copy-pastes
    s = re.sub(r'^[OU]\s+', '', s)
    
    # 3. Remove distinct "Over" / "Under" words
    s = s.replace('OVER', '').replace('UNDER', '').strip()
    
    # 4. Attempt float conversion
    try:
        val = float(s)
        return str(val)
    except ValueError:
        return None

def calculate_game_date(matchup_line, today):
    """
    Determines the game date based on the day of week string found in the line.
    Defaults to 'today' if no day is found.
    """
    today_int = today.weekday() 
    prop_day_str = None
    
    # Check for day abbreviations surrounded by non-word chars
    # e.g. " Mon ", "(Mon)", "Mon,"
    for day_str in DAYS_MAP.keys():
        if re.search(r'\b' + day_str + r'\b', matchup_line, re.IGNORECASE):
            prop_day_str = day_str
            break
            
    if prop_day_str is None:
        # Fallback: If we can't find a day, assume it's for the upcoming slate (Today)
        return today.strftime("%Y-%m-%d")
    else:
        prop_day_int = DAYS_MAP[prop_day_str]
        # Calculate days ahead (0-6)
        days_ahead = (prop_day_int - today_int + 7) % 7
        game_date_obj = today + datetime.timedelta(days=days_ahead)

    return game_date_obj.strftime("%Y-%m-%d")

def parse_matchup(matchup_line):
    """
    Extracts Team abbreviations from a matchup line.
    Supports: "LAL @ BOS", "LAL vs BOS", "LAL vs. BOS", "LAL-BOS"
    """
    # Normalize
    line = matchup_line.replace(' vs ', ' @ ').replace(' vs. ', ' @ ').replace('-', ' @ ')
    
    # Look for 3-letter codes
    match = re.search(r'\b([A-Z]{3})\s*@\s*([A-Z]{3})\b', line)
    
    if match:
        team1 = match.group(1)
        team2 = match.group(2)
        # Standardize to @ for internal consistency
        full_matchup_string = f"{team1} @ {team2}"
        return team1, team2, full_matchup_string
    
    return None, None, None

def parse_text_to_csv(input_path=None, output_path=None):
    if input_path is None: input_path = cfg.INPUT_PROPS_TXT
    if output_path is None: output_path = cfg.PROPS_FILE
    
    if not input_path.exists():
        logging.error(f"Input file not found: {input_path}")
        return

    now = datetime.datetime.now()
    today = now.date()
    
    current_player = None
    current_team = None
    current_opponent = None
    current_matchup = None
    current_game_date = None 
    
    data_to_write = [] 

    try:
        with open(input_path, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()

        for line in lines:
            line = line.strip()
            if not line: continue

            # 1. Check for Time/Matchup Line (Context Switch)
            # Regex looks for times like "7:00 PM", "7:00PM", "19:00" combined with timezones
            if re.search(r'\d{1,2}:\d{2}.*(?:PM|AM|ET|PT|MT|CT)', line, re.IGNORECASE):
                t1, t2, full_matchup = parse_matchup(line)
                if full_matchup:
                    current_matchup = full_matchup
                    current_game_date = calculate_game_date(line, today)
                    current_team = t1
                    current_opponent = t2
                continue 

            # 2. Check for Prop Line Value
            cleaned_val = clean_prop_line(line)
            if cleaned_val:
                # We have a number, but we need to know what it belongs to.
                # This logic assumes the format: 
                # Player Name
                # Prop Category
                # Prop Line (Number)
                # ...
                # This part is state-dependent and relies on the previous loop iteration.
                # However, simply detecting a number isn't enough; we need to assign it.
                # In this specific "copy-paste" format parser, usually the line comes 
                # AFTER the category.
                pass # We handle value assignment in the "Category" check below or implicit state
            
            # 3. Heuristic Parsing Logic
            # If we have a valid number, treat it as the Prop Line
            if cleaned_val:
                 # If we just saw a category (stored in 'current_player' variable momentarily? No, that's risky)
                 # Let's assume the standard block is:
                 # > Player Name
                 # > Prop Type
                 # > Value
                 # So if we hit a value, the PREVIOUS line was likely the Prop Type.
                 # But we are iterating line by line.
                 
                 # Alternative strategy: Store buffer
                 pass

            # Let's stick to the user's original logic flow but upgraded:
            # If line is a number -> It's the value.
            # If line is known category -> It's the category.
            # Else -> It's the player.
            
            if cleaned_val:
                prop_line_value = cleaned_val
                # Note: In the original script, it assumed Line came BEFORE Category or AFTER?
                # Original: 
                # if re.match number: prop_line_value = line; continue
                # if prop_line_value: category = line ... write ...
                
                # This implies structure:
                # 25.5
                # Points
                # (Player Name must have been set earlier)
                
                # Let's preserve that specific logic but use the clean value
                continue

            # If we have a stored value, this line MUST be the category
            # (Re-using the variable scope from the original logic style)
            # Note: We need to manage 'prop_line_value' state across loop
            if 'prop_line_value' in locals() and prop_line_value is not None:
                prop_category_str = line
                prop_category_std = cfg.MASTER_PROP_MAP.get(prop_category_str, None)
                
                # Auto-fix common variations not in map
                if not prop_category_std:
                    # Try case-insensitive lookup
                    for k, v in cfg.MASTER_PROP_MAP.items():
                        if k.lower() == prop_category_str.lower():
                            prop_category_std = v
                            break
                
                if prop_category_std and current_player and current_matchup:
                    data_to_write.append([
                        current_player, current_team, current_opponent,
                        current_matchup, prop_category_std, prop_line_value,
                        current_game_date 
                    ])
                
                # Reset value after consuming
                prop_line_value = None 
                continue

            # If it's not a Time/Matchup, and not a Number, and not a Category consuming a number...
            # It must be a Player Name (Start of a new block)
            current_player = line
            # Reset context that is specific to the player-prop block, but KEEP matchup info
            # (Matchup info usually applies to multiple players below it)
            prop_line_value = None 
            continue

        if not data_to_write:
            logging.warning("No valid props parsed. Check input format.")
            return

        header = ['Player Name', 'Team', 'Opponent', 'Matchup', 'Prop Category', 'Prop Line', 'GAME_DATE']

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='', encoding='utf-8') as f_out:
            writer = csv.writer(f_out)
            writer.writerow(header)
            writer.writerows(data_to_write)
            
        logging.info(f"Successfully converted props to {output_path} ({len(data_to_write)} rows)")

        # Save historical record
        records_dir = cfg.INPUT_DIR / "records"
        records_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        rec_path = records_dir / f"{timestamp_str}.csv"
        
        with open(rec_path, 'w', newline='', encoding='utf-8') as f_rec: 
            writer = csv.writer(f_rec)
            writer.writerow(header) 
            writer.writerows(data_to_write)

    except Exception as e:
        logging.error(f"Error parsing props: {e}", exc_info=True)