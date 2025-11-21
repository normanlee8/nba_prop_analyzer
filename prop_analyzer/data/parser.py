import csv
import re
import datetime
import logging
from pathlib import Path
from prop_analyzer import config as cfg

DAYS_MAP = {
    'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3,
    'Fri': 4, 'Sat': 5, 'Sun': 6
}

def calculate_game_date(matchup_line, today):
    today_int = today.weekday() 
    prop_day_str = None
    
    for day_str in DAYS_MAP.keys():
        if f" {day_str} " in matchup_line:
            prop_day_str = day_str
            break
            
    if prop_day_str is None:
        game_date_obj = today
    else:
        prop_day_int = DAYS_MAP[prop_day_str]
        days_ahead = (prop_day_int - today_int + 7) % 7
        game_date_obj = today + datetime.timedelta(days=days_ahead)

    return game_date_obj.strftime("%Y-%m-%d")

def parse_matchup(matchup_line):
    match = re.search(r'([A-Z]{3})\s+(?:vs|@)\s+([A-Z]{3})', matchup_line)
    if not match:
        return None, None, None
        
    team1 = match.group(1)
    team2 = match.group(2)
    full_matchup_string = f"{team1} @ {team2}" if ' @ ' in matchup_line else f"{team1} vs {team2}"
    return team1, team2, full_matchup_string

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
    prop_line_value = None 
    data_to_write = [] 

    try:
        with open(input_path, 'r') as f_in:
            lines = f_in.readlines()

        for line in lines:
            line = line.strip()
            if not line: continue

            if 'PM PST' in line or 'AM PST' in line or 'PM ET' in line or 'AM ET' in line:
                team1, team2, full_matchup = parse_matchup(line)
                if full_matchup:
                    current_matchup = full_matchup
                    current_game_date = calculate_game_date(line, today)
                    current_team = team1
                    current_opponent = team2
                continue 

            if re.match(r'^[0-9\.]+$', line):
                prop_line_value = line
                continue 

            if prop_line_value:
                prop_category_str = line
                prop_category_std = cfg.MASTER_PROP_MAP.get(prop_category_str, None)
                
                if prop_category_std and current_player and current_matchup:
                    data_to_write.append([
                        current_player, current_team, current_opponent,
                        current_matchup, prop_category_std, prop_line_value,
                        current_game_date 
                    ])
                prop_line_value = None
                continue 
                
            current_player = line
            current_matchup = None
            current_game_date = None
            current_team = None
            current_opponent = None
            prop_line_value = None 
            continue

        if not data_to_write:
            logging.warning("No valid props parsed.")
            return

        header = ['Player Name', 'Team', 'Opponent', 'Matchup', 'Prop Category', 'Prop Line', 'GAME_DATE']

        with open(output_path, 'w', newline='') as f_out:
            writer = csv.writer(f_out)
            writer.writerow(header)
            writer.writerows(data_to_write)
            
        logging.info(f"Successfully converted props to {output_path} ({len(data_to_write)} rows)")

        # Optional: Save historical record
        records_dir = cfg.INPUT_DIR / "records"
        records_dir.mkdir(exist_ok=True)
        timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        rec_path = records_dir / f"{timestamp_str}.csv"
        
        with open(rec_path, 'w', newline='') as f_rec: 
            writer = csv.writer(f_rec)
            writer.writerow(header) 
            writer.writerows(data_to_write)

    except Exception as e:
        logging.error(f"Error parsing props: {e}")