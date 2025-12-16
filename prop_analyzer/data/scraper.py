import pandas as pd
import requests
import time
import logging
import sys
import re
import io
import random
import concurrent.futures
from pathlib import Path
from datetime import datetime, timedelta
from bs4 import BeautifulSoup, Comment
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from nba_api.stats.static import players

# Import project config
from prop_analyzer import config as cfg

try:
    # We primarily use LeagueDashPlayerStats now for batch fetching
    from nba_api.stats.endpoints.leaguedashplayerstats import LeagueDashPlayerStats
    from nba_api.stats.endpoints.leaguedashteamstats import LeagueDashTeamStats
    from nba_api.stats.endpoints.leaguedashptdefend import LeagueDashPtDefend
    from nba_api.stats.endpoints.leaguedashoppptshot import LeagueDashOppPtShot
except ImportError as e:
    print("--- FATAL ERROR ---")
    print(f"Failed to import a module from 'nba-api': {e}")
    sys.exit(1)

# --- DYNAMIC CONFIGURATION ---

def get_season_config():
    now = datetime.now()
    if now.month >= 10:
        current_start_year = now.year
    else:
        current_start_year = now.year - 1
        
    current_end_year = current_start_year + 1
    prev_start_year = current_start_year - 1
    prev_end_year = current_start_year

    curr_season_str = f"{current_start_year}-{str(current_end_year)[-2:]}"
    prev_season_str = f"{prev_start_year}-{str(prev_end_year)[-2:]}"

    return [
        {
            "id": "last_season",
            "season_str": prev_season_str,
            "bball_ref_year": prev_end_year,
            "is_current": False,
            "tr_date_param": f"{prev_end_year}-07-01" 
        },
        {
            "id": "current_season",
            "season_str": curr_season_str,
            "bball_ref_year": current_end_year,
            "is_current": True,
            "tr_date_param": None
        }
    ]

# --- TUNING SETTINGS ---
# 2 Workers is the safest for stability.
# Timeout increased to 120s to handle server stalls.
MAX_WORKERS = 2  
NBA_API_TIMEOUT = 120

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Referer': 'https://www.nba.com/',
    'Origin': 'https://www.nba.com/',
    'Accept-Language': 'en-US,en;q=0.9',
    'Connection': 'keep-alive'
}

# --- MAPPINGS ---
TEAM_NAME_MAP = {
    "Atlanta": "ATL", "Atlanta Hawks": "ATL",
    "Boston": "BOS", "Boston Celtics": "BOS",
    "Brooklyn": "BKN", "Brooklyn Nets": "BKN",
    "Charlotte": "CHA", "Charlotte Hornets": "CHA",
    "Chicago": "CHI", "Chicago Bulls": "CHI",
    "Cleveland": "CLE", "Cleveland Cavaliers": "CLE",
    "Dallas": "DAL", "Dallas Mavericks": "DAL",
    "Denver": "DEN", "Denver Nuggets": "DEN",
    "Detroit": "DET", "Detroit Pistons": "DET",
    "Golden State": "GSW", "Golden State Warriors": "GSW",
    "Houston": "HOU", "Houston Rockets": "HOU",
    "Indiana": "IND", "Indiana Pacers": "IND",
    "LA Clippers": "LAC", "Los Angeles Clippers": "LAC",
    "LA Lakers": "LAL", "Los Angeles Lakers": "LAL",
    "Memphis": "MEM", "Memphis Grizzlies": "MEM",
    "Miami": "MIA", "Miami Heat": "MIA",
    "Milwaukee": "MIL", "Milwaukee Bucks": "MIL",
    "Minnesota": "MIN", "Minnesota Timberwolves": "MIN",
    "New Orleans": "NOP", "New Orleans Pelicans": "NOP",
    "New York": "NYK", "New York Knicks": "NYK",
    "Okla City": "OKC", "Oklahoma City Thunder": "OKC",
    "Orlando": "ORL", "Orlando Magic": "ORL",
    "Philadelphia": "PHI", "Philadelphia 76ers": "PHI",
    "Phoenix": "PHX", "Phoenix Suns": "PHX",
    "Portland": "POR", "Portland Trail Blazers": "POR",
    "Sacramento": "SAC", "Sacramento Kings": "SAC",
    "San Antonio": "SAS", "San Antonio Spurs": "SAS",
    "Toronto": "TOR", "Toronto Raptors": "TOR",
    "Utah": "UTA", "Utah Jazz": "UTA",
    "Washington": "WAS", "Washington Wizards": "WAS",
}

TEAMRANKINGS_SLUG_MAP = {
    "Points per Game": "points-per-game",
    "Average Scoring Margin": "average-scoring-margin",
    "Offensive Efficiency": "offensive-efficiency",
    "Floor %": "floor-percentage",
    "1st Quarter Points per Game": "1st-quarter-points-per-game",
    "2nd Quarter Points per Game": "2nd-quarter-points-per-game",
    "3rd Quarter Points per Game": "3rd-quarter-points-per-game",
    "4th Quarter Points per Game": "4th-quarter-points-per-game",
    "1st Half Points per Game": "1st-half-points-per-game",
    "2nd Half Points per Game": "2nd-half-points-per-game",
    "Overtime Points per Game": "overtime-points-per-game",
    "Points in Paint per Game": "points-in-paint-per-game",
    "Fastbreak Points per Game": "fastbreak-points-per-game",
    "Fastbreak Efficiency": "fastbreak-efficiency",
    "Average Biggest Lead": "average-biggest-lead",
    "Average 1st Quarter Margin": "average-1st-quarter-margin",
    "Average 2nd Quarter Margin": "average-2nd-quarter-margin",
    "Average 3rd Quarter Margin": "average-3rd-quarter-margin",
    "Average 4th Quarter Margin": "average-4th-quarter-margin",
    "Average 1st Half Margin": "average-1st-half-margin",
    "Average 2nd Half Margin": "average-2nd-half-margin",
    "Average Overtime Margin": "average-overtime-margin",
    "Average Margin Thru 3 Quarters": "average-margin-thru-3-quarters",
    "Points from 2 pointers": "points-from-2-pointers",
    "Points from 3 pointers": "points-from-3-pointers",
    "Percent of Points from 2 Pointers": "percent-of-points-from-2-pointers",
    "Percent of Points from 3 Pointers": "percent-of-points-from-3-pointers",
    "Percent of Points from Free Throws": "percent-of-points-from-free-throws",
    "Shooting %": "shooting-pct",
    "Effective Field Goal %": "effective-field-goal-pct",
    "Three Point %": "three-point-pct",
    "Two Point %": "two-point-pct",
    "Free Throw %": "free-throw-pct",
    "True Shooting %": "true-shooting-percentage",
    "Field Goals Made per Game": "field-goals-made-per-game",
    "Field Goals Attempted per Game": "field-goals-attempted-per-game",
    "Three Pointers Made per Game": "three-pointers-made-per-game",
    "Three Pointers Attempted per Game": "three-pointers-attempted-per-game",
    "Free Throws Made per Game": "free-throws-made-per-game",
    "Free Throws Attempted per Game": "free-throws-attempted-per-game",
    "Three Point Rate": "three-point-rate",
    "Two Point Rate": "two-point-rate",
    "FTA per FGA": "fta-per-fga",
    "FTM per 100 Possessions": "ftm-per-100-possessions",
    "Free Throws Attempted per Offensive Play": "free-throw-rate",
    "Non-blocked 2 Pt %": "non-blocked-2-pt-pct",
    "Offensive Rebounds per Game": "offensive-rebounds-per-game",
    "Defensive Rebounds per Game": "defensive-rebounds-per-game",
    "Team Rebounds per Game": "team-rebounds-per-game",
    "Total Rebounds per Game": "total-rebounds-per-game",
    "Offensive Rebounding %": "offensive-rebounding-pct",
    "Defensive Rebounding %": "defensive-rebounding-pct",
    "Total Rebounding % (Rebound Rate)": "total-rebounding-percentage",
    "Blocks per Game": "blocks-per-game",
    "Steals per Game": "steals-per-game",
    "Block %": "block-pct",
    "Steals per Defensive Play": "steal-pct",
    "Assists per Game": "assists-per-game",
    "Turnovers per Game": "turnovers-per-game",
    "Turnovers per Possession": "turnovers-per-possession",
    "Assist / Turnover Ratio": "assist--per--turnover-ratio",
    "Assists per FGM": "assists-per-fgm",
    "Assists per Possession": "assists-per-possession",
    "Turnovers per Offensive Play": "turnover-pct",
    "Personal Fouls per Game": "personal-fouls-per-game",
    "Technical Fouls per Game": "technical-fouls-per-game",
    "Personal Fouls per Possession": "personal-fouls-per-possession",
    "Personal Fouls per Defensive Play": "personal-foul-pct",
    "Opponent Points per Game": "opponent-points-per-game",
    "Opponent Average Scoring Margin": "opponent-average-scoring-margin",
    "Defensive Efficiency": "defensive-efficiency",
    "Opponent Floor %": "opponent-floor-percentage",
    "Opponent 1st Quarter Points per Game": "opponent-1st-quarter-points-per-game",
    "Opponent 2nd Quarter Points per Game": "opponent-2nd-quarter-points-per-game",
    "Opponent 3rd Quarter Points per Game": "opponent-3rd-quarter-points-per-game",
    "Opponent 4th Quarter Points per Game": "opponent-4th-quarter-points-per-game",
    "Opponent Overtime Points per Game": "opponent-overtime-points-per-game",
    "Opponent Points in Paint per Game": "opponent-points-in-paint-per-game",
    "Opponent Fastbreak Points per Game": "opponent-fastbreak-points-per-game",
    "Opponent Fastbreak Efficiency": "opponent-fastbreak-efficiency",
    "Opponent Average Biggest Lead": "opponent-average-biggest-lead",
    "Opponent 1st Half Points per Game": "opponent-1st-half-points-per-game",
    "Opponent 2nd Half Points per Game": "opponent-2nd-half-points-per-game",
    "Opponent Points from 2 pointers": "opponent-points-from-2-pointers",
    "Opponent Points from 3 pointers": "opponent-points-from-3-pointers",
    "Opponent Percent of Points from 2 Pointers": "opponent-percent-of-points-from-2-pointers",
    "Opponent Percent of Points from 3 Pointers": "opponent-percent-of-points-from-3-pointers",
    "Opponent Percent of Points from Free Throws": "opponent-percent-of-points-from-free-throws",
    "Opponent Shooting %": "opponent-shooting-pct",
    "Opponent Effective Field Goal %": "opponent-effective-field-goal-pct",
    "Opponent Three Point %": "opponent-three-point-pct",
    "Opponent Two Point %": "opponent-two-point-pct",
    "Opponent Free Throw %": "opponent-free-throw-pct",
    "Opponent True Shooting %": "opponent-true-shooting-percentage",
    "Opponent Field Goals Made per Game": "opponent-field-goals-made-per-game",
    "Opponent Field Goals Attempted per Game": "opponent-field-goals-attempted-per-game",
    "Opponent Three Pointers Made per Game": "opponent-three-pointers-made-per-game",
    "Opponent Three Pointers Attempted per Game": "opponent-three-pointers-attempted-per-game",
    "Opponent Free Throws Made per Game": "opponent-free-throws-made-per-game",
    "Opponent Free Throws Attempted per Game": "opponent-free-throws-attempted-per-game",
    "Opponent Three Point Rate": "opponent-three-point-rate",
    "Opponent Two Point Rate": "opponent-two-point-rate",
    "Opponent FTA per FGA": "opponent-fta-per-fga",
    "Opponent Free Throws Made per 100 Possessions": "opponent-ftm-per-100-possessions",
    "Opponent Free Throws Attempted per Offensive Play": "opponent-free-throw-rate",
    "Opponent Non-blocked 2 Pt %": "opponent-non-blocked-2-pt-pct",
    "Opponent Offensive Rebounds per Game": "opponent-offensive-rebounds-per-game",
    "Opponent Defensive Rebounds per Game": "opponent-defensive-rebounds-per-game",
    "Opponent Team Rebounds per Game": "opponent-team-rebounds-per-game",
    "Opponent Total Rebounds per Game": "opponent-total-rebounds-per-game",
    "Opponent Offensive Rebounding %": "opponent-offensive-rebounding-pct",
    "Opponent Defensive Rebounding %": "opponent-defensive-rebounding-pct",
    "Opponent Blocks per Game": "opponent-blocks-per-game",
    "Opponent Steals per Game": "opponent-steals-per-game",
    "Opponent Block %": "opponent-block-pct",
    "Opponent Steals per Possession": "opponent-steals-perpossession",
    "Opponent Steals per Defensive Play": "opponent-steal-pct",
    "Opponent Assists per Game": "opponent-assists-per-game",
    "Opponent Turnovers per Game": "opponent-turnovers-per-game",
    "Opponent Assist / Turnover Ratio": "opponent-assist--per--turnover-ratio",
    "Opponent Assists per FGM": "opponent-assists-per-fgm",
    "Opponent Assists per Possession": "opponent-assists-per-possession",
    "Opponent Turnovers per Possession": "opponent-turnovers-per-possession",
    "Opponent Turnovers per Offensive Play": "opponent-turnover-pct",
    "Opponent Personal Fouls per Game": "opponent-personal-fouls-per-game",
    "Opponent Technical Fouls per Game": "opponent-technical-fouls-per-game",
    "Opponent Personal Fouls per Possession": "opponent-personal-fouls-per-possession",
    "Opponent Personal Fouls per Defensive Play": "opponent-personal-foul-pct",
    "Games Played": "games-played",
    "Possessions per Game": "possessions-per-game",
    "Extra Scoring Chances per Game": "extra-chances-per-game",
    "Effective Possession Ratio": "effective-possession-ratio",
    "Opponent Effective Possession Ratio": "opponent-effective-possession-ratio",
    "Points + Rebounds + Assists per Game": "points-plus-rebounds-plus-assists-per-game",
    "Points + Rebounds per Game": "points-plus-rebounds-per-game",
    "Points + Assists per Game": "points-plus-assists-per-game",
    "Rebounds + Assists per Game": "rebounds-plus-assists-per-game",
    "Steals + Blocks per Game": "steals-plus-blocks-per-game",
    "Opponent Points + Rebounds + Assists per Game": "opponent-points-plus-rebounds-plus-assists-per-gam",
    "Opponent Points + Rebounds per Game": "opponent-points-plus-rebounds-per-game",
    "Opponent Points + Assists per Game": "opponent-points-plus-assists-per-game",
    "Opponent Rebounds + Assists per Game": "opponent-rebounds-plus-assists-per-game",
    "Opponent Steals + Blocks per Game": "opponent-steals-plus-blocks-per-game",
    "Win % - All Games": "win-pct-all-games",
    "Win % - Close Games": "win-pct-close-games",
    "Opponent Win % - All Games": "opponent-win-pct-all-games",
    "Opponent Win % - Close Games": "opponent-win-pct-close-games",
}

MASTER_FILE_MAP = {
    "NBA Player Stats Away:Road.csv": ("nba_api", "player_stats_road"),
    "NBA Player Stats Home.csv": ("nba_api", "player_stats_home"),
    "NBA Player Stats Last 5 Games.csv": ("nba_api", "player_stats_last_5"),
    "NBA Player Individual Defense.csv": ("nba_api", "player_defense"),
    "NBA Player Opponent Stats Against Them.csv": ("nba_api", "player_opponent_stats"),
    "NBA Player Per Game Averages.csv": ("bball_ref", ("https://www.basketball-reference.com/leagues/NBA_{YEAR}_per_game.html", "per_game_stats")),
    "NBA Player Play-by-Play Stats.csv": ("bball_ref", ("https://www.basketball-reference.com/leagues/NBA_{YEAR}_play-by-play.html", "pbp_stats")),
    "NBA Player Advanced Stats.csv": ("bball_ref", ("https://www.basketball-reference.com/leagues/NBA_{YEAR}_advanced.html", "advanced")),
    "NBA Player Shooting Stats.csv": ("bball_ref", ("https://www.basketball-reference.com/leagues/NBA_{YEAR}_shooting.html", "shooting")),
    "NBA Player Adjusted Shooting Stats.csv": ("bball_ref", ("https://www.basketball-reference.com/leagues/NBA_{YEAR}_adj_shooting.html", "adj_shooting")),
    "NBA Team Defense.csv": ("nba_api", "team_defense_summary"),
    "NBA Team General Stats.csv": ("nba_api", "team_general_summary"),
}

def create_robust_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=2, 
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(HEADERS)
    return session

def save_clean_parquet(df, filename_stem, output_dir):
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        clean_name = filename_stem.replace('.csv', '') + ".parquet"
        file_path = output_dir / clean_name
        
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str)
            
        df.to_parquet(file_path, index=False)
        logging.info(f"Successfully saved {file_path}")
    except Exception as e:
        logging.error(f"FAILED to save {filename_stem}: {e}")

def deduplicate_columns(df):
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique(): 
        cols[cols[cols == dup].index.values.tolist()] = [
            dup if i == 0 else f"{dup}_{i}" 
            for i in range(sum(cols == dup))
        ]
    df.columns = cols
    return df

def scrape_daily_injuries(session, output_dir):
    logging.info("--- Scraping Daily Injury Report (CBS Sports) ---")
    url = "https://www.cbssports.com/nba/injuries/"
    filename = "daily_injuries" 
    
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        tables = soup.find_all('table')
        
        if not tables:
            logging.warning("No injury tables found on CBS Sports.")
            return

        all_rows = []
        
        for table in tables:
            team_abbr = "UNK"
            team_name = "UNKNOWN"
            
            team_header = table.find_previous(class_="TeamLogoNameLockup-name")
            if team_header:
                raw_team_name = team_header.get_text(strip=True)
                team_name = raw_team_name
                team_abbr = TEAM_NAME_MAP.get(raw_team_name, "UNK")
            
            rows = table.find_all('tr')
            
            for row in rows:
                cols = row.find_all('td')
                if not cols:
                    continue 
                
                name_cell = cols[0]
                long_name_span = name_cell.find('span', class_=lambda x: x and 'long' in x)
                
                if long_name_span:
                    player_text = long_name_span.get_text(strip=True)
                else:
                    a_tag = name_cell.find('a')
                    if a_tag:
                        player_text = a_tag.get_text(strip=True)
                    else:
                        player_text = name_cell.get_text(strip=True)

                try:
                    pos = cols[1].get_text(strip=True)
                    updated = cols[2].get_text(strip=True)
                    injury = cols[3].get_text(strip=True)
                    status = cols[4].get_text(strip=True)
                    
                    all_rows.append({
                        "Team": team_abbr,
                        "Team_Full": team_name,
                        "Player": player_text,
                        "Position": pos,
                        "Updated": updated,
                        "Injury": injury,
                        "Injury Status": status
                    })
                except IndexError:
                    continue

        if not all_rows:
            logging.warning("Parsed 0 injury rows. Layout may have changed.")
            return

        injury_df = pd.DataFrame(all_rows)
        
        if 'Injury Status' in injury_df.columns:
            def clean_status(val):
                s = str(val).lower()
                if 'out' in s: return 'OUT'
                if 'doubtful' in s: return 'DOUBTFUL'
                if 'questionable' in s: return 'GTD'
                if 'game time decision' in s: return 'GTD'
                return 'UNKNOWN'
            
            injury_df['Status_Clean'] = injury_df['Injury Status'].apply(clean_status)
        
        save_clean_parquet(injury_df, filename, output_dir)
        logging.info(f"Scraped {len(injury_df)} injury records.")
        
    except Exception as e:
        logging.error(f"Failed to scrape injuries: {e}", exc_info=True)

def scrape_teamrankings(session, slug, filename, season_cfg, output_dir):
    url = f"https://www.teamrankings.com/nba/stat/{slug}"
    if season_cfg['tr_date_param']:
        url += f"?date={season_cfg['tr_date_param']}"
        
    logging.info(f"Fetching [TeamRankings] {filename} for {season_cfg['season_str']}...")

    try:
        response = session.get(url, timeout=30)
        response.raise_for_status() 
        
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table') 
        
        if not table:
            logging.warning(f"No <table> found at {url}")
            return

        dfs = pd.read_html(io.StringIO(str(table)))
        if not dfs:
            return
            
        df = dfs[0]
        
        if isinstance(df.columns, pd.MultiIndex):
             df.columns = [col[1] if len(col) > 1 else col[0] for col in df.columns]
        else:
            df.columns = [str(col) for col in df.columns]

        df = deduplicate_columns(df)

        if len(df.columns) >= 8:
            cols_to_keep = [0, 1, 2, 3, 4, 5, 6] 
            df = df.iloc[:, cols_to_keep].copy()
            season_year = season_cfg['bball_ref_year']
            df.columns = ["Rank", "Team", str(season_year), "Last 3", "Last 1", "Home", "Away"]
        else:
            df = df.copy() 
            df.columns = [str(c) for c in df.columns]

        if 'Team' in df.columns:
            df['Team'] = df['Team'].apply(lambda x: str(x).split('(')[0].strip())
        
        save_clean_parquet(df, filename, output_dir)
        
    except Exception as e:
        logging.error(f"Failed to scrape {url}: {e}")
    finally:
        time.sleep(1.0 + random.random()) 

def scrape_bball_ref(session, url_template, table_id, filename, season_cfg, output_dir):
    url = url_template.replace("{YEAR}", str(season_cfg['bball_ref_year']))
    logging.info(f"Fetching [BBall-Ref] {filename} for {season_cfg['season_str']}...")
    
    try:
        response = session.get(url, timeout=45)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', id=table_id)
        
        if not table:
            comment = soup.find(string=lambda text: isinstance(text, Comment) and table_id in text)
            if comment:
                try:
                    comment_soup = BeautifulSoup(comment, 'html.parser')
                    table = comment_soup.find('table', id=table_id)
                except: table = None 
        
        if not table:
            logging.warning(f"Could not find table '{table_id}' at {url}.")
            return
            
        df = pd.read_html(io.StringIO(str(table)))[0]
        
        if isinstance(df.columns, pd.MultiIndex):
            new_cols = []
            for col in df.columns:
                c_name = col[1] if len(col) > 1 else col[0]
                new_cols.append(str(c_name))
            df.columns = new_cols
        else:
            df.columns = [str(col) for col in df.columns]

        df = deduplicate_columns(df)

        if 'Rk' in df.columns:
            df = df[df['Rk'] != 'Rk']
        
        if 'Awards' in df.columns:
            df = df.drop(columns=['Awards'])
            
        save_clean_parquet(df, filename, output_dir)
        
    except Exception as e:
        logging.error(f"Failed to scrape {url}: {e}", exc_info=True)
    finally:
        time.sleep(3) 

def fetch_and_save_parquet(filename, api_class, output_dir, **kwargs):
    retries = 3
    for attempt in range(retries):
        try:
            data = api_class(timeout=NBA_API_TIMEOUT, **kwargs)
            save_clean_parquet(data.get_data_frames()[0], filename, output_dir)
            return
        except Exception as e:
            if attempt < retries - 1:
                wait_time = (attempt + 1) * 5
                time.sleep(wait_time)
            else:
                logging.error(f"Failed to fetch {filename} after {retries} attempts: {e}")

def get_season_dates(season_str):
    """
    Returns start_date, end_date for a given season ID (e.g. '2024-25').
    """
    start_year = int(season_str.split('-')[0])
    
    # NBA usually starts mid-Oct and ends mid-April
    start_date = datetime(start_year, 10, 22) # Safe approx
    end_date = datetime(start_year + 1, 4, 15)
    
    # Cap end date at today (inclusive)
    if end_date > datetime.now():
        end_date = datetime.now()
        
    return start_date, end_date

def fetch_daily_player_stats(target_date, timeout=120):
    """
    Fetches stats for ALL players on a specific date in ONE call.
    Includes explicit retry logic for READ TIMEOUTS and Stagger Delay.
    """
    # STAGGER DELAY: Prevents 2 requests from hitting API at exact same ms
    time.sleep(random.uniform(0.5, 1.5))
    
    date_str = target_date.strftime('%m/%d/%Y')
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # LeagueDashPlayerStats returns rows for everyone who played in the window
            logs = LeagueDashPlayerStats(
                date_from_nullable=date_str,
                date_to_nullable=date_str,
                season_type_all_star='Regular Season',
                timeout=timeout
            )
            df = logs.get_data_frames()[0]
            if not df.empty:
                df['GAME_DATE'] = target_date.strftime('%Y-%m-%d')
            return df
            
        except Exception as e:
            logging.warning(f"Attempt {attempt+1}/{max_retries} failed for {date_str}: {e}")
            if attempt < max_retries - 1:
                time.sleep(3 * (attempt + 1)) # Backoff: 3s, 6s...
            else:
                logging.error(f"Final failure for {date_str}")
                return pd.DataFrame()

def scrape_nba_api_stats(season_cfg, output_dir):
    target_season = season_cfg['season_str']
    logging.info(f"--- Fetching all nba-api data (Season: {target_season}) ---")
    
    try:
        # --- BATCH FETCHING FOR BOX SCORES ---
        logging.info("Fetching Player Box Scores (Batch by Date)...")
        
        start_date, end_date = get_season_dates(target_season)
        current_date = start_date
        
        all_daily_stats = []
        dates_to_fetch = []
        
        while current_date <= end_date:
            dates_to_fetch.append(current_date)
            current_date += timedelta(days=1)
            
        logging.info(f"Queued {len(dates_to_fetch)} days to fetch.")
        
        # Parallel Fetching by Date
        # Uses tuned settings: MAX_WORKERS=2, TIMEOUT=120
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_date = {executor.submit(fetch_daily_player_stats, d, NBA_API_TIMEOUT): d for d in dates_to_fetch}
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_date):
                date_val = future_to_date[future]
                try:
                    df = future.result()
                    if not df.empty:
                        all_daily_stats.append(df)
                except Exception as e:
                    logging.error(f"Error fetching {date_val}: {e}")
                
                completed += 1
                if completed % 5 == 0:
                    logging.info(f"  Fetches complete: {completed}/{len(dates_to_fetch)}")

        if not all_daily_stats:
            box_scores_df = pd.DataFrame()
        else:
            box_scores_df = pd.concat(all_daily_stats, ignore_index=True)
            
        save_clean_parquet(box_scores_df, "NBA Player Box Scores", output_dir)
        logging.info(f"Saved aggregated box scores: {len(box_scores_df)} rows.")

        # --- REMAINING GENERAL STATS ---
        logging.info("Fetching remaining Player and Team Stats (Parallel)...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            
            futures.append(executor.submit(fetch_and_save_parquet, "NBA Player Stats Home", LeagueDashPlayerStats, output_dir,
                season=target_season, location_nullable="Home"))
            futures.append(executor.submit(fetch_and_save_parquet, "NBA Player Stats Away:Road", LeagueDashPlayerStats, output_dir,
                season=target_season, location_nullable="Road"))
            futures.append(executor.submit(fetch_and_save_parquet, "NBA Player Stats Last 5 Games", LeagueDashPlayerStats, output_dir,
                season=target_season, last_n_games=5))
            futures.append(executor.submit(fetch_and_save_parquet, "NBA Player Individual Defense", LeagueDashPtDefend, output_dir,
                season=target_season))
            futures.append(executor.submit(fetch_and_save_parquet, "NBA Player Opponent Stats Against Them", LeagueDashOppPtShot, output_dir,
                season=target_season))
            futures.append(executor.submit(fetch_and_save_parquet, "NBA Team General Stats", LeagueDashTeamStats, output_dir,
                season=target_season, measure_type_detailed_defense="Base"))
            futures.append(executor.submit(fetch_and_save_parquet, "NBA Team Defense", LeagueDashTeamStats, output_dir,
                season=target_season, measure_type_detailed_defense="Opponent"))

            # Quarter Stats (Aggregates)
            for q in range(1, 5):
                futures.append(executor.submit(fetch_and_save_parquet, f"NBA Player Q{q}", LeagueDashPlayerStats, output_dir,
                    season=target_season, period=q))

            for future in concurrent.futures.as_completed(futures):
                future.result() 

        # --- Q1/Q2 RECENT GRADING FETCH ---
        if season_cfg['is_current']:
            scrape_recent_quarter_stats(output_dir)

        logging.info("--- All nba-api data fetched successfully ---")

    except Exception as e:
        logging.error(f"CRITICAL FAILURE in nba-api section: {e}", exc_info=True)

def scrape_recent_quarter_stats(output_dir):
    """
    Fetches Q1 AND Q2 stats for TODAY to allow immediate post-game grading.
    """
    # Changed from Yesterday to Today as requested
    target_dt = datetime.now()
    target_str = target_dt.strftime('%m/%d/%Y') 
    save_str = target_dt.strftime('%Y-%m-%d')
    
    logging.info(f"--- Fetching Q1 & Q2 Box Scores for Grading ({target_str}) ---")
    
    for period in [1, 2]:
        try:
            stats = LeagueDashPlayerStats(
                period=period,
                date_from_nullable=target_str,
                date_to_nullable=target_str,
                season_type_all_star='Regular Season',
                timeout=NBA_API_TIMEOUT
            )
            df = stats.get_data_frames()[0]
            
            if not df.empty:
                df['GAME_DATE'] = save_str
                q_dir = output_dir / f"q{period}_logs"
                q_dir.mkdir(parents=True, exist_ok=True)
                filename = f"daily_q{period}_stats_{save_str}"
                
                save_clean_parquet(df, filename, q_dir)
                logging.info(f"Saved Q{period} stats for {save_str}")
            else:
                logging.info(f"No Q{period} stats found for {target_str} (Games might not be finished)")
                
            time.sleep(1.0 + random.random()) 
            
        except Exception as e:
            logging.error(f"Failed to scrape Q{period} stats: {e}")

def should_skip_season_file(output_dir, filename_stem, is_current_season):
    if is_current_season:
        return False 
    
    clean_name = filename_stem.replace('.csv', '') + ".parquet"
    file_path = output_dir / clean_name
    
    if file_path.exists() and file_path.stat().st_size > 0:
        return True 
    
    return False

def main():
    start_time = time.time()
    
    logging.info("========= STARTING NBA DATA SCRAPER (BATCH OPTIMIZED) =========")
    
    session = create_robust_session()
    seasons_to_scrape = get_season_config()
    
    for season_cfg in seasons_to_scrape:
        season_str = season_cfg['season_str']
        is_current = season_cfg['is_current']
        
        output_dir = cfg.DATA_DIR / season_str
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"--- Processing Season: {season_str} (Live: {is_current}) ---")
        
        if is_current:
            scrape_daily_injuries(session, output_dir)
            
        logging.info("--- Checking Basketball-Reference Files ---")
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for filename, (source, data) in MASTER_FILE_MAP.items():
                if source == 'bball_ref':
                    if should_skip_season_file(output_dir, filename, is_current):
                        logging.info(f"Skipping cached file: {filename}")
                        continue
                        
                    url_template, table_id = data
                    futures.append(executor.submit(scrape_bball_ref, session, url_template, table_id, filename, season_cfg, output_dir))
            
            for future in concurrent.futures.as_completed(futures):
                future.result() 
                
        # Always run NBA API scrape for current season to get fresh box scores
        if not is_current and should_skip_season_file(output_dir, "NBA Team General Stats", is_current):
            logging.info("Skipping NBA API stats (Cached)")
        else:
            scrape_nba_api_stats(season_cfg, output_dir)
        
        logging.info("--- Checking TeamRankings Files ---")
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for friendly_name, slug in TEAMRANKINGS_SLUG_MAP.items():
                sanitized_name = re.sub(r"\(.*\)", "", friendly_name).strip()
                sanitized_name = sanitized_name.replace(" / ", " per ").replace("/", " per ")
                filename = f"NBA Team {sanitized_name}" 
                
                if should_skip_season_file(output_dir, filename, is_current):
                    continue
                
                futures.append(executor.submit(scrape_teamrankings, session, slug, filename, season_cfg, output_dir))
                
            for future in concurrent.futures.as_completed(futures):
                future.result()
            
    session.close()
    
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    
    logging.info(f"========= NBA DATA SCRAPER FINISHED in {minutes}:{seconds:02d} minutes =========")

if __name__ == "__main__":
    main()