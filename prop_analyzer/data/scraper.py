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
from datetime import datetime
from bs4 import BeautifulSoup, Comment
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from nba_api.stats.static import players

# Import project config
from prop_analyzer import config as cfg

try:
    from nba_api.stats.endpoints.playergamelog import PlayerGameLog
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

MAX_WORKERS = 5 # Reduced threads to be nicer to servers
NBA_API_TIMEOUT = 60 # Increased timeout for slow responses

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Referer': 'https://www.nba.com/',
    'Origin': 'https://www.nba.com/',
    'Accept-Language': 'en-US,en;q=0.9',
    'Connection': 'keep-alive'
}

# --- MAPPINGS (Preserved) ---
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
    "NBA Player Box Scores.csv": ("nba_api", "player_box_scores"),
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
    """
    Creates a requests session with automatic retries and backoff.
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=2, # Slower backoff (1s, 2s, 4s, 8s, 16s)
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(HEADERS)
    return session

def save_clean_csv(df, filename, output_dir):
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / filename
        df.to_csv(file_path, index=False, encoding='utf-8')
        logging.info(f"Successfully saved clean {file_path}")
    except Exception as e:
        logging.error(f"FAILED to save {filename}: {e}")

def scrape_daily_injuries(session, output_dir):
    logging.info("--- Scraping Daily Injury Report (CBS Sports) ---")
    url = "https://www.cbssports.com/nba/injuries/"
    filename = "daily_injuries.csv"
    
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
        
        save_clean_csv(injury_df, filename, output_dir)
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

        if len(df.columns) >= 8:
            cols_to_keep = [0, 1, 2, 3, 4, 5, 6] # Rank, Team, [SeasonStat], Last 3, Last 1, Home, Away
            df = df.iloc[:, cols_to_keep].copy()
            season_year = season_cfg['bball_ref_year']
            df.columns = ["Rank", "Team", str(season_year), "Last 3", "Last 1", "Home", "Away"]
        else:
            df = df.copy() 
            df.columns = [str(c) for c in df.columns]

        if 'Team' in df.columns:
            df['Team'] = df['Team'].apply(lambda x: str(x).split('(')[0].strip())
        
        save_clean_csv(df, filename, output_dir)
        
    except Exception as e:
        logging.error(f"Failed to scrape {url}: {e}")
    finally:
        time.sleep(1.0 + random.random()) # Random 1-2s delay

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
            df.columns = [col[1] if len(col) > 1 else col[0] for col in df.columns]
        else:
            df.columns = [str(col) for col in df.columns]

        if 'Rk' in df.columns:
            df = df[df['Rk'] != 'Rk']
        
        if 'Awards' in df.columns:
            df = df.drop(columns=['Awards'])
            
        save_clean_csv(df, filename, output_dir)
        
    except Exception as e:
        logging.error(f"Failed to scrape {url}: {e}", exc_info=True)
    finally:
        time.sleep(3) # Politeness

def fetch_and_save(filename, api_class, output_dir, **kwargs):
    # Retry wrapper for Generic API calls
    retries = 3
    for attempt in range(retries):
        try:
            data = api_class(timeout=NBA_API_TIMEOUT, **kwargs)
            save_clean_csv(data.get_data_frames()[0], filename, output_dir)
            return
        except Exception as e:
            if attempt < retries - 1:
                wait_time = (attempt + 1) * 5
                logging.warning(f"Timeout for {filename}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logging.error(f"Failed to fetch {filename} after {retries} attempts: {e}")

def scrape_nba_api_stats(season_cfg, output_dir):
    target_season = season_cfg['season_str']
    logging.info(f"--- Fetching all nba-api data (Season: {target_season}) ---")
    
    try:
        logging.info("Fetching Player Box Scores...")
        all_players = players.get_players()
        
        active_players = [p for p in all_players if p['is_active']]
        total_active = len(active_players)
        
        logging.info(f"Found {total_active} active players. Starting sequential scrape...")
        
        all_logs = []

        for i, player in enumerate(active_players):
            if (i + 1) % 10 == 0:
                logging.info(f"  Scraped {i + 1}/{total_active} player box scores ({player['full_name']})...")
            
            # --- RETRY LOGIC FOR INDIVIDUAL PLAYERS ---
            max_retries = 3
            success = False
            
            for attempt in range(max_retries):
                try:
                    # Random sleep before request to act human
                    time.sleep(random.uniform(0.6, 1.2))
                    
                    log = PlayerGameLog(
                        player_id=player['id'],
                        season=target_season,
                        season_type_all_star="Regular Season",
                        timeout=NBA_API_TIMEOUT
                    )
                    df = log.get_data_frames()[0]
                    if not df.empty:
                        all_logs.append(df)
                    success = True
                    break # Success, exit retry loop
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        # Exponential backoff: 5s, 10s...
                        wait = (attempt + 1) * 5
                        # logging.warning(f"Timeout for {player['full_name']}. Retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        logging.warning(f"Error scraping {player['full_name']} after retries: {e}")
            
            # Safety: If we failed 3 times, maybe pause longer for the *next* player
            if not success:
                time.sleep(10)

        if not all_logs:
            box_scores_df = pd.DataFrame() 
        else:
            box_scores_df = pd.concat(all_logs, ignore_index=True)
            
        save_clean_csv(box_scores_df, "NBA Player Box Scores.csv", output_dir)
        
        logging.info("Fetching remaining Player and Team Stats (parallel)...")
        
        # Parallel execution can trigger rate limits too fast. 
        # Reduced max_workers to 5 in constants.
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            
            futures.append(executor.submit(fetch_and_save, "NBA Player Stats Home.csv", LeagueDashPlayerStats, output_dir,
                season=target_season, location_nullable="Home"))
            futures.append(executor.submit(fetch_and_save, "NBA Player Stats Away:Road.csv", LeagueDashPlayerStats, output_dir,
                season=target_season, location_nullable="Road"))
            futures.append(executor.submit(fetch_and_save, "NBA Player Stats Last 5 Games.csv", LeagueDashPlayerStats, output_dir,
                season=target_season, last_n_games=5))
            futures.append(executor.submit(fetch_and_save, "NBA Player Individual Defense.csv", LeagueDashPtDefend, output_dir,
                season=target_season))
            futures.append(executor.submit(fetch_and_save, "NBA Player Opponent Stats Against Them.csv", LeagueDashOppPtShot, output_dir,
                season=target_season))
            futures.append(executor.submit(fetch_and_save, "NBA Team General Stats.csv", LeagueDashTeamStats, output_dir,
                season=target_season, measure_type_detailed_defense="Base"))
            futures.append(executor.submit(fetch_and_save, "NBA Team Defense.csv", LeagueDashTeamStats, output_dir,
                season=target_season, measure_type_detailed_defense="Opponent"))

            # Quarter Stats
            for q in range(1, 5):
                futures.append(executor.submit(fetch_and_save, f"NBA Player Q{q}.csv", LeagueDashPlayerStats, output_dir,
                    season=target_season, period=q))

            for future in concurrent.futures.as_completed(futures):
                future.result() 

        logging.info("--- All nba-api data fetched successfully ---")

    except Exception as e:
        logging.error(f"CRITICAL FAILURE in nba-api section: {e}", exc_info=True)

def should_skip_season_file(output_dir, filename, is_current_season):
    if is_current_season:
        return False 
    
    file_path = output_dir / filename
    if file_path.exists() and file_path.stat().st_size > 0:
        return True 
    
    return False

def main():
    start_time = time.time()
    
    logging.info("========= STARTING NBA DATA SCRAPER (MULTI-SEASON) =========")
    
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
                
        if should_skip_season_file(output_dir, "NBA Team General Stats.csv", is_current):
            logging.info("Skipping NBA API stats (Cached)")
        else:
            scrape_nba_api_stats(season_cfg, output_dir)
        
        logging.info("--- Checking TeamRankings Files ---")
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for friendly_name, slug in TEAMRANKINGS_SLUG_MAP.items():
                sanitized_name = re.sub(r"\(.*\)", "", friendly_name).strip()
                sanitized_name = sanitized_name.replace(" / ", " per ").replace("/", " per ")
                filename = f"NBA Team {sanitized_name}.csv"
                
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