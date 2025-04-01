import pandas as pd
import os
from mapping import team_name_to_id

def preprocess_data(input_path, output_path):
    """
    Loads raw NFL game data, cleans it, and saves the preprocessed dataset.
    """
    df = pd.read_csv(input_path)
    
    # Drop all rows from 1978 and prior
    df = df[df['schedule_season'] >= 1979].reset_index(drop=True)
    # Drop irrelevant columns
    df.drop(columns=['schedule_date', 'stadium', 'weather_temperature', 'weather_wind_mph', 'weather_humidity', 'weather_detail'], inplace=True, errors='ignore')
    # Drop rows that are missing key data
    df = df.dropna(subset=['team_favorite_id', 'spread_favorite', 'over_under_line'])

    # Map full team names to abbreviations
    df['team_home_abbrev'] = df['team_home'].map(team_name_to_id)
    df['team_away_abbrev'] = df['team_away'].map(team_name_to_id)

    # Drop the original full-name columns to clean up the dataset
    df.drop(['team_home', 'team_away'], axis=1, inplace=True)

    # Create column to track whether the home team is favored
    df['home_team_favored'] = (df['team_home_abbrev'] == df['team_favorite_id']).astype(int)

    # # Create binary column for playoff status
    # df['is_playoff_game'] = df['schedule_playoff'].astype(int)
    # # Drop the original schedule_playoff column
    # df.drop(columns=['schedule_playoff'], inplace=True)

    df['point_diff'] = df['score_home'] - df['score_away']
    df['home_team_won'] = (df['point_diff'] > 0).astype(int)
    

    # Convert over_under_line to numeric (handle non-numeric cases)
    df['over_under_line'] = pd.to_numeric(df['over_under_line'], errors='coerce')
    
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save cleaned dataset
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")
    
if __name__ == "__main__":
    input_file = "ml-model/data/spreadspoke_scores.csv"
    output_file = "ml-model/data/cleaned_nfl_data.csv"
    preprocess_data(input_file, output_file)


# FIRST DRAFT -- FIX AND FINALIZE
def add_week_index_column(df):
    
    # Convert week to numeric for regular season
    df['week_num'] = pd.to_numeric(df['schedule_week'], errors='coerce')
    
    # Mapping for playoff rounds
    playoff_mapping = {
        'Wildcard': 1,
        'Divisional': 2,
        'Division': 2,
        'Conference Championship': 3,
        'Super Bowl': 4,
        'Superbowl': 4
    }

    # Get max week per season from regular season games
    regular_week_max = df[~df['schedule_playoff']].groupby('schedule_season')['week_num'].max()

    # Function to determine full week index
    def compute_week_index(row):
        if not row['schedule_playoff']:
            return row['week_num']
        else:
            max_week = regular_week_max.get(row['schedule_season'], 17)
            offset = playoff_mapping.get(row['schedule_week'], 1)
            return max_week + offset

    df['week_index'] = df.apply(compute_week_index, axis=1)
    return df
