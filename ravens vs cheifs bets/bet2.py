import pandas as pd
import numpy as np
from joblib import Parallel, delayed

# Load the rosters for Ravens and Chiefs
ravens = pd.read_csv('C:/Users/Chuks/Documents/ravens vs cheifs bets/ravens roster ratings.csv')
chiefs = pd.read_csv('C:/Users/Chuks/Documents/ravens vs cheifs bets/chiefs roster ratings.csv')

# Normalize the ratings to a 0-1 scale and ensure missing values are handled
ravens['OVR'] = pd.to_numeric(ravens['OVR'], errors='coerce').fillna(0) / 100
chiefs['OVR'] = pd.to_numeric(chiefs['OVR'], errors='coerce').fillna(0) / 100

# Define position weighting
position_weights = {
    'QB': 0.3,
    'RB': 0.1,
    'WR': 0.15,
    'DE': 0.1,
    'DT': 0.1,
    'CB': 0.08,
    'S': 0.05,
    'LT': 0.1,
    'LG': 0.1,
    'RT': 0.1,
    'RG': 0.1
}

# Function to calculate weighted position ratings and extract all positions' ratings
def extract_team_features(df, position_weights):
    features = {}
    
    if 'POSITION' not in df.columns or 'OVR' not in df.columns:
        print("Error: Missing 'POSITION' or 'OVR' column in data.")
        return features
    
    unique_positions = df['POSITION'].unique()

    for pos in unique_positions:
        features[f'{pos}_rating'] = df[df['POSITION'] == pos]['OVR'].mean()
    
    weighted_rating = 0
    for pos, weight in position_weights.items():
        if pos in unique_positions:
            weighted_rating += df[df['POSITION'] == pos]['OVR'].mean() * weight
    
    features['weighted_rating'] = weighted_rating if not np.isnan(weighted_rating) else 0
    features['avg_overall_rating'] = df['OVR'].mean() if not np.isnan(df['OVR'].mean()) else 0
    
    return features

# Extract features for Ravens and Chiefs
ravens_features = extract_team_features(ravens, position_weights)
chiefs_features = extract_team_features(chiefs, position_weights)

# Load the betting odds data
odds_df = pd.read_csv('C:/Users/Chuks/Documents/ravens vs cheifs bets/ravens_chiefs_odds.csv')

# Function to extract just the spread number from the "Best Odds" column
def extract_spread(spread_str):
    try:
        spread_value = spread_str.split()[0]
        return float(spread_value)
    except ValueError:
        print(f"Error extracting spread: {spread_str}")
        return 0  # Default to 0 if there's an error

# Extract the spread for Ravens and Chiefs
ravens_spread_str = odds_df.loc[odds_df['Market'] == 'Ravens Spread', 'Best Odds'].values[0]
chiefs_spread_str = odds_df.loc[odds_df['Market'] == 'Chiefs Spread', 'Best Odds'].values[0]

# Convert the spread strings to floats
ravens_spread = extract_spread(ravens_spread_str)
chiefs_spread = extract_spread(chiefs_spread_str)

# Load and compute head-to-head metrics, ensuring no missing values
rushing_stats = pd.read_csv('C:/Users/Chuks/Documents/ravens vs cheifs bets/ravens vs chiefs rushing.csv').fillna(0)
passing_stats = pd.read_csv('C:/Users/Chuks/Documents/ravens vs cheifs bets/ravens vs chiefs passing.csv').fillna(0)
defense_stats = pd.read_csv('C:/Users/Chuks/Documents/ravens vs cheifs bets/ravens vs chiefs defense.csv').fillna(0)

# Calculate key metrics for Ravens and Chiefs with safety checks
def calculate_team_metrics(stats_df, team_code, metric_name, column_name):
    if not stats_df[stats_df['Team'] == team_code].empty:
        return stats_df.loc[stats_df['Team'] == team_code, column_name].sum()
    else:
        print(f"Warning: No data found for {team_code} in {metric_name}. Defaulting to 0.")
        return 0

ravens_metrics = {
    'rushing_yards': calculate_team_metrics(rushing_stats, 'BAL', 'rushing yards', 'Yds'),
    'passing_yards': calculate_team_metrics(passing_stats, 'BAL', 'passing yards', 'Yds'),
    'sacks': calculate_team_metrics(defense_stats, 'BAL', 'sacks', 'Sk')
}

chiefs_metrics = {
    'rushing_yards': calculate_team_metrics(rushing_stats, 'KAN', 'rushing yards', 'Yds'),
    'passing_yards': calculate_team_metrics(passing_stats, 'KAN', 'passing yards', 'Yds'),
    'sacks': calculate_team_metrics(defense_stats, 'KAN', 'sacks', 'Sk')
}

# Function to calculate Expected Value (EV) of a bet
def calculate_ev(probability, odds_decimal, bet_amount=100):
    ev = (probability * (odds_decimal * bet_amount - bet_amount)) - ((1 - probability) * bet_amount)
    return ev

# Function to convert American odds to decimal odds
def american_to_decimal(odds):
    if odds > 0:  # Positive American odds
        return 1 + (odds / 100)
    else:  # Negative American odds
        return 1 + (100 / abs(odds))

# Extract betting odds for Ravens and Chiefs
ravens_moneyline = odds_df.loc[odds_df['Market'] == 'Ravens Moneyline', 'Best Odds'].values[0]
chiefs_moneyline = odds_df.loc[odds_df['Market'] == 'Chiefs Moneyline', 'Best Odds'].values[0]

# Convert odds
ravens_odds_decimal = american_to_decimal(float(ravens_moneyline))
chiefs_odds_decimal = american_to_decimal(float(chiefs_moneyline))

# Function to calculate the spread from simulated scores
def calculate_spread_from_simulation(avg_ravens_score, avg_chiefs_score):
    simulated_spread = avg_ravens_score - avg_chiefs_score
    #print(f"Simulated Spread: Ravens by {simulated_spread:.2f} points.")
    return simulated_spread

# Function to simulate outcomes based on team ratings and track scores
def simulate_game_outcomes_with_scores(team1_features, team2_features, team1_metrics, team2_metrics, num_simulations=100):
    variance_factor = 9.0  # Increased variance for more randomness
    home_field_advantage = 0.05  # Applied to team1 (Chiefs)
    
    # Increase the influence of metrics by reducing the divisor
    team1_rushing_factor = team1_metrics.get('rushing_yards', 0) / 500  # More influence
    team1_passing_factor = team1_metrics.get('passing_yards', 0) / 500
    team1_defense_factor = team1_metrics.get('sacks', 0) / 3

    team2_rushing_factor = team2_metrics.get('rushing_yards', 0) / 500
    team2_passing_factor = team2_metrics.get('passing_yards', 0) / 500
    team2_defense_factor = team2_metrics.get('sacks', 0) / 3
    
    team1_scores = []
    team2_scores = []

    # Get average overall ratings for both teams
    team1_avg_overall_rating = team1_features.get('avg_overall_rating', 0)
    team2_avg_overall_rating = team2_features.get('avg_overall_rating', 0)

    for _ in range(num_simulations):
        team1_weighted_rating = team1_features.get('weighted_rating', 0)
        team2_weighted_rating = team2_features.get('weighted_rating', 0)

        if np.isnan(team1_weighted_rating):
            team1_weighted_rating = team1_avg_overall_rating
        if np.isnan(team2_weighted_rating):
            team2_weighted_rating = team2_avg_overall_rating

        # Constrain the randomness to avoid extreme outcomes
        random_factor = np.clip(variance_factor * np.random.randn(), -0.5, 3)

        # Calculate the team scores with the adjusted metrics and randomness
        team1_score = max(0, team1_weighted_rating * (1 + random_factor) * (1 + team1_rushing_factor) * (1 + team1_passing_factor) * (1 + team1_defense_factor))
        team2_score = max(0, team2_weighted_rating * (1 + random_factor) * (1 + team2_rushing_factor) * (1 + team2_passing_factor) * (1 + team2_defense_factor) * (1 + home_field_advantage))

        # Round scores to nearest whole number immediately after calculation
        team1_score = round(team1_score)
        team2_score = round(team2_score)

        # Store the rounded scores in the game results
        team1_scores.append(team1_score)
        team2_scores.append(team2_score)

    # Calculate average scores
    avg_team1_score = round(np.nanmean(team1_scores)) if team1_scores else 0
    avg_team2_score = round(np.nanmean(team2_scores)) if team2_scores else 0
    
    return avg_team1_score, avg_team2_score

# Number of iterations to run the simulation and EV calculation
num_iterations = 1000

# Lists to store EVs and spreads for each iteration
ravens_ev_list = []
chiefs_ev_list = []
spread_list = []
ravens_avg_scores_list = []
chiefs_avg_scores_list = []

# Run the simulation and EV calculation multiple times
for _ in range(num_iterations):
    # Run the simulation for each iteration
    avg_ravens_score, avg_chiefs_score = simulate_game_outcomes_with_scores(
        ravens_features, chiefs_features, ravens_metrics, chiefs_metrics, num_simulations=100)

    # Calculate probabilities from simulations
    ravens_wins = 1 if avg_ravens_score > avg_chiefs_score else 0
    chiefs_wins = 1 if avg_chiefs_score > avg_ravens_score else 0

    ravens_prob = ravens_wins / 1
    chiefs_prob = chiefs_wins / 1

    # Calculate the spread based on the simulated scores
    simulated_spread = calculate_spread_from_simulation(avg_ravens_score, avg_chiefs_score)

    # Calculate the EV for betting on Ravens and Chiefs
    ravens_ev = calculate_ev(ravens_prob, ravens_odds_decimal)
    chiefs_ev = calculate_ev(chiefs_prob, chiefs_odds_decimal)

    # Store the EV and spread for this iteration
    ravens_ev_list.append(ravens_ev)
    chiefs_ev_list.append(chiefs_ev)
    spread_list.append(simulated_spread)
    ravens_avg_scores_list.append(avg_ravens_score)
    chiefs_avg_scores_list.append(avg_chiefs_score)

# Calculate the average EV and point spread after all iterations
avg_ravens_ev = np.mean(ravens_ev_list)
avg_chiefs_ev = np.mean(chiefs_ev_list)
avg_spread = np.mean(spread_list)
avg_ravens_score_final = np.mean(ravens_avg_scores_list)
avg_chiefs_score_final = np.mean(chiefs_avg_scores_list)

# Print the average EVs, point spreads, and scores after all iterations
print(f"\nAverage Expected Value for betting on Ravens after {num_iterations} runs: ${avg_ravens_ev:.2f}")
print(f"Average Expected Value for betting on Chiefs after {num_iterations} runs: ${avg_chiefs_ev:.2f}")
print(f"Average Simulated Spread after {num_iterations} runs: Ravens by {avg_spread:.2f} points.")
print(f"Average Score after {num_iterations} runs: Ravens {avg_ravens_score_final:.2f} - Chiefs {avg_chiefs_score_final:.2f}")

# Response based on the average EV
if avg_ravens_ev > 0:
    print("Betting on Ravens offers a positive average expected value over multiple simulations.")
else:
    print("Betting on Ravens offers a negative average expected value over multiple simulations.")

if avg_chiefs_ev > 0:
    print("Betting on Chiefs offers a positive average expected value over multiple simulations.")
else:
    print("Betting on Chiefs offers a negative average expected value over multiple simulations.")

