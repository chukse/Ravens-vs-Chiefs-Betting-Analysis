import pandas as pd
import numpy as np
from joblib import Parallel, delayed

# Load the rosters
ravens = pd.read_csv('C:/Users/Chuks/Documents/ravens vs cheifs bets/ravens roster ratings.csv')
chiefs = pd.read_csv('C:/Users/Chuks/Documents/ravens vs cheifs bets/chiefs roster ratings.csv')

# Normalize the ratings to a 0-1 scale
ravens['OVR'] = ravens['OVR'] / 100
chiefs['OVR'] = chiefs['OVR'] / 100

# Load the betting odds data
odds_df = pd.read_csv('C:/Users/Chuks/Documents/ravens vs cheifs bets/ravens_chiefs_odds.csv')

# Extract specific averages for Ravens and Chiefs Moneyline
ravens_moneyline_avg = float(odds_df.loc[odds_df['Market'] == 'Ravens Moneyline', 'Best Odds'].values[0])
chiefs_moneyline_avg = float(odds_df.loc[odds_df['Market'] == 'Chiefs Moneyline', 'Best Odds'].values[0])

# Position weighting - Adjust these weights based on the importance of each position
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
    unique_positions = df['POSITION'].unique()

    for pos in unique_positions:
        features[f'{pos}_rating'] = df[df['POSITION'] == pos]['OVR'].mean()
    
    weighted_rating = 0
    for pos, weight in position_weights.items():
        if pos in unique_positions:
            weighted_rating += df[df['POSITION'] == pos]['OVR'].mean() * weight
    
    features['weighted_rating'] = weighted_rating
    features['avg_overall_rating'] = df['OVR'].mean()
    
    return features

# Extract features for Ravens and Chiefs
ravens_features = extract_team_features(ravens, position_weights)
chiefs_features = extract_team_features(chiefs, position_weights)
print("Ravens Features:", ravens_features)
print("Chiefs Features:", chiefs_features)

# Load and compute key metrics from the records data
records_stats = pd.read_csv('C:/Users/Chuks/Documents/ravens vs cheifs bets/ravens vs chiefs records.csv')

# Calculate key metrics for Ravens and Chiefs
ravens_records_stats = records_stats[records_stats['Team'] == 'Baltimore Ravens']
chiefs_records_stats = records_stats[records_stats['Team'] == 'Kansas City Chiefs']

ravens_wins = ravens_records_stats['W'].sum()
ravens_losses = ravens_records_stats['L'].sum()
ravens_points_scored = ravens_records_stats['Pts'].sum()

chiefs_wins = chiefs_records_stats['W'].sum()
chiefs_losses = chiefs_records_stats['L'].sum()
chiefs_points_scored = chiefs_records_stats['Pts'].sum()

# Load and compute additional key metrics for rushing
rushing_stats = pd.read_csv('C:/Users/Chuks/Documents/ravens vs cheifs bets/ravens vs chiefs rushing.csv')

# Calculate key metrics for Ravens and Chiefs rushing
ravens_rushing_stats = rushing_stats[rushing_stats['Team'] == 'BAL']
chiefs_rushing_stats = rushing_stats[rushing_stats['Team'] == 'KAN']

ravens_rushing_yards = ravens_rushing_stats['Yds'].sum()
ravens_rushing_tds = ravens_rushing_stats['TD'].sum()


chiefs_rushing_yards = chiefs_rushing_stats['Yds'].sum()
chiefs_rushing_tds = chiefs_rushing_stats['TD'].sum()


# Load and compute passing metrics from the new file
passing_stats = pd.read_csv('C:/Users/Chuks/Documents/ravens vs cheifs bets/ravens vs chiefs passing.csv')

# Calculate key metrics for Ravens and Chiefs passing
ravens_passing_stats = passing_stats[passing_stats['Team'] == 'BAL']
chiefs_passing_stats = passing_stats[passing_stats['Team'] == 'KAN']

ravens_passing_yards = ravens_passing_stats['Yds'].sum()
ravens_passing_tds = ravens_passing_stats['TD'].sum()
ravens_interceptions = ravens_passing_stats['Int'].sum()

chiefs_passing_yards = chiefs_passing_stats['Yds'].sum()
chiefs_passing_tds = chiefs_passing_stats['TD'].sum()
chiefs_interceptions = chiefs_passing_stats['Int'].sum()

# Define metrics dictionary
metrics = {
    'ravens_wins': ravens_wins,
    'ravens_losses': ravens_losses,
    'ravens_points_scored': ravens_points_scored,
    'chiefs_wins': chiefs_wins,
    'chiefs_losses': chiefs_losses,
    'chiefs_points_scored': chiefs_points_scored,
    'ravens_rushing_yards': ravens_rushing_yards,
    'ravens_rushing_tds': ravens_rushing_tds,
    'chiefs_rushing_yards': chiefs_rushing_yards,
    'chiefs_rushing_tds': chiefs_rushing_tds,
    'ravens_passing_yards': ravens_passing_yards,
    'ravens_passing_tds': ravens_passing_tds,
    'ravens_interceptions': ravens_interceptions,
    'chiefs_passing_yards': chiefs_passing_yards,
    'chiefs_passing_tds': chiefs_passing_tds,
    'chiefs_interceptions': chiefs_interceptions
}

# Function to simulate outcomes based on team ratings and print scores
def simulate_game_outcomes(team1_features, team2_features, metrics, num_simulations=10000):
    variance_factor = 0.5  # Increased variance for more randomness
    home_field_advantage = 0.05  # Applied to Ravens (team1 in this case)

    # Incorporate additional metrics into simulation
    win_factor_ravens = metrics['ravens_wins'] / (metrics['ravens_wins'] + metrics['ravens_losses'])
    points_factor_ravens = metrics['ravens_points_scored'] / 100
    rushing_factor_ravens = metrics['ravens_rushing_yards'] / 1000
    passing_factor_ravens = metrics['ravens_passing_yards'] / 1000
    interception_factor_ravens = metrics['ravens_interceptions'] / 10

    win_factor_chiefs = metrics['chiefs_wins'] / (metrics['chiefs_wins'] + metrics['chiefs_losses'])
    points_factor_chiefs = metrics['chiefs_points_scored'] / 100
    rushing_factor_chiefs = metrics['chiefs_rushing_yards'] / 1000
    passing_factor_chiefs = metrics['chiefs_passing_yards'] / 1000
    interception_factor_chiefs = metrics['chiefs_interceptions'] / 10

    def simulate_once():
        team1_score = (team1_features['weighted_rating'] * 
                       (1 + variance_factor * np.random.randn()) * 
                       (1 + win_factor_ravens) * (1 + points_factor_ravens) * 
                       (1 + rushing_factor_ravens) * (1 - interception_factor_ravens) *
                       (1 + passing_factor_ravens))
        team2_score = (team2_features['weighted_rating'] * 
                       (1 + variance_factor * np.random.randn()) * 
                       (1 + win_factor_chiefs) * (1 + points_factor_chiefs) * 
                       (1 + rushing_factor_chiefs) * (1 - interception_factor_chiefs) *
                       (1 + passing_factor_chiefs) * (1 + home_field_advantage))
        
        return 1 if team1_score > team2_score else 0
    
    results = Parallel(n_jobs=-1)(delayed(simulate_once)() for _ in range(num_simulations))
    team1_wins = sum(results)
    team2_wins = num_simulations - team1_wins
    
    return team1_wins, team2_wins

# Simulate the outcomes
ravens_wins, chiefs_wins = simulate_game_outcomes(ravens_features, chiefs_features, metrics, num_simulations=10000)

print(f"Ravens won {ravens_wins} games.")
print(f"Chiefs won {chiefs_wins} games.")

# Calculate probabilities from simulations
ravens_prob = ravens_wins / 10000
chiefs_prob = chiefs_wins / 10000

# Convert odds to decimal format
ravens_odds_decimal = 1 + (100 / abs(ravens_moneyline_avg)) if ravens_moneyline_avg < 0 else (1 + ravens_moneyline_avg / 100)
chiefs_odds_decimal = 1 + (chiefs_moneyline_avg / 100) if chiefs_moneyline_avg > 0 else (1 + 100 / abs(chiefs_moneyline_avg))

# Function to calculate Expected Value (EV)
def calculate_ev(prob, payout, bet_amount=100):
    return (prob * payout * bet_amount) - ((1 - prob) * bet_amount)

# Calculate EV for betting on each team
ravens_ev = calculate_ev(ravens_prob, ravens_odds_decimal)
chiefs_ev = calculate_ev(chiefs_prob, chiefs_odds_decimal)

# Output the EVs
print(f'Expected Value for betting on Ravens: ${ravens_ev:.2f}')
print(f'Expected Value for betting on Chiefs: ${chiefs_ev:.2f}')

# Determine if each bet is good or not
if ravens_ev > 0:
    print("Betting on Ravens is a good bet.")
else:
    print("Betting on Ravens is not a good bet.")

if chiefs_ev > 0:
    print("Betting on Chiefs is a good bet.")
else:
    print("Betting on Chiefs is not a good bet.")
