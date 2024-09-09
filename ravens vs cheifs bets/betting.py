import pandas as pd
import numpy as np
from joblib import Parallel, delayed

# Load the rosters for Ravens and Chiefs
ravens = pd.read_csv('C:/Users/Chuks/Documents/ravens vs cheifs bets/ravens roster ratings.csv')
chiefs = pd.read_csv('C:/Users/Chuks/Documents/ravens vs cheifs bets/chiefs roster ratings.csv')

# Normalize the ratings to a 0-1 scale
ravens['OVR'] = ravens['OVR'] / 100
chiefs['OVR'] = chiefs['OVR'] / 100

# Load the betting odds data
odds_df = pd.read_csv('C:/Users/Chuks/Documents/ravens vs cheifs bets/ravens_chiefs_odds.csv')

# Extract specific averages for Ravens and Chiefs Moneyline
ravens_moneyline_avg = odds_df.loc[odds_df['Market'] == 'Ravens Moneyline', 'Best Odds'].values[0]
chiefs_moneyline_avg = odds_df.loc[odds_df['Market'] == 'Chiefs Moneyline', 'Best Odds'].values[0]

# Convert the odds to numeric values
ravens_moneyline_avg = float(ravens_moneyline_avg)
chiefs_moneyline_avg = float(chiefs_moneyline_avg)

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

# Load and compute head-to-head metrics
rushing_stats = pd.read_csv('C:/Users/Chuks/Documents/ravens vs cheifs bets/ravens vs chiefs rushing.csv')
passing_stats = pd.read_csv('C:/Users/Chuks/Documents/ravens vs cheifs bets/ravens vs chiefs passing.csv')
defense_stats = pd.read_csv('C:/Users/Chuks/Documents/ravens vs cheifs bets/ravens vs chiefs defense.csv')

# Calculate key rushing metrics
ravens_rushing_yards = rushing_stats.loc[rushing_stats['Team'] == 'BAL', 'Yds'].sum()
chiefs_rushing_yards = rushing_stats.loc[rushing_stats['Team'] == 'KAN', 'Yds'].sum()
ravens_avg_yards_per_carry = rushing_stats.loc[rushing_stats['Team'] == 'BAL', 'Yds'].sum() / rushing_stats.loc[rushing_stats['Team'] == 'BAL', 'Att'].sum() if rushing_stats.loc[rushing_stats['Team'] == 'BAL', 'Att'].sum() != 0 else 0
chiefs_avg_yards_per_carry = rushing_stats.loc[rushing_stats['Team'] == 'KAN', 'Yds'].sum() / rushing_stats.loc[rushing_stats['Team'] == 'KAN', 'Att'].sum() if rushing_stats.loc[rushing_stats['Team'] == 'KAN', 'Att'].sum() != 0 else 0
ravens_rushing_touchdowns = rushing_stats.loc[rushing_stats['Team'] == 'BAL', 'TD'].sum()
chiefs_rushing_touchdowns = rushing_stats.loc[rushing_stats['Team'] == 'KAN', 'TD'].sum()

# Calculate key passing metrics
ravens_passing_yards = passing_stats.loc[passing_stats['Team'] == 'BAL', 'Yds'].sum()
chiefs_passing_yards = passing_stats.loc[passing_stats['Team'] == 'KAN', 'Yds'].sum()
ravens_completion_pct = passing_stats.loc[passing_stats['Team'] == 'BAL', 'Cmp'].sum() / passing_stats.loc[passing_stats['Team'] == 'BAL', 'Att'].sum() if passing_stats.loc[passing_stats['Team'] == 'BAL', 'Att'].sum() != 0 else 0
chiefs_completion_pct = passing_stats.loc[passing_stats['Team'] == 'KAN', 'Cmp'].sum() / passing_stats.loc[passing_stats['Team'] == 'KAN', 'Att'].sum() if passing_stats.loc[passing_stats['Team'] == 'KAN', 'Att'].sum() != 0 else 0
ravens_passing_touchdowns = passing_stats.loc[passing_stats['Team'] == 'BAL', 'TD'].sum()
chiefs_passing_touchdowns = passing_stats.loc[passing_stats['Team'] == 'KAN', 'TD'].sum()

# Calculate key defense metrics
ravens_sacks = defense_stats.loc[defense_stats['Team'] == 'BAL', 'Sk'].sum()
chiefs_sacks = defense_stats.loc[defense_stats['Team'] == 'KAN', 'Sk'].sum()
ravens_interceptions = defense_stats.loc[defense_stats['Team'] == 'BAL', 'Int'].sum()
chiefs_interceptions = defense_stats.loc[defense_stats['Team'] == 'KAN', 'Int'].sum()
ravens_forced_fumbles = defense_stats.loc[defense_stats['Team'] == 'BAL', 'FF'].sum()
chiefs_forced_fumbles = defense_stats.loc[defense_stats['Team'] == 'KAN', 'FF'].sum()

# Define metrics for Ravens and Chiefs
ravens_metrics = {
    'rushing_yards': ravens_rushing_yards,
    'passing_yards': ravens_passing_yards,
    'sacks': ravens_sacks,
    'avg_yards_per_carry': ravens_avg_yards_per_carry,
    'completion_pct': ravens_completion_pct,
    'passing_touchdowns': ravens_passing_touchdowns,
    'rushing_touchdowns': ravens_rushing_touchdowns
}

chiefs_metrics = {
    'rushing_yards': chiefs_rushing_yards,
    'passing_yards': chiefs_passing_yards,
    'sacks': chiefs_sacks,
    'avg_yards_per_carry': chiefs_avg_yards_per_carry,
    'completion_pct': chiefs_completion_pct,
    'passing_touchdowns': chiefs_passing_touchdowns,
    'rushing_touchdowns': chiefs_rushing_touchdowns
}

# Function to simulate outcomes based on team ratings and track scores
def simulate_game_outcomes_with_scores(team1_features, team2_features, team1_metrics, team2_metrics, num_simulations=10000):
    variance_factor = 0.5  # Increased variance for more randomness
    home_field_advantage = 0.05  # Applied to team1 (Ravens)
    
    # Incorporate additional metrics into simulation for both teams
    team1_rushing_factor = team1_metrics.get('rushing_yards', 0) / 1000 if team1_metrics.get('rushing_yards', 0) != 0 else 1
    team1_passing_factor = team1_metrics.get('passing_yards', 0) / 1000 if team1_metrics.get('passing_yards', 0) != 0 else 1
    team1_defense_factor = team1_metrics.get('sacks', 0) / 10 if team1_metrics.get('sacks', 0) != 0 else 1

    team2_rushing_factor = team2_metrics.get('rushing_yards', 0) / 1000 if team2_metrics.get('rushing_yards', 0) != 0 else 1
    team2_passing_factor = team2_metrics.get('passing_yards', 0) / 1000 if team2_metrics.get('passing_yards', 0) != 0 else 1
    team2_defense_factor = team2_metrics.get('sacks', 0) / 10 if team2_metrics.get('sacks', 0) != 0 else 1
    
    team1_scores = []
    team2_scores = []

    # Get average overall ratings for both teams
    team1_avg_overall_rating = team1_features.get('avg_overall_rating', 0)
    team2_avg_overall_rating = team2_features.get('avg_overall_rating', 0)

    def simulate_once():
        # Ensure no NaN in the ratings by replacing them with the average overall rating
        team1_weighted_rating = team1_features.get('weighted_rating', 0)
        team2_weighted_rating = team2_features.get('weighted_rating', 0)

        if np.isnan(team1_weighted_rating):
            team1_weighted_rating = team1_avg_overall_rating
        if np.isnan(team2_weighted_rating):
            team2_weighted_rating = team2_avg_overall_rating

        # Calculate the team scores using the weighted ratings and respective factors
        team1_score = team1_weighted_rating * (1 + variance_factor * np.random.randn()) * (1 + team1_rushing_factor) * (1 + team1_passing_factor) * (1 + team1_defense_factor)
        team2_score = team2_weighted_rating * (1 + variance_factor * np.random.randn()) * (1 + team2_rushing_factor) * (1 + team2_passing_factor) * (1 + team2_defense_factor) * (1 + home_field_advantage)
        
        team1_scores.append(team1_score)
        team2_scores.append(team2_score)
        return 1 if team1_score > team2_score else 0
    
    # Simulate the games
    results = Parallel(n_jobs=-1)(delayed(simulate_once)() for _ in range(num_simulations))
    team1_wins = sum(results)
    team2_wins = num_simulations - team1_wins

    # Calculate average scores while handling NaN values
    avg_team1_score = np.nanmean(team1_scores) if team1_scores else 0
    avg_team2_score = np.nanmean(team2_scores) if team2_scores else 0
    
    return team1_wins, team2_wins, avg_team1_score, avg_team2_score


# Simulate the outcomes and calculate average scores using the respective team metrics
ravens_wins, chiefs_wins, avg_ravens_score, avg_chiefs_score = simulate_game_outcomes_with_scores(ravens_features, chiefs_features, ravens_metrics, chiefs_metrics, num_simulations=10000)

# Print the simulation results
print(f"\nRavens won {ravens_wins} games with an average score of {avg_ravens_score:.2f}.")
print(f"Chiefs won {chiefs_wins} games with an average score of {avg_chiefs_score:.2f}.")

# Calculate probabilities from simulations
ravens_prob = ravens_wins / 10000
chiefs_prob = chiefs_wins / 10000

# Convert odds to decimal format
ravens_odds_decimal = 1 + (100 / abs(ravens_moneyline_avg)) if ravens_moneyline_avg < 0 else (1 + ravens_moneyline_avg / 100)
chiefs_odds_decimal = 1 + (chiefs_moneyline_avg / 100) if chiefs_moneyline_avg > 0 else (1 + 100 / abs(chiefs_moneyline_avg))

# Function to calculate the outcome of a single bet
def single_bet_outcome(prob, payout, bet_amount=100):
    # Calculate expected winnings or loss
    winnings = (payout * bet_amount) if prob > np.random.rand() else -bet_amount
    return winnings

# Simulate a single bet on each team
ravens_single_bet_result = single_bet_outcome(ravens_prob, ravens_odds_decimal)
chiefs_single_bet_result = single_bet_outcome(chiefs_prob, chiefs_odds_decimal)

# Output the results of a single bet
print(f"\nSingle Bet Outcome for Ravens: ${ravens_single_bet_result:.2f}")
print(f"Single Bet Outcome for Chiefs: ${chiefs_single_bet_result:.2f}")

# Determine if each bet is good or not based on the single outcome
if ravens_single_bet_result > 0:
    print("Betting on Ravens is a profitable single bet.")
else:
    print("Betting on Ravens is a losing single bet.")

if chiefs_single_bet_result > 0:
    print("Betting on Chiefs is a profitable single bet.")
else:
    print("Betting on Chiefs is a losing single bet.")
