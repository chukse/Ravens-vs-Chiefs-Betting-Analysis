# Ravens vs Chiefs NFL Analysis and Betting Prediction

This repository contains various datasets and Python scripts focused on analyzing and predicting outcomes for the NFL game between the Baltimore Ravens and Kansas City Chiefs. The data covers different aspects of the teams' performance, including rushing, passing, defense, and player ratings.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Data Description](#data-description)
  - [Odds Data](#odds-data)
  - [Game Performance Data](#game-performance-data)
  - [Team Roster Ratings](#team-roster-ratings)
- [How to Run the Analysis](#how-to-run-the-analysis)
- [Installation](#installation)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project aims to analyze key statistics of both the Baltimore Ravens and Kansas City Chiefs from a recent matchup to gain insights into their performance and provide betting odds predictions.

The analysis involves:
- Rushing and passing statistics.
- Defensive performance.
- Roster ratings.
- Betting odds.

The Python script `betting.py` processes the provided data and generates predictions for betting purposes.

## Project Structure

The project files are organized as follows:


├── betting.py                            # Python script for betting prediction analysis
├── ravens_chiefs_odds.csv                 # Historical betting odds for the matchup
├── ravens vs chiefs rushing.csv           # Rushing statistics for the game
├── ravens vs chiefs passing.csv           # Passing statistics for the game
├── ravens vs chiefs defense.csv           # Defensive statistics for the game
├── ravens vs chiefs records.csv           # Overall game records
├── ravens roster ratings.csv              # Ratings of Ravens players
├── chiefs roster ratings.csv              # Ratings of Chiefs players

## Data Description

### Odds Data

**File:** `ravens_chiefs_odds.csv`

This file contains the historical betting odds for the Ravens vs. Chiefs matchup. The data is structured to include information on:

- The point spread.
- The money line.
- Over/under bets.

### Game Performance Data

**Files:**
- `ravens vs chiefs rushing.csv`: Contains rushing statistics such as total yards, attempts, and touchdowns for each team.
- `ravens vs chiefs passing.csv`: Contains passing statistics, including yards, completions, and touchdowns.
- `ravens vs chiefs defense.csv`: Defensive statistics covering tackles, sacks, and interceptions.
- `ravens vs chiefs records.csv`: Overall game records for both teams, including wins, losses, and notable highlights.

### Team Roster Ratings

**Files:**
- `ravens roster ratings.csv`: Ratings for the Ravens players, covering various attributes like speed, strength, and overall performance.
- `chiefs roster ratings.csv`: Similar ratings for the Chiefs players.
## How to Run the Analysis

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/ravens-chiefs-analysis.git
   cd ravens-chiefs-analysis
   pip install -r requirements.txt
   python betting.py

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request. All contributions are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


