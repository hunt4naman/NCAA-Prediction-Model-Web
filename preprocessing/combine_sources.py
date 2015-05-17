# -*- coding: utf-8 -*-
"""
Created on Thu May 07 14:31:04 2015

This code combines data from three separate sources:
    1. team_data_kenpom.csv        team data (from kenpom.com)
    2. team_data_kaggle.csv        Amit's team data (derived from 
                                   data available at kaggle.com)
    3. games_regular_season.csv    list of games (from kaggle.com)

There are two files with team data because different sources had different
metrics.

We also derive a couple new metrics from the data.

To combine data from multiple sources, we need to join by the team ID. All
input files must have this team ID. All team IDs are found in the file:
    team_ids.csv

The output is the final combined dataset.

@author: Julian Sy
"""

import pandas as pd

# Get file names
games_file = 'games_regular_season.csv'
ncaa_games_file = 'ncaa_tournament_games_randomized.csv'
team_data_kaggle_file = 'team_data_kaggle.csv'
team_data_kenpom_file = 'team_data_kenpom.csv'

# Import files to dataframes
df_games = pd.read_csv(games_file)
df_ncaa_games = pd.read_csv(ncaa_games_file)
df_team_data_kaggle = pd.read_csv(team_data_kaggle_file)
df_team_data_kenpom = pd.read_csv(team_data_kenpom_file)

## Derive new metrics using the team data from kaggle

# effective field goal percentage
df_team_data_kaggle['eFG'] = (.5*df_team_data_kaggle['fgm3'] + df_team_data_kaggle['fgm']) / df_team_data_kaggle['fga']

# free throw rate
df_team_data_kaggle['ftrate'] = df_team_data_kaggle['fta'] / df_team_data_kaggle['fga']

# total possessions
df_team_data_kaggle['possesions'] = df_team_data_kaggle['fga'] - df_team_data_kaggle['or'] + df_team_data_kaggle['to'] + .475*df_team_data_kaggle['fta']

# turnover percentage
df_team_data_kaggle['to_percent'] = df_team_data_kaggle['to'] / df_team_data_kaggle['possesions']


# TRAINING SET: Join dataframes using the team ID
df_games_kaggle_A = pd.merge(df_games, df_team_data_kaggle, how='inner', left_on='teamA', right_on='teamId')
df_games_kaggle_kenpom_A = pd.merge(df_games_kaggle_A, df_team_data_kenpom, how='inner', left_on='teamA', right_on='team_id')
df_games_kaggle_kenpom_A_kaggle_B = pd.merge(df_games_kaggle_kenpom_A, df_team_data_kaggle, how='inner', left_on='teamB', right_on='teamId')
df_games_kaggle_kenpom_A_kaggle_kenpom_B = pd.merge(df_games_kaggle_kenpom_A_kaggle_B, df_team_data_kenpom, how='inner', left_on='teamB', right_on='team_id')

# TRAINING SET: Write to file
df_games_kaggle_kenpom_A_kaggle_kenpom_B.to_csv('combined_dataset.csv')

# TEST SET: Join dataframes using the team ID
df_ncaa_games_kaggle_A = pd.merge(df_ncaa_games, df_team_data_kaggle, how='inner', left_on='teamA', right_on='teamId')
df_ncaa_games_kaggle_kenpom_A = pd.merge(df_ncaa_games_kaggle_A, df_team_data_kenpom, how='inner', left_on='teamA', right_on='team_id')
df_ncaa_games_kaggle_kenpom_A_kaggle_B = pd.merge(df_ncaa_games_kaggle_kenpom_A, df_team_data_kaggle, how='inner', left_on='teamB', right_on='teamId')
df_ncaa_games_kaggle_kenpom_A_kaggle_kenpom_B = pd.merge(df_ncaa_games_kaggle_kenpom_A_kaggle_B, df_team_data_kenpom, how='inner', left_on='teamB', right_on='team_id')

# TEST SET: Write to file
df_ncaa_games_kaggle_kenpom_A_kaggle_kenpom_B.to_csv('combined_ncaa_tournament_dataset.csv')