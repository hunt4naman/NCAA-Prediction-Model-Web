# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 16:54:48 2015

Randomizer for game data. Currently, all winning teams are in
the same column. We want to randomize this so that roughly
50% of the time, winning teams are on the left column
and 50% of the time, winning teams are on the right column, so
that our classifier isn't biased towards any particular column.

@author: Julian
"""

import pandas as pd
import numpy as np

###############################################################################
## For the training data

# Import data
data = 'regular_season_compact_results.csv'
df = pd.read_csv(data)
df.columns = ['season','daynum','teamA','scoreA','teamB','scoreB','wloc','numot']
print "Total data size: %d" % len(df)

# Choose random rows to be switched. Do this by creating
# random number from uniform distribution between 0 and 1.
# 50% of rows should be <.5, 50% of rows should be between .5 and 1
random = np.random.random(len(df))
switch = random < 0.50

# If row is a switched row, winner
# will be team B. Otherwise, winner
# will be team A.
df['winner'] = np.where(switch, 'B', 'A')

# Switch the columns.
# Why does .values work?
df.head()
df.loc[switch[:len(df)], ['teamA','teamB']] = df.loc[switch[:len(df)], ['teamB','teamA']].values
df.loc[switch[:len(df)], ['scoreA','scoreB']] = df.loc[switch[:len(df)], ['scoreB','scoreA']].values
df.head()

# Fix location
def get_loser(winner):
    if winner == 'A':
        return 'B'
    else:
        return 'A'
        
df['loser'] = df['winner'].apply(get_loser)
df['location'] = np.where( df.wloc == 'N', 'N', None )
df.loc[ df.wloc == 'H' , ['location'] ] = df.winner
df.loc[ df.wloc == 'A' , ['location'] ] = df.loser

# Write to file
df.to_csv('randomized_teams.csv')
df[ df.season == 2014 ].to_csv('randomized_teams_2014.csv')

###############################################################################
## For the test data (the tournament games)

# Import data
data = 'ncaa_tournament_games.csv'
df = pd.read_csv(data)
print "Total data size: %d" % len(df)

# Choose random rows to be switched. Do this by creating
# random number from uniform distribution between 0 and 1.
# 50% of rows should be <.5, 50% of rows should be between .5 and 1
random = np.random.random(len(df))
switch = random < 0.50

# If row is a switched row, winner
# will be team B. Otherwise, winner
# will be team A.
df['winner'] = np.where(switch, 'B', 'A')

# Switch the columns.
# Why does .values work?
df.head()
df.loc[switch[:len(df)], ['teamA','teamB']] = df.loc[switch[:len(df)], ['teamB','teamA']].values
df.loc[switch[:len(df)], ['scoreA','scoreB']] = df.loc[switch[:len(df)], ['scoreB','scoreA']].values
df.loc[switch[:len(df)], ['TeamName_x','TeamName_y']] = df.loc[switch[:len(df)], ['TeamName_y','TeamName_x']].values

df.head()

# Write to file
df.to_csv('ncaa_tournament_games_randomized.csv')