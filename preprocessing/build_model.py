# -*- coding: utf-8 -*-
"""
Created on Thu May 07 19:45:18 2015

This code imports the final dataset, and builds the predictive model.

@author: JS-HP-SLIM
"""

import pandas as pd

# Get filename
data_file = 'combined_dataset.csv'
ncaa_data_file = 'combined_ncaa_tournament_dataset.csv'

# Import data to dataframe
df_original = pd.read_csv(data_file)
df_ncaa_original = pd.read_csv(ncaa_data_file)

# Base dataframe (this has everything about the game EXCEPT team metrics);
# we can use this to create a cleaner version of df_original
base = df_original[ ['teamA', 'TeamName_x', 'scoreA', 'teamB', 'TeamName_y', 'scoreB', 'winner', 'location'] ]
base_ncaa = df_ncaa_original[ ['round','teamA', 'TeamName_x', 'scoreA', 'teamB', 'TeamName_y', 'scoreB', 'winner'] ]

print base_ncaa.columns
###############################################################################
## Get differences of attribute from kaggle dataset only

# A list of attributes only found in Kaggle dataset
kaggle_attributes = [u'fgm', u'fga', u'fgm3', u'fga3', u'ftm', u'fta', u'or', u'dr', u'ast', u'to', u'stl', u'blk', u'pf']

# A list of derived attributes from the kaggle dataset
kaggle_derived_attr = [u'eFG', u'ftrate', u'possesions', u'to_percent']

# A dataframe to store the differences
df_kaggle_only = base.copy(deep=True)
df_kaggle_only_ncaa = base_ncaa.copy(deep=True)

# Build the dataset with the differences (note: _x, and _y are how
# pandas named the attributes with the same names when we joined the data.)
for attribute in kaggle_attributes:
    
    # create column names
    team_A_value = attribute + '_x'
    team_B_value = attribute + '_y'
    difference = attribute + '_diff'
    
    df_kaggle_only[ difference ] = df_original[team_A_value] - df_original[team_B_value]
    df_kaggle_only_ncaa[ difference ] = df_ncaa_original[team_A_value] - df_ncaa_original[team_B_value]



###############################################################################
## Get differences of attributes from kenpom dataset only

# A list of attributes only found in the kenpom dataset
kenpom_attributes = [ u'Tempo', u'RankTempo', u'AdjTempo', u'RankAdjTempo', u'OE', u'RankOE', u'AdjOE', u'RankAdjOE', u'DE', u'RankDE', u'AdjDE', u'RankAdjDE', u'Pythag', u'RankPythag']

# A dataframe to store the differences
df_kenpom_only = base.copy(deep=True)
df_kenpom_only_ncaa = base_ncaa.copy(deep=True)

# Build the dataset with the desired differences (note: _x, and _y are how
# pandas named the attributes with the same names when we joined the data.)

# difference in tempo between the two teams
df_kenpom_only['tempo_diff'] = df_original['Tempo_x'] - df_original['Tempo_y']
df_kenpom_only_ncaa['tempo_diff'] = df_ncaa_original['Tempo_x'] - df_ncaa_original['Tempo_y']

# difference between A's offensive efficiency and B's defensive efficiency
df_kenpom_only['oe_de_diff'] = df_original['AdjOE_x'] - df_original['AdjDE_y']
df_kenpom_only_ncaa['oe_de_diff'] = df_ncaa_original['AdjOE_x'] - df_ncaa_original['AdjDE_y']

# difference between A's defensive efficiency and B's offensive efficiency
df_kenpom_only['de_oe_diff'] = df_original['AdjDE_x'] - df_original['AdjOE_y']
df_kenpom_only_ncaa['de_oe_diff'] = df_ncaa_original['AdjDE_x'] - df_ncaa_original['AdjOE_y']

final = pd.concat([df_kaggle_only,df_kenpom_only[ ['tempo_diff', 'oe_de_diff', 'de_oe_diff'] ] ], axis=1)
final.to_csv('final.csv')

final_ncaa = pd.concat([df_kaggle_only_ncaa, df_kenpom_only_ncaa[ ['tempo_diff', 'oe_de_diff', 'de_oe_diff'] ] ], axis=1)
final_ncaa.to_csv('final_ncaa.csv')