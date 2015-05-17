from flask import render_template, flash, redirect, session, send_from_directory
from flask import request
from app import app
import pandas as pd
import sys
import os
import pickle
from sklearn.linear_model import LogisticRegression

import flask
import pandas
import sklearn
print flask.__version__
print pandas.__version__
print sklearn.__version__

# Import list of teams
df_teams = pd.read_csv('./team_ids.csv')
team_names = df_teams['team_name'].values.tolist()
team_ids = df_teams['team_id'].values.tolist()

# Zip list of teams 
team_list = zip(team_names, team_ids)

# Import team data
team_data_kaggle = pd.read_csv('./team_data_kaggle.csv')
team_data_kenpom = pd.read_csv('./team_data_kenpom.csv')

# Merge data from kaggle and kenpom sources
full_team_data = pd.merge(team_data_kaggle, team_data_kenpom, how='inner', left_on='teamId', right_on='team_id')

# Import model
clf_log = pickle.load( open('./logistic_regression.pickle', 'rb') )
@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home(title='Home'):
    """
    Home page: The user will see two dropdown menus, each a list of all NCAA division
    I basketball teams. The user must select two teams, and this app will predict the
    winner.
    """

    # if a form was submitted
    if request.method=='POST':
        team_id_A = int(request.form['team_A'])
        team_id_B = int(request.form['team_B'])

        game_info = ['teamA', 'teamB']
        attributes = [u'TeamName', u'fgm', u'fga', u'fgm3', u'fga3', u'ftm', u'fta', u'or', u'dr', u'ast', u'to', u'stl', u'blk', u'pf']
        attributes_kenpom = [u'AdjTempo', u'AdjOE', u'AdjDE']

        print full_team_data.columns

        team_A_data = full_team_data.loc[ full_team_data['teamId']==team_id_A , attributes ]
        team_B_data = full_team_data.loc[ full_team_data['teamId']==team_id_B , attributes ]
        
        team_A_data_kenpom = full_team_data.loc[ full_team_data['teamId']==team_id_A , attributes_kenpom ]
        team_B_data_kenpom = full_team_data.loc[ full_team_data['teamId']==team_id_A , attributes_kenpom ]

        matchup = pd.DataFrame({'teamA': [team_id_A], 'teamB': [team_id_B]})

        for attribute in attributes:

            if attribute == u'TeamName':
                continue

            difference = attribute + '_diff'

            print team_A_data[attribute].values
            print team_B_data[attribute].values

            matchup[difference] = team_A_data[attribute].values - team_B_data[attribute].values

        matchup['tempo_diff'] = team_A_data_kenpom['AdjTempo'].values - team_B_data_kenpom['AdjTempo'].values
        matchup['oe_de_diff'] = team_A_data_kenpom['AdjOE'].values - team_B_data_kenpom['AdjDE'].values
        matchup['de_oe_diff'] = team_A_data_kenpom['AdjDE'].values - team_B_data_kenpom['AdjOE'].values

        test_features = matchup.ix[:, 'fgm_diff':'de_oe_diff']

        preds_logreg_test = clf_log.predict(test_features)[0]
        probs = clf_log.predict_proba(test_features.astype(float)).tolist()[0]

        
        teams_in_matchup = (team_A_data['TeamName'].values[0], team_B_data['TeamName'].values[0])

        loser = 0
        winner = teams_in_matchup[preds_logreg_test]
        if preds_logreg_test == 1:
            loser = teams_in_matchup[0]
        else:
            loser = teams_in_matchup[1]

        team_A = teams_in_matchup[0]
        team_B = teams_in_matchup[1]

        winner_prob = probs[preds_logreg_test]
        loser_prob = probs[preds_logreg_test]

        return render_template('home.html', title=title, team_list=team_list, teams_in_matchup=teams_in_matchup, winner_prob=winner_prob, winner=winner, loser=loser, team_A=team_id_A, team_B=team_id_B) 

    return render_template('home.html', title=title, team_list=team_list)

@app.route('/search_results')
def search_results(title='Search Results', links=None):
    """
    Search results, shown after user enters something in the home page. The user
    can click on any search result to see its visualization
    """

    # We need to grab to the links we got from the form on the home page.
    links = session.pop('links', None)

    # Get rid of the beginning of each link to get the term (article title)
    terms = [link[link.rfind('/')+1:] for link in links]
    return render_template('search_results.html', title=title, links=terms)

@app.route('/about')
def about(title='About'):
    """
    About page.
    """
    return render_template('about.html', title=title)

if __name__ == '__main__':
    app.run()
