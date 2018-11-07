# gametrend

# Table of Contents

1. [Summary](#summary)
2. [Data Collection](#datacollection)
3. [Data Analysis & Model Development](#analysis)
4. [Deployment of Model](#deploy)

# Summary<a id='summary'></a>

Uses API calls to collect game feature data from Steam.com and game performance metrics from Steamspy.com for all games in Steam's database. Creates new variables and analyzes all data for outliers, normality, trends, etc. Builds two linear regression models from training data. The first model predicts expected downloads of game per month, and the second model predicts expected median hours played per owner. Validate these models with holdout test set and compare them to more complicated models (ridge regression and random forest regressor). Save models using pickle. Allow user input via web app and calculate predicted game success from user input. Return both predicted success for user input and predicted success for user input plus the next three highest ROI features. Display results via web app.

# Data Collection<a id='datacollection'></a>

* idlist.py - Generate a list of all game IDs from Steam.com (saved to idlist.csv) to be used in games.py and steamspy.py
* games.py - Create games.json using game ID's in idlist.csv
* games-features.py - Create games-features.csv with game feature data from games.json.
* steamspy.py - Create steamspy.csv using game ID's in idlist.csv. API calls to Steamspy.com return:
    1) appid - Steam application ID (matched to game IDs from idlist.csv)
    2) name - name of game
    3) developer - comma separated list of the developers of the game
    4) positive - number of positive reviews of the game on Steam
    5) negative - number of negative reviews of the game on Steam
    6) userscore - score rank of the game based on user reviews
    7) owners - number of owners of this game on Steam, expressed as a range
    8) average_forever - average playtime per owner (in minutes) since March 2009
    9) average_2weeks - average playtime per owner (in minutes) in the last two weeks
    10) median_forever - median playtime per owner (in minutes) since March 2009
    11) median_2weeks - median playtime per owner (in minutes) in the last two weeks
    12) price - current price of the game in the United States in US cents
    13) initialprice - initial price of the game in the United States in US cents

# Data Analysis & Model Development<a id='analysis'></a>

* analysis.py - Merge games-features.csv and steamspy.csv into final data set. Analyze data, build model, validate model and export model to pickle for deployment to website
* model_to_csv.py - Train regression models from training data and save with pickle. Returns finalized_model.sav and finalized_hours_model.sav

# Deployment of Model<a id='deploy'></a>

* app.py - Formats user input from feat.html, computes model predictions and renders players.html to display predictions
* feat.html - Display possible user inputs (i.e., model features) and accept input values. Pass user input to app.py to calculate model predicitons
* players.html - Rendered from app.py. Display model predictions for user input as well as model predictions for next three highest ROIs for both expected downloads and median hours played per owner
