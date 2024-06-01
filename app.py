from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the neural network model
model = load_model('neural_network_model.h5')

# Load the scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Load the classification report
with open('classification_report.pkl', 'rb') as file:
    classification_report = pickle.load(file)

# Load the dataset
data = pd.read_csv("nba_games.csv")

# Define the list of teams
teams = [
    {'name': 'Lakers', 'logo': 'lakers.gif', 'initials': 'LAL'},
    {'name': 'Celtics', 'logo': 'celtics.gif', 'initials': 'BOS'},
    {'name': 'Warriors', 'logo': 'warriors.gif', 'initials': 'GSW'},
    {'name': 'Bulls', 'logo': 'bulls.gif', 'initials': 'CHI'},
    {'name': 'Heat', 'logo': 'heat.gif', 'initials': 'MIA'},
    {'name': 'Knicks', 'logo': 'knicks.gif', 'initials': 'NYK'},
    {'name': 'Spurs', 'logo': 'spurs.gif', 'initials': 'SAS'},
    {'name': 'Raptors', 'logo': 'raptors.gif', 'initials': 'TOR'},
    {'name': 'Mavericks', 'logo': 'mavericks.gif', 'initials': 'DAL'},
    {'name': 'Clippers', 'logo': 'clippers.gif', 'initials': 'LAC'},
    {'name': '76ers', 'logo': '76ers.gif', 'initials': 'PHI'},
    {'name': 'Hawks', 'logo': 'hawks.gif', 'initials': 'ATL'},
    {'name': 'Nets', 'logo': 'nets.gif', 'initials': 'BRK'},
    {'name': 'Bucks', 'logo': 'bucks.gif', 'initials': 'MIL'},
    {'name': 'Suns', 'logo': 'suns.gif', 'initials': 'PHO'},
    {'name': 'Nuggets', 'logo': 'nuggets.gif', 'initials': 'DEN'},
    {'name': 'Grizzlies', 'logo': 'grizzlies.gif', 'initials': 'MEM'},
    {'name': 'Pacers', 'logo': 'pacers.gif', 'initials': 'IND'},
    {'name': 'Pistons', 'logo': 'pistons.gif', 'initials': 'DET'},
    {'name': 'Jazz', 'logo': 'jazz.gif', 'initials': 'UTA'},
    {'name': 'Thunder', 'logo': 'okc.gif', 'initials': 'OKC'},
    {'name': 'Kings', 'logo': 'kings.gif', 'initials': 'SAC'},
    {'name': 'Rockets', 'logo': 'rockets.gif', 'initials': 'HOU'},
    {'name': 'Hornets', 'logo': 'hornets.gif', 'initials': 'CHO'},
    {'name': 'Magic', 'logo': 'magic.gif', 'initials': 'ORL'},
    {'name': 'Cavaliers', 'logo': 'cleveland.gif', 'initials': 'CLE'},
    {'name': 'Wizards', 'logo': 'wizards.gif', 'initials': 'WAS'},
    {'name': 'Timberwolves', 'logo': 'timberwolves.gif', 'initials': 'MIN'},
]

selected_features = ['home', 'trb', 'ast', 'tov', 'stl', 'blk', '+/-', 'ortg', 'drtg', 'ts%']

# Helper function to get team name from initials
def get_team_name_by_initials(initials):
    for team in teams:
        if team['initials'] == initials:
            return team['name']
    return None

# Function to get team features
def get_team_features(team_name, home_status):
    if home_status:
        team_data = data[data['team'] == team_name]
    else:
        team_data = data[data['team_opp'] == team_name]

    # Handle missing values by filling with 0 or another appropriate value
    team_data = team_data.fillna(0)
    
    # Convert all columns to numeric, coercing errors
    team_data = team_data.apply(pd.to_numeric, errors='coerce')

    print(team_data)

    team_features = team_data.mean()
    return team_features

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html', teams=teams)

# Define the route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    home_team = request.form['home_team']
    away_team = request.form['away_team']
    
    # Get features for both teams
    home_team_features = get_team_features(home_team, home_status=True)
    away_team_features = get_team_features(away_team, home_status=False)
    
    # Combine features into a single feature set
    features = home_team_features[selected_features] - away_team_features[selected_features]
    
    # Create a DataFrame for prediction
    features_df = pd.DataFrame([features])
    
    # Scale the features
    features_scaled = scaler.transform(features_df)
    
    # Make a prediction
    prediction = model.predict(features_scaled)[0][0]

    home_team_name = get_team_name_by_initials(home_team)
    away_team_name = get_team_name_by_initials(away_team)
    
    # Map the prediction to a human-readable label
    result = f"{home_team_name} Wins" if prediction > 0.5 else f"{away_team_name} Wins"
    
    return render_template('index.html', prediction_text=f'Prediction: {result}', classification_report=classification_report, teams=teams, teams_playing=f'{home_team_name} VS {away_team_name}')

if __name__ == '__main__':
    app.run(debug=True)
