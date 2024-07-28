import streamlit as st
import pandas as pd
import numpy as np
import base64
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import os
import webbrowser
import subprocess

# Read data
bat = pd.read_csv("bat.csv")
bowl = pd.read_csv("bowl.csv")
point = {"6s": 2, "50": 8, "100": 16, "200": 32, "Wkts": 25, "4W": 8, "5W": 16, "10W": 32}

match_types = ['IPL', 'ODI', 'T20', 'Test']
teams = [team.title() for team in list(bat['Team'].unique())]
sort_types = ['Batting', 'Bowling', 'Overall']

def train_random_forest_model(num_folds=5):
    # Merge batting and bowling data
    players = pd.merge(bat, bowl, how='outer', on='Player')

    # Select only numeric columns for imputation
    numeric_columns = players.select_dtypes(include=['number']).columns

    # Impute missing values with mean for numeric columns
    imputer = SimpleImputer(strategy='mean')
    players_filled = pd.DataFrame(imputer.fit_transform(players[numeric_columns]), columns=numeric_columns)

    # Concatenate non-numeric columns with imputed numeric columns
    players_filled = pd.concat([players[['Player']], players_filled], axis=1)

    # Feature engineering
    players_filled['Batting Points'] = players_filled['Runs_x'] + players_filled['4s'] + players_filled['6s']*point['6s'] + players_filled['50']*point['50'] + players_filled['100']*point['100'] + players_filled['200']*point['200']
    players_filled['Bowling Points'] = players_filled['Wkts']*point['Wkts'] + players_filled['5W']*(point['4W'] + point['5W']) + players_filled['10W']*point['10W']
    players_filled['Overall Points'] = players_filled['Batting Points'] + players_filled['Bowling Points']
    
    # Split data into features and target
    X = players_filled[['Runs_x', 'SR_x', '4s', '6s', '50', '100', '200', 'Wkts', 'Econ', '5W', '10W']]
    y = players_filled['Overall Points']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model with k-fold cross-validation
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    cv_rmse_scores = []
    for train_index, test_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_fold, y_train_fold)
        
        y_pred_fold = model.predict(X_val_fold)
        rmse_fold = np.sqrt(mean_squared_error(y_val_fold, y_pred_fold))
        cv_rmse_scores.append(rmse_fold)
    
    mean_cv_rmse = np.mean(cv_rmse_scores)
    
    # Train the model on full training data for train RMSE
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    
    # Calculate test RMSE
    y_pred_test = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    # Train the model on full data for accuracy calculation
    model.fit(X, y)
    y_pred = model.predict(X)
    threshold = 20
    accurate_predictions = sum(abs(y - y_pred) <= threshold)
    total_predictions = len(y)
    accuracy = accurate_predictions / total_predictions
    
    return model, train_rmse, test_rmse, mean_cv_rmse, accuracy

def make_my_dream11(match_type, team1, team2, team3, batsman, bowlers):
    bat_df = bat[bat['Type'] == match_type]
    bowl_df = bowl[bowl['Type'] == match_type]

    bat_data = bat_df[(bat_df['Team'] == team1) | (bat_df['Team'] == team2) | (bat_df['Team'] == team3)]
    bowl_data = bowl_df[(bowl_df['Team'] == team1) | (bowl_df['Team'] == team2) | (bowl_df['Team'] == team3)]

    bat_data = bat_data[['Runs', 'SR', '4s', '6s', '50', '100', '200', 'Player', 'Team']]
    bowl_data = bowl_data[['Wkts', 'Econ', '5W', '10W', 'Player']]

    players = pd.merge(bat_data, bowl_data, how='outer', on='Player')
    players['Batting Points'] = players['Runs'] + players['4s'] + players['6s']*point['6s'] + players['50']*point['50'] + players['100']*point['100'] + players['200']*point['200']
    players['Bowling Points'] = players['Wkts']*point['Wkts'] + players['5W']*(point['4W'] + point['5W']) + players['10W']*point['10W']

    players['Overall Points'] = players['Batting Points'] + players['Bowling Points']
    df = players[['Player', 'Team', 'Batting Points', 'Bowling Points', 'Overall Points']]

    team = pd.DataFrame([], columns=['Player', 'Team', 'Batting Points', 'Bowling Points', 'Overall Points'])
    count = 1

    bat_rankings = df.sort_values(by='Batting Points', ascending=False)
    # taking top 3 batsmen on the basis of batting points
    for i in range(batsman):
        name = bat_rankings.iloc[i]['Player']
        team.loc[count] = bat_rankings.iloc[i]
        df.drop(bat_rankings.loc[bat_rankings['Player'] == name].index, inplace=True)
        count += 1

    bowl_rankings = df.sort_values(by='Bowling Points', ascending=False)
    # taking top 3 bowlers on the basis of bowling points
    for i in range(bowlers):
        name = bowl_rankings.iloc[i]['Player']
        team.loc[count] = bowl_rankings.iloc[i]
        df.drop(bowl_rankings.loc[bowl_rankings['Player'] == name].index, inplace=True)
        count += 1

    net_rankings = df.sort_values(by='Overall Points', ascending=False)
    # taking rest of the players on the basis of Overall points
    for i in range(11 - batsman - bowlers):
        name = net_rankings.iloc[i]['Player']
        team.loc[count] = net_rankings.iloc[i]
        df.drop(net_rankings.loc[net_rankings['Player'] == name].index, inplace=True)
        count += 1

    return team

def load_data(match_type, team1, team2, team3):
    print("Loading data...")  # Debugging
    
    bat_df = bat[bat['Type'] == match_type]
    bowl_df = bowl[bowl['Type'] == match_type]

    print("Bat DataFrame shape:", bat_df.shape)  # Debugging
    print("Bowl DataFrame shape:", bowl_df.shape)  # Debugging

    bat_data = bat_df[(bat_df['Team'] == team1) | (bat_df['Team'] == team2) | (bat_df['Team'] == team3)]
    bowl_data = bowl_df[(bowl_df['Team'] == team1) | (bowl_df['Team'] == team2) | (bowl_df['Team'] == team3)]

    print("Bat Data shape after filtering:", bat_data.shape)  # Debugging
    print("Bowl Data shape after filtering:", bowl_data.shape)  # Debugging

    players = pd.merge(bat_data, bowl_data, how='outer', on='Player')
    print("Merged DataFrame shape:", players.shape)  # Debugging

    # Handle missing values
    players.fillna(0, inplace=True)  # Replace missing values with zeros

    # Calculate Batting Points and Bowling Points
    
    
    return players


def file_download(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV file</a>'
    return href

def open_power_bi_file():
    power_bi_file_path = r"C:/Users/Kaustubh/Desktop/Player Prediction.pbix"
    if os.path.exists(power_bi_file_path):
        try:
            subprocess.Popen([power_bi_file_path], shell=True)
        except Exception as e:
            st.error(f"Error opening Power BI file: {e}")
    else:
        st.error("Power BI file not found!")

# Set up the Streamlit app
st.set_page_config(layout="wide")

col1 = st.sidebar

st.title("Player Prediction - A Data Web App")

details_expander = st.expander("App Details")
details_expander.markdown("""This app performs player prediction based on the player's past data of batting and bowling
- **Creator:** *Kaustubh Ayare, Yash Badgujar, Siddhi Ghodke, Tanisha Barot*
- **Data Source:** [cricbuzz](https://www.cricbuzz.com)
- **Guide:** Prof. Dilip Kale""")
points_expander = st.expander("Points Distribution")
points_expander.markdown("""
- **Runs:** Total Runs Scored (1 Point)
- **SR:** Batting Strike Rate
- **4s:** No. of fours (1 Point)
- **6s:** No. of Sixes (2 Points)
- **50:** No. of Half Centuries (8 Points)
- **100:** No. of Centuries (16 Points)
- **200:** No. of Double Centuries (32 Points)
- **Wkts:** No. of Wickets taken (25 Points)
- **Econ:** Bowler Economy Rate
- **5W:** No. of times bowler took 5 wickets in a single match (8 Points)
- **10W:** No. of times bowler took 10 wickets in a single match (16 Points)
""")

col1.header("User Input Features")
selected_type = col1.selectbox('Match Type', match_types, index=2)
team1 = col1.selectbox('Team 1', teams)
team2 = col1.selectbox('Team 2', teams)
team3 = col1.selectbox('Team 3', teams)
sort_type = col1.selectbox('Sort by', sort_types)
batsman = col1.slider('No. of Batsman', 1, 9, 3)
bowlers = col1.slider('No. of Bowlers', 1, (11 - batsman), 2)

col1.write("** Rest of the players will be selected based on their overall performances")

# load data of the team players
players = make_my_dream11(selected_type, team1, team2, team3, batsman, bowlers)

# Create buttons for navigation
nav_option = col1.radio("Navigation", ["Dream Team", "Machine Learning", "Compare Players", "Player Stats", "Graph", "Open Power BI File"])

if nav_option == "Dream Team":
    # Dream 11 Section
    st.header("DREAM TEAM")

    # Sort players DataFrame based on sort_type
    if sort_type == 'Batting':
        players = players.sort_values(by='Batting Points', ascending=False)
    elif sort_type == 'Bowling':
        players = players.sort_values(by='Bowling Points', ascending=False)
    elif sort_type == 'Overall':
        players = players.sort_values(by='Overall Points', ascending=False)

    # Display the predicted players in a table
    st.subheader("Predicted Dream 11 Team")
    st.table(players[['Player', 'Team', 'Batting Points', 'Bowling Points', 'Overall Points']])

elif nav_option == "Machine Learning":
    st.header("Machine Learning - Random Forest Model")

    # Train Random Forest model
    st.write("Training Random Forest model...")
    model, train_rmse, test_rmse, mean_cv_rmse, accuracy = train_random_forest_model(num_folds=5)
    st.write("Random Forest Model trained successfully!")

    # Display model information
    st.subheader("Model Information")
    st.write("Train RMSE:", train_rmse)
    st.write("Test RMSE:", test_rmse)
    st.write("Mean Cross-Validated RMSE:", mean_cv_rmse)
    st.write("Accuracy:", accuracy)

elif nav_option == "Compare Players":
    st.header("Compare Players")

    # Multiselect widget for selecting players to compare
    selected_team_players = pd.concat([bat, bowl])[(bat['Team'].isin([team1, team2, team3])) | (bowl['Team'].isin([team1, team2, team3]))]
    all_players = selected_team_players['Player'].unique()
    selected_players = st.multiselect('Select Players', all_players)

    # Filter data for selected players
    selected_players_stats = selected_team_players[selected_team_players['Player'].isin(selected_players)]

    # Display stats of selected players for each match type
    match_types = selected_players_stats['Type'].unique()
    for match_type in match_types:
        st.subheader(f"Stats for {match_type}")
        match_type_stats = selected_players_stats[selected_players_stats['Type'] == match_type]
        if not match_type_stats.empty:
            st.dataframe(match_type_stats)
        else:
            st.write(f"No data available for {match_type} matches.")

elif nav_option == "Player Stats":
    # Player Stats Section
    players_stats = load_data(selected_type, team1, team2, team3)
    st.markdown(f"## **{selected_type} Players Stats of selected Teams**")
    st.dataframe(players_stats)

    # Download CSV button
    st.markdown(file_download(players_stats), unsafe_allow_html=True)

elif nav_option == "Graph":
    # Graph Section
    plt.figure(figsize=(15, 10))  # Increase figure size to accommodate multiple plots
    plt.subplots_adjust(top=0.95, bottom=0.05, hspace=0.4)  # Adjust spacing between subplots

    # Plotting Batting Points
    plt.subplot(2, 1, 1)  # Creating subplot 1
    plt.barh(players['Player'], players['Batting Points'], color='skyblue', label='Batting Points')
    plt.xlabel('Batting Points')
    plt.ylabel('Player')
    plt.title('Batting Points of Players')
    plt.legend()

    # Plotting Bowling Points
    plt.subplot(2, 1, 2)  # Creating subplot 2
    plt.barh(players['Player'], players['Bowling Points'], color='lightgreen', label='Bowling Points')
    plt.xlabel('Bowling Points')
    plt.ylabel('Player')
    plt.title('Bowling Points of Players')
    plt.legend()

    # Show the plot
    st.pyplot(plt)


elif nav_option == "Open Power BI File":
    # Function to open Power BI file
    def open_power_bi_file():
        power_bi_file_path = "C:/Users/Kaustubh/Desktop/Player Prediction.pbix"  # Replace 'your_power_bi_file_name.pbix' with your actual Power BI file name
        if os.path.exists(power_bi_file_path):
            os.startfile(power_bi_file_path)
        else:
            st.error("Power BI file not found!")

    # Add button to open Power BI file
    if st.button("Open Power BI File"):
        open_power_bi_file()

