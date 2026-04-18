## day_trip.py - day tripper
# use this to get the day's market regime

import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn import hmm
from datetime import datetime
import matplotlib.pyplot as plt
import trip_guess as guess
import time

# use SPY (S&P 500 ETF) for testing
data_set = "SPY"
# switch to taking this as an input in a function

start_date = "2020-01-01"
end_date = "2024-04-01"

# for day trip, interval should always be 1 day
interval = "1d"

# TODO: take start date as function input
# end_date_count = 1000

# start_datetime = datetime.date(start_date)

# end_date = start_date + datetime.timedelta(days=end_date_count)

# get data
data = yf.download(data_set, start=start_date, end=end_date, interval=interval)

# separate dataset into training and testing data
train_size = int(len(data) * 0.70)
train_data = data[:train_size].copy()
test_data = data[train_size:].copy()

# more data
last_date = test_data.index[-1].strftime("%Y-%m-%d")
most_recent_date = datetime.today().strftime("%Y-%m-%d")
new_data = yf.download(data_set, start=last_date) #end=most_recent_date?

# features that define the "state" of the market
# using returns and volatility:
train_data['Returns'] = np.log(train_data['Close'] / train_data['Close'].shift(1))
train_data['Range'] = (train_data['High'] - train_data['Low']) / train_data['Close']
train_data.dropna(inplace=True)

# hmmlearn expects a 2D array of shape (n_samples, n_features)
X = train_data[['Returns', 'Range']].values

# number of market regimes
n_components = 2
# "spherical" - each state uses a single variance value that applies to all features (default)
# "diag" - each state uses a diagonal covariance matrix
# "full" - each state uses a full (i.e. unrestricted) covariance matrix
# (originally said it 'allows features to correlate within a state')
# "tied" - all states use the same full covariance matrix
covariance_type = "full"
# number of model iterations
n_iter = 100
# add min_covar to prevent "non-positive definite" error
min_covar=1e-4
# use Viterbi algorithm
algorithm = "viterbi"
# "" keeps set variables
# "stmc" reinitializes parameters each time 
init_params = "stmc"

# variables for looping
model_number = 0
max_model_count = 32
model_list = []
score_list = []
win_rate_list = []
signals_and_states = []

# before loop, start time
start = time.time()

# loop the model and select the one with the best win rate
# score is used as a comparison, but win % is the most important
while (model_number < max_model_count):

    # create model
    model = hmm.GaussianHMM(
        n_components=n_components, 
        covariance_type=covariance_type,
        min_covar=min_covar,
        n_iter=n_iter, 
        algorithm=algorithm,
        init_params=init_params
    )
    model.fit(X)

    # model estimates which "hidden state" generated the data for each day
    hidden_states = model.predict(X)

    # add states back to the dataframe for analysis
    train_data['State'] = hidden_states

    positive_return_regimes = np.where(model.means_[:, 0] > 0)[0]

    # use volatility check
    volatility_threshold = 0.070
    low_volatility_regimes = np.where(model.means_[:, 1] < volatility_threshold)[0]

    bull_regimes = []
    for i in positive_return_regimes:
        if i in low_volatility_regimes:
            bull_regimes.append(i)

    train_bull_state_list = []
    # create a signal: 1 if in bullish state, 0 otherwise
    train_data['Signal'] = np.where(train_data['State'].isin(bull_regimes), 1, 0)

    # get the transitional matrix for the final state
    train_features_final = train_data.iloc[-1:][['Returns', 'Range']].values
    train_prediction_final = model.predict(train_features_final)[0]
    train_transmat_final = model.transmat_[train_prediction_final]
    # get largest state, to see if first index of new model.predict matches 
    train_predicted_chance_final = train_transmat_final.max()
    train_predicted_state_final = np.where(train_transmat_final == train_predicted_chance_final)[0]

    # prepare the test features (must be the same columns as training)
    test_data['Returns'] = np.log(test_data['Close'] / test_data['Close'].shift(1))
    test_data['Range'] = (test_data['High'] - test_data['Low']) / test_data['Close']
    test_data.dropna(inplace=True)
    X_test = test_data[['Returns', 'Range']].values

    # predict uses the existing model parameters to predict the next state
    test_states = model.predict(X_test)
    test_data['State'] = test_states

    # add to dataframe and calculate returns
    test_data = test_data.copy() # Avoid SettingWithCopyWarning
    # reset bull market signal using new data
    positive_return_regimes = np.where(model.means_[:, 0] > 0)[0]
    low_volatility_regimes = np.where(model.means_[:, 1] < volatility_threshold)[0]
    bull_regimes = []
    for i in positive_return_regimes:
        if i in low_volatility_regimes:
            bull_regimes.append(i)

    test_data['Signal'] = np.where(test_data['State'].isin(bull_regimes), 1, 0)

    # next phase - rolling window and walk-forward
    # roll up to present day; 
    # guess latest regime for most recent market close; 
    # compare with actual results for a final test.
    # then apply model to the next day?

    # we'll do a 1-year rolling window for the day-trip
    # 252 trading days in a year
    window_size = 252 

    # flatten MultiIndex columns if they exist
    # if isinstance(new_data.columns, pd.MultiIndex):
    #     new_data.columns = new_data.columns.get_level_values(0)

    # combine with a bit of old data so the first "new" prediction has a training window
    # use most recent trading year from old data

    # drop the old 'Returns' and 'Range' columns from the tail of test_data 
    # so they don't create NaN columns in the new_data section during concat
    # then concatenate and remove duplicates (the overlapping last_date)
    buffer_data = test_data.tail(window_size)[['Open', 'High', 'Low', 'Close', 'Volume']]
    full_df = pd.concat([buffer_data, new_data])
    full_df = full_df[~full_df.index.duplicated(keep='last')]

    full_df['Returns'] = np.log(full_df['Close'] / full_df['Close'].shift(1))
    full_df['Range'] = (full_df['High'] - full_df['Low']) / full_df['Close']
    full_df.dropna(inplace=True)

    # now before rolling window, check the model's prediction
    # get the transitional matrix for the final state
    test_features_final = test_data.iloc[-1:][['Returns', 'Range']].values
    test_prediction_final = model.predict(test_features_final)[0]
    test_transmat_final = model.transmat_[test_prediction_final]
    # get largest state, to see if first index of new model.predict matches 
    test_predicted_chance_final = test_transmat_final.max()
    test_predicted_state_final = np.where(test_transmat_final == test_predicted_chance_final)[0][0]

    run_count = 0
    signals = []
    states = []
    exception_list = []

    # use prediction from test data
    current_predicted_high_chance = test_predicted_chance_final 
    current_predicted_index = test_predicted_state_final 
    model_score = 0
    correct_predictions = 0

    for i in range(window_size, len(full_df)):
        run_count += 1
        X_train = full_df.iloc[i-window_size:i][['Returns', 'Range']].values
        current_features = full_df.iloc[i:i+1][['Returns', 'Range']].values
        
        try:
            model.fit(X_train)
            current_state = model.predict(current_features)[0]

            positive_return_regimes = np.where(model.means_[:, 0] > 0)[0]
            low_volatility_regimes = np.where(model.means_[:, 1] < volatility_threshold)[0]
            bull_regimes = []
            for regime in positive_return_regimes:
                if regime in low_volatility_regimes:
                    bull_regimes.append(regime)

            signal = 1 if current_state in bull_regimes else 0
            signals.append(signal)
            states.append(current_state)

            # "score" model based on whether or not prediction is correct
            # using percentage like a 0-100 confidence scale
            if current_state == current_predicted_index:
                model_score += current_predicted_high_chance
                correct_predictions += 1
            else:
                model_score -= current_predicted_high_chance
            # set next values to "current"
            current_transmat = model.transmat_[current_state]
            current_predicted_high_chance = current_transmat.max()
            current_predicted_index = np.where(current_transmat == current_predicted_high_chance)[0][0]

        except Exception as e:
            # if model fails to converge, use signal from previous day
            print(f"Exception caught on window {i}! Exception: {e}")

            signals.append(signals[-1] if signals else 0)
            states.append(states[-1] if states else 0)
            exception_list.append(i)
            continue

    win_rate = correct_predictions / run_count
    model_list.append(model)
    score_list.append(model_score)
    win_rate_list.append(win_rate)
    signals_and_states.append((signals, states))

    # add model number and continue loop
    model_number += 1

end = time.time()

# determine winningest model and use that one
winning_model_number = np.argmax(win_rate_list)
winning_model = model_list[winning_model_number]
model = winning_model
signals = signals_and_states[winning_model_number][0]
states = signals_and_states[winning_model_number][1]

print(f"\nRun time: {end - start:.2f}s")

print(f"\nWinning model: {winning_model_number}")
print(f"High score: {score_list[winning_model_number]:.2f}")
print(f"High win rate: {win_rate_list[winning_model_number]:.2%}")

# set new bullish states in case they've changed
positive_return_regimes = np.where(model.means_[:, 0] > 0)[0]
low_volatility_regimes = np.where(model.means_[:, 1] < volatility_threshold)[0]
bull_regimes = []
for i in positive_return_regimes:
    if i in low_volatility_regimes:
        bull_regimes.append(i)

# Add the signals and states to dataframe
full_results = full_df.copy()
new_results = full_results[window_size:]
new_results['Signal'] = signals
new_results['State'] = states

# try Monte Carlo guessing method and see what the best guess is

# function returns winning (sim, guesses, guess_score, win_rate)
guess_tuples = guess.guess_mc(states, max_model_count * 10, n_components)

print(f"\nWinning guess model: {guess_tuples[0]}")
print(f"Winning guess rate: {guess_tuples[3]:.2%}")
print(f"Winning guess score: {guess_tuples[2]}")

# get the state for most recent time interval
# use iloc[-1:] to get the latest data point
most_recent_features = new_results.iloc[-1:][['Returns', 'Range']].values
most_recent_state = model.predict(most_recent_features)[0]
# most_recent_state = model.predict(most_recent_features)[-1]

# access the transition matrix
# a matrix of [Current State, Next State] probabilities
# shape is (n_components, n_components)
transition_matrix = model.transmat_

# find the most likely next state
probs_for_next_state = transition_matrix[most_recent_state]
next_predicted_state = np.argmax(probs_for_next_state)

# check if future state is bullish
is_bullish = 1 if next_predicted_state in bull_regimes else 0

# leaving this for now so that I have some idea of what's going on
print("\nMeans and variances of each state:")
for i in range(model.n_components):
    print(f"State {i}{" (Bullish)" if i in bull_regimes else ""}:")
    print(f"  Mean Returns: {model.means_[i][0]:.5f}")
    print(f"  Mean Volatility: {model.means_[i][1]:.5f}")

# table of most recent dates and states
end_date_range = 10
print(f"\nMarket: {data_set}")
print("|--- Date ---|-- State --|---Bull?---|")
for i in range(0, end_date_range):
    # reverse index to go in order of dates, from -10 to -1
    index = end_date_range - i
    print_date = new_results.index[-index].strftime("%Y-%m-%d")
    print_state = new_results['State'].iloc[-index]
    print(f"| {print_date} |     {print_state}     |    {"Yes" if print_state in bull_regimes else "No "}    |") # formatting
print("|------------|-----------|-----------|\n")

print(f"Most recent date used: {new_results.index[-1].strftime("%Y-%m-%d")}")
print(f"Model prediction of most recent state: {most_recent_state}")
print("Probabilities for tomorrow:")

for i in range(0, probs_for_next_state.size):
    print(f"  State {i}{" (Bullish)" if i in bull_regimes else ""}: {probs_for_next_state[i]:.2%}")

print(f"Predicted state for {data_set} tomorrow: {next_predicted_state}")
print(f"Action for {data_set} Tomorrow: {'🚀 BUY BUY BUY' if is_bullish else '💰 SELL SELL SELL'}")

# plot heatmap of transmat
plt.imshow(transition_matrix, aspect='auto', cmap='YlOrRd')
plt.title('Generated Transition Matrix')
plt.xticks([0, 1])
plt.xlabel('State To')
plt.yticks([0, 1])
plt.ylabel('State From')
plt.show()
