## trip.py

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from hmmlearn import hmm
from datetime import datetime
import trip_guess

# use SPY (S&P 500 ETF) for testing
data_set = "SPY"
# switch to taking this as an input in a function

# same here - eventually take start date as function input
# will need to determine end date programmatically
# 2021-01-01 to 2025-01-01 is 252*4=1008 days
# end_date = start_date + interval * 1008
start_date = "2021-01-01"
end_date = "2025-01-01"

# same here - eventually take start date as function input
interval = "1d"

# get data
data = yf.download(data_set, start=start_date, end=end_date, interval=interval)

# separate dataset into training and testing data
train_size = int(len(data) * 0.70)
train_data = data[:train_size]
test_data = data[train_size:]
# get the initial start and end dates of testing
test_start = test_data['Close'].iloc[0]
test_end = test_data['Close'].iloc[-1]

# we need features that define the "state" of the market
# common choices are returns and volatility
train_data['Returns'] = np.log(train_data['Close'] / train_data['Close'].shift(1))
train_data['Range'] = (train_data['High'] - train_data['Low']) / train_data['Close']
train_data.dropna(inplace=True)

# hmmlearn expects a 2D array of shape (n_samples, n_features)
X = train_data[['Returns', 'Range']].values

# number of market regimes
n_components = 2
# "full" allows features to correlate within a state
# "diag" allows features to be modeled w/o diagonal correlation
covariance_type = "diag"
# number of model iterations
n_iter = 100
# add min_covar to prevent "non-positive definite" error
min_covar=1e-3
# use Viterbi algorithm
algorithm = "viterbi"
# "" keeps set variables
# "stmc" reinitializes parameters each time 
init_params = "stmc"

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


# try volatility check
volatility_threshold = 0.070
low_volatility_regimes = np.where(model.means_[:, 1] < volatility_threshold)[0]

bull_regimes = []
for i in positive_return_regimes:
    if i in low_volatility_regimes:
        bull_regimes.append(i)

train_bull_state_list = []
# create a signal: 1 if in bullish state, 0 otherwise
# for now, grabbing any state with positive mean returns (not factoring in volatility)
train_data['Signal'] = np.where(train_data['State'].isin(bull_regimes), 1, 0)


# calculate returns on HMM
# We shift signal by 1 because we trade at the close based on today's state for tomorrow
train_data['Strategy_Returns'] = train_data['Signal'].shift(1) * train_data['Returns']

# calculate buy & hold returns
train_data['Cumulative_Market'] = np.exp(train_data['Returns'].cumsum())
train_data['Cumulative_Strategy'] = np.exp(train_data['Strategy_Returns'].cumsum())

market_final_train = train_data['Cumulative_Market'].iloc[-1]
strategy_final_train = train_data['Cumulative_Strategy'].iloc[-1]

# TODO: use final_train data to test how the model is doing; if it doesn't beat the market, consider looping

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

# calculate returns (shift by 1 to avoid look-ahead bias)
test_data['Strategy_Returns'] = test_data['Signal'].shift(1) * test_data['Returns']

# calculate cumulative growth
test_data['Cumulative_Market'] = np.exp(test_data['Returns'].cumsum())
test_data['Cumulative_Strategy'] = np.exp(test_data['Strategy_Returns'].cumsum())

market_final_test = test_data['Cumulative_Market'].iloc[-1]
strategy_final_test = test_data['Cumulative_Strategy'].iloc[-1]

# TODO: use (if strategy_final_test > market_final_test:) to determine whether or not to proceed

# next phase - rolling window and walk-forward

last_date = test_data.index[-1].strftime("%Y-%m-%d")
most_recent_date = datetime.today().strftime("%Y-%m-%d")

new_data = yf.download(data_set, start=last_date, end=most_recent_date)

# flatten MultiIndex columns if they exist
# if isinstance(new_data.columns, pd.MultiIndex):
#     new_data.columns = new_data.columns.get_level_values(0)

# combine with a bit of old data so the first "new" prediction has a training window
# use most recent trading year from old data

# drop the old 'Returns' and 'Range' columns from the tail of test_data 
# so they don't create NaN columns in the new_data section during concat
# then concatenate and remove duplicates (the overlapping last_date)
buffer_data = test_data.tail(252)[['Open', 'High', 'Low', 'Close', 'Volume']]
full_df = pd.concat([buffer_data, new_data])
full_df = full_df[~full_df.index.duplicated(keep='last')]

full_df['Returns'] = np.log(full_df['Close'] / full_df['Close'].shift(1))
full_df['Range'] = (full_df['High'] - full_df['Low']) / full_df['Close']
full_df.dropna(inplace=True)

# we'll do a 1-year rolling window
# 252 trading days in a year
window_size = 252 
signals = []
states = []
exception_list = []

for i in range(window_size, len(full_df)):
    X_train = full_df.iloc[i-window_size:i][['Returns', 'Range']].values
    current_features = full_df.iloc[i:i+1][['Returns', 'Range']].values
    
    try:
        model.fit(X_train)
        bull_indices = np.where(model.means_[:, 0] > 0)[0]
        current_state = model.predict(current_features)[0]
        signal = 1 if current_state in bull_indices else 0

        signals.append(signal)
        states.append(current_state)
    except Exception as e:
        # if model fails to converge, use signal from previous day
        print(f"Exception caught on window {i}! Exception: {e}")

        signals.append(signals[-1] if signals else 0)
        states.append(states[-1] if states else 0)
        exception_list.append(i)

        continue

# TODO: if exception_list is above margin of error, build new model (or try new time interval?)

# set new bullish states in case they've changed
positive_return_regimes = np.where(model.means_[:, 0] > 0)[0]
low_volatility_regimes = np.where(model.means_[:, 1] < volatility_threshold)[0]
bull_regimes = []
for i in positive_return_regimes:
    if i in low_volatility_regimes:
        bull_regimes.append(i)

# Add the signals to dataframe
full_results = full_df.copy()
new_results = full_results[window_size:]
new_results['Signal'] = signals
new_results['State'] = states

new_results['Strategy_Returns'] = new_results['Signal'].shift(1) * new_results['Returns']
new_results['Cumulative_Market'] = np.exp(new_results['Returns'].cumsum())
new_results['Cumulative_Strategy'] = np.exp(new_results['Strategy_Returns'].cumsum())

market_final = new_results['Cumulative_Market'].iloc[-1]
strategy_final = new_results['Cumulative_Strategy'].iloc[-1]

# get "control group" of random guesses
new_results_guesses = trip_guess.guess_list(signals, n_components)
new_results['Guesses'] = new_results_guesses

# compare with signal
guess_score = 0
for item in range(len(signals)):
    if signals[item] == new_results_guesses[item]:
        guess_score += 1
win_rate = guess_score / len(signals)

print(f"Guessing win rate: {win_rate}")

# get the state for most recent time interval
# use iloc[-1:] to get the latest data point
most_recent_features = new_results.iloc[-1:][['Returns', 'Range']].values
most_recent_state = model.predict(most_recent_features)[0]

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
end_date_range = 5
print(f"\nMarket: {data_set}")
print("|--- Date ---|-- State --|---Bull?---|")
for i in range(0, end_date_range):
    # reverse index to go in order of dates, from -10 to -1
    index = end_date_range - i
    print_date = new_results.index[-index].strftime("%Y-%m-%d")
    print_state = new_results['State'].iloc[-index]
    print(f"| {print_date} |     {print_state}     |    {"Yes" if print_state in bull_regimes else "No "}    |") # formatting
print("|------------|-----------|-----------|")

print(f"Today's state: {most_recent_state}")
print("Probabilities for tomorrow:")

for i in range(0, probs_for_next_state.size):
    print(f"  State {i}: {probs_for_next_state[i]:.2%}")

print(f"Predicted state for {data_set} tomorrow: {next_predicted_state}")
print(f"Action for {data_set} Tomorrow: {'🚀 BUY BUY BUY' if is_bullish else '💰 SELL SELL SELL'}")
