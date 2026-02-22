# HMM-TRIP: Hidden Markov Model - Train, Roll, Infer, Predict

trip.py is a program that will build a Hidden Markov Model, train it, test it, use a rolling window to infer market regimes, and predict the next market state.
In short, trippy tells you if tomorrow will be a good or bad day for the market.

## Goals

I want trippy to take inputs of the market/ticker, start date, interval, and return the next market state corresponding to that interval; e.g. if I input BTC-USD, 1 month ago, and 5m, I want trippy to return the predicted BTC-USD market regime for the next 5 minutes.

## Getting Started

### Create Virtual Env and Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

### Run Program

```bash
python trip.py
```
