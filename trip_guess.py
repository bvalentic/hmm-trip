# trip but it's a random guess each time
# i want to Monte Carlo style it and get a high score / win rate to compare
# against a regular trip score

import random

def guess_list(data_frame, n_components = 2):
    random.seed()
    length = len(data_frame)
    guesses = []
    for i in range(length):
        rng = random.randrange(0, n_components)
        guesses.append(rng)
    return guesses

def guess_mc(data_frame, mc_count, n_components = 2):
    random.seed()
    guess_lists = []
    winning_guess = 0
    winning_score = 0
    for sim in range(mc_count):
        guesses = guess_list(data_frame, n_components)
        guess_score = 0
        for item in range(len(data_frame)):
            if data_frame[item] == guesses[item]:
                guess_score += 1
            else:
                guess_score -= 1
        win_rate = guess_score / len(data_frame)
        if guess_score > winning_score:
            winning_guess = sim
            winning_score = guess_score
        guess_tuple = (sim, guesses, guess_score, win_rate)
        guess_lists.append(guess_tuple)
    return guess_lists[winning_guess]
