"""Generates probabilities for every game in the tournament and calculates the average
score in the ESPN Tournament Challenge using the model created from train.py.
"""


import defy_logging
import defy_model
import json
import os
import pickle
import train
import yaml
import numpy as np
from sportsreference.ncaab.conferences import Conference


def get_conference_teams() -> list:
    """Get all teams for each conference in the current year.
	
	Returns:
		conference_teams (list): list of lists of teams for each conference
	"""
    conference_teams = []

    for conference_name in conference_names:
        conference = Conference(conference_name.lower(), str(year))
        conference_teams.append(list(conference.teams))
    
    return conference_teams


def simulate_game(team1: str, team2: str) -> dict:
	"""Simulate a game between the two teams.
	
	Parameters:
		team1 (str): one of the teams playing
		team2 (str): one of the teams playing

	Returns:
		game (dict): dictionary of both teams and the probability that either will win
	"""
	global model

	# Get each team's stats
	team1_players = teams[team1.upper()]
	team2_players = teams[team2.upper()]

	team1_game_roster = [team1_players[p] for p in list(team1_players) if p not in removed_players][:12]
	team2_game_roster = [team2_players[p] for p in list(team2_players) if p not in removed_players][:12]

	for i in range(12 - len(team1_game_roster)):
		team1_game_roster.append(np.zeros_like(team1_game_roster[0]))
	
	for i in range(12 - len(team2_game_roster)):
		team2_game_roster.append(np.zeros_like(team2_game_roster[0]))

	team1_stats = np.concatenate(team1_game_roster, axis=1)
	team2_stats = np.concatenate(team2_game_roster, axis=1)

	# Get each team's conference id
	team1_conference = np.expand_dims(np.array([int(team1.lower() in c) for c in conference_teams]), axis=0)
	team2_conference = np.expand_dims(np.array([int(team2.lower() in c) for c in conference_teams]), axis=0)

	# Concatenate stats, fix stats, normalize, and put it through the model
	game_input = np.concatenate((team1_stats, team1_conference, team2_stats, team2_conference), axis=1)
	game_input = np.nan_to_num(game_input.astype('float32'))

	normal_game_input = np.divide(game_input, input_norm, out=np.zeros_like(game_input), where=input_norm!=0)
	normal_game_input_flip = train.flip_data(normal_game_input)

	# Get probabilities
	result = model(normal_game_input).numpy()
	result_flip = model(normal_game_input_flip).numpy()

	return {
		team1: (result[0][0] + result_flip[0][1]) / 2,
		team2: (result[0][1] + result_flip[0][0]) / 2
	}


def simulate_combinations(first_teams: dict, second_teams: dict) -> dict:
	"""Simulate every matchup between two teams and return probabilities.
	
	Parameters:
		first_teams (dict): holds each first team in the matchup and the probability 
			that the team will make it to the matchup
		second_teams (dict): same as first_teams, but for the second team in the matchup
	
	Returns:
		results (dict): probability that each team in first_teams and second_teams will
			win the matchup
	"""
	first_team_keys = list(first_teams.keys())
	second_team_keys = list(second_teams.keys())
	results = {}

	for f in first_team_keys:
		if f not in results:
			results[f] = 0

		for s in second_team_keys:
			if s not in results:
				results[s] = 0

			result_prob = simulate_game(f, s)
			result_prob = {k: result_prob[k] * first_teams[f] * second_teams[s] for k in result_prob}
			
			results[f] += result_prob[f]
			results[s] += result_prob[s]
	
	return results


def simulate_round(teams: list, correct_teams: list=[]) -> tuple[list, list]:
	"""Simulate each game in the round.
	
	Parameters:
		teams (list): list of dictionaries of teams in the round. Each dictionary has all
			teams that can reach that round, and the probability that they will reach that 
			round. Every two teams in the list will play each other.
		correct_teams (list), default []: the correct results of the round. If set, 
			result_probs will be set for each game to be the probability that the correct 
			team will win. Otherwise, result_probs will be set to 1 for each game.
	
	Returns:
		results (list): the list of dictionaries of teams after simulating the round.
		result_probs (list): see correct_teams.
	"""
	results = []
	result_probs = []

	for game in range(int(len(teams) / 2)):
		result = simulate_combinations(teams[game * 2], teams[(game * 2) + 1])

		# Low-pass filter
		result = {k: result[k] if result[k] > lpf else 0 for k in result}
		result_sum = sum(list(result.values()))
		result = {k: result[k] / result_sum for k in result}

		# Record results
		results.append(result)

		if check_results:
			result_probs.append(result[correct_teams[game]])
		else:
			result_probs.append(1)

	return results, result_probs


def display_results(round_prob: list) -> None:
	"""Use logger to display results of a round.
	
	Parameters:
		round_prob (list): the list of probabilities for the round.
	"""
	for i, p in enumerate(round_prob):
		logger.info(f'{i}, {round(p * 100, 2)}%')


if __name__ == '__main__':
	logger = defy_logging.get_logger()
	
	with open('setup/config.yml') as file:
		config = yaml.safe_load(file)

	year = config['global']['start_year']
	check_year = config['global']['check_year']
	check_results = config['global']['check_results']

	use_removed_players = config['simulate']['use_removed_players']
	lpf = config['simulate']['lpf_score']

	checkpoint_path = config['simulate']['checkpoint_path']
	use_epoch = config['simulate']['use_epoch']
	checkpoint_path += '/{epoch:04d}.hdf5'.format(epoch=use_epoch)
	
	with open(f'setup/conferences/{check_year}.txt') as file:
		conference_names = file.read().split('\n')
	
	if use_removed_players:
		with open(f'setup/removed_players/{check_year}.txt') as file:
			removed_player_rounds = {p.split('\t')[0]: int(p.split('\t')[1]) for p in file.read().split('\n')}
	else:
		removed_player_rounds = {}
	
	if check_results:
		with open(f'setup/true_results/{check_year}.txt') as file:
			true_results = [r.split('\n') for r in file.read().split('\n\n')]

	# Get all conference teams and team data
	logger.info('Get conference teams...')
	with open(f'data/conferences_{year}.json') as file:
		conference_teams = json.load(file)

	logger.info('Get team data...')
	with open(f'data/teams_{year}.json') as file:
		teams = json.load(file)

	# Get teams for tournament
	logger.info('Getting tournament teams and normalization value...')
	with open(f'setup/tournament_teams/{check_year}.txt') as file:
		split_text = file.read().split('\n')
	
	tournament_teams = []

	for t in split_text:
		tournament_teams.append({ t: 1 })

	# Get normalization value
	npz_file = np.load(f'data/data_{year}.npz', allow_pickle=True)
	inputs = np.nan_to_num(npz_file['inputs'].astype('float32'))
	input_norm = np.linalg.norm(inputs, axis=0)

	# Load model
	logger.info('Loading network...')

	with open(f'{checkpoint_path[:-5]}.temp') as file:
		temperature = float(file.read())

	model = defy_model.get_model(simulate=True, temperature=temperature)
	model.load_weights(checkpoint_path)

	# Variables to record results
	results_probs = []
	expected_score = 0
	variance = 0

	# Simulate main tournament
	for tournament_round in range(6):
		logger.info(f'Round {tournament_round + 1}:')

		# Get removed players and score round
		removed_players = [p for p in removed_player_rounds if removed_player_rounds[p] < tournament_round]

		if check_results:
			round_results, round_prob = simulate_round(tournament_teams, true_results[tournament_round])
		else:
			round_results, round_prob = simulate_round(tournament_teams)
		
		display_results(round_prob)

		# Calculate running probabilities, variances, and expected score
		round_score = ((2 ** tournament_round) * 10)
		round_means = [p * round_score for p in round_prob]

		correct_variance = [(round_score - m)**2 * p for p, m in zip(round_prob, round_means)]
		incorrect_variance = [m**2 * (1 - p) for p, m in zip(round_prob, round_means)]
		
		expected_score += sum(round_means)
		variance += sum(correct_variance) + sum(incorrect_variance)
		
		# Record results and repeat
		results_probs.append(round_results[:])
		tournament_teams = round_results[:]

	logger.info(f'Expected score: {expected_score}')
	logger.info(f'Standard deviation: {variance ** (1/2)}')
	results_file_name = checkpoint_path.replace('networks/', '')[:-5]

	if not os.path.exists(f'results/{results_file_name[:-4]}'):
		os.makedirs(f'results/{results_file_name[:-4]}')

	with open(f'results/{results_file_name}_probs.txt', 'wb') as file:
		pickle.dump(results_probs, file)
