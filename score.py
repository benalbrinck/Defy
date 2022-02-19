import collections
import defy_logging
import defy_model
import json
import yaml
import numpy as np
from sportsreference.ncaab.conferences import Conference


def get_conference_teams():
    """Get all teams for each conference in the current year"""
    conference_teams = []

    for conference_name in conference_names:
        conference = Conference(conference_name.lower(), str(year))
        conference_teams.append(list(conference.teams))
    
    return conference_teams


def simulate_game(team1, team2):
	"""Simulate a game between the two teams."""
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

	# You can also add in flipped data

	# Get probabilities
	result = model(normal_game_input).numpy()

	return {
		team1: result[0][0],
		team2: result[0][1]
	}


def simulate_combinations(first_teams, second_teams):
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


def simulate_round(teams, correct_teams):
	"""Simulate each game in the round."""
	results = []
	result_probs = []

	for game in range(int(len(teams) / 2)):
		result = simulate_combinations(teams[game * 2], teams[(game * 2) + 1])
		results.append(result)
		result_probs.append(result[correct_teams[game]])

	return results, result_probs


def display_results(round_prob):
	for i, p in enumerate(round_prob):
		logger.info(f'{i}, {round(p * 100, 2)}%')


if __name__ == '__main__':
	logger = defy_logging.get_logger()

	with open('setup/conferences.txt') as file:
		conference_names = file.read().split('\n')
	
	with open('setup/removed_players.txt') as file:
		removed_players = file.read().split('\n')
	
	with open('setup/true_results.txt') as file:
		true_results = [r.split('\n') for r in file.read().split('\n\n')]

	with open('setup/config.yml') as file:
		config = yaml.safe_load(file)

	year = config['global']['start_year']
	checkpoint_path = config['simulate']['checkpoint_path']

	# Get all conference teams and team data
	# logger.info('Get conference teams...')
	with open(f'data/conferences_{year}.json') as file:
		conference_teams = json.load(file)

	logger.info('Get team data...')
	with open(f'data/teams_{year}.json') as file:
		teams = json.load(file)

	# Get teams for tournament
	logger.info('Getting tournament teams and normalization value...')
	with open('setup/tournament_teams.txt') as file:
		split_text = file.read().split('\n\n')
	
	first_four_teams = []
	tournament_teams = []

	for t in split_text[0].split('\n'):
		first_four_teams.append({ t: 1 })
	
	for t in split_text[1].split('\n'):
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

	# String to record results
	results_string = ''

	# First Four
	logger.info('First Four:')
	first_four_results, first_four_prob = simulate_round(first_four_teams, true_results[0])
	display_results(first_four_prob)
	results_string += '\n'.join([str(p) for p in first_four_prob]) + '\n\n'

	# Rest of tournament
	# Insert first four into the main tournament
	expected_score = 0
	first_four_index = 0

	for i, t in enumerate(tournament_teams):
		if list(t.keys())[0] == '{}':
			tournament_teams[i] = first_four_results[first_four_index]
			first_four_index += 1

	# Simulate main tournament
	for tournament_round in range(6):
		logger.info(f'Round {tournament_round + 1}:')
		round_results, round_prob = simulate_round(tournament_teams, true_results[tournament_round + 1])
		display_results(round_prob)

		# Calculate running probabilities and expected score
		expected_score += sum([p * ((2 ** tournament_round) * 10) for p in round_prob])
		
		# Record results and repeat
		results_string += '\n'.join([str(p) for p in round_prob]) + '\n\n'
		tournament_teams = round_results[:]

	logger.info(f'Expected score: {expected_score}')
	logger.info(tournament_teams)

	results_file_name = checkpoint_path.replace('networks/', '')[:-5]

	with open(f'results/{results_file_name}_probs.txt', 'w') as file:
		file.write(results_string)
