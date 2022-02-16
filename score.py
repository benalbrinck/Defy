
# Probability: literally just all the things multiplied together
# Expected value: I think it is the prob that the matchup happens * prob that it picked correctly
#   * the points you would get for that round all added up

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

	return result[0]


def simulate_round(teams, correct_teams):
    """Simulate each game in the round."""
    result_probs = []

    for game in range(int(len(teams) / 2)):
        result_prob = simulate_game(teams[game * 2], teams[(game * 2) + 1])

        if teams[game * 2] == correct_teams[game]:
            # First team wins
            result_probs.append(result_prob[0])
        else:
            result_probs.append(result_prob[1])

    return result_probs


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
	
	first_four_teams = split_text[0].split('\n')
	tournament_string = split_text[1] + ''

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
	first_four_prob = simulate_round(first_four_teams, true_results[0])
	display_results(first_four_prob)
	results_string += '\n'.join([str(p) for p in first_four_prob]) + '\n\n'
	first_four_split = true_results[0]

	# Rest of tournament
	# Insert first four into the main tournament
	expected_score = 0
	tournament_teams = tournament_string.format(*first_four_split).split('\n')
	tournament_probs = []

	first_four_index = 0
	for t in tournament_string.split('\n'):
		if t == '{}':
			tournament_probs.append(first_four_prob[first_four_index])
			first_four_index += 1
		else:
			tournament_probs.append(1)

	# Simulate main tournament
	for tournament_round in range(6):
		logger.info(f'Round {tournament_round + 1}:')
		round_prob = simulate_round(tournament_teams, true_results[tournament_round + 1])
		display_results(round_prob)

		# Calculate running probabilities and expected score
		tournament_probs = [round_prob[p] * (tournament_probs[p * 2] * tournament_probs[(p * 2) + 1]) for p in range(len(round_prob))]
		expected_score += sum([p * ((2 ** tournament_round) * 10) for p in tournament_probs])
		
		# Record results and repeat
		results_string += '\n'.join([str(p) for p in round_prob]) + '\n\n'
		tournament_teams = true_results[tournament_round + 1]

	logger.info(f'Probability of picking perfectly: {tournament_probs[0] * 100}%')
	logger.info(f'Expected score: {expected_score}')

	results_file_name = checkpoint_path.replace('networks/', '')[:-5]

	with open(f'results/{results_file_name}_probs.txt', 'w') as file:
		file.write(results_string)
