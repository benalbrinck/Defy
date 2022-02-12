import defy_logging
import json
import yaml
import numpy as np
import tensorflow as tf
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

	# Pick index of the winner based on probabilities from result
	result = model(normal_game_input).numpy()
	# max_index = np.random.choice(np.array([0, 1]), p=np.squeeze(result))
	max_index = int(result[0][1] > result[0][0])

	return max_index, result


def simulate_round(teams):
	"""Simulate each game in the round."""
	result_teams = []
	result_arrays = []

	for game in range(int(len(teams) / 2)):
		result_index, result_array = simulate_game(teams[game * 2], teams[(game * 2) + 1])
		result_teams.append(teams[(game * 2) + result_index])
		result_arrays.append(result_array)
	
	return result_teams, result_arrays


if __name__ == '__main__':
	logger = defy_logging.get_logger()

	with open('setup/conferences.txt') as file:
		conference_names = file.read().split('\n')
	
	with open('setup/removed_players.txt') as file:
		removed_players = file.read().split('\n')

	with open('setup/config.yml') as file:
		config = yaml.safe_load(file)

	year = config['global']['start_year']
	activation_function = config['train']['activation_function']
	checkpoint_path = config['simulate']['checkpoint_path']

	# Get all conference teams and team data
	logger.info('Get conference teams...')
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
	model = tf.keras.models.Sequential([
		tf.keras.layers.Dense(128, activation=activation_function, input_shape=(1168,)),
		tf.keras.layers.Dropout(0.1),
		tf.keras.layers.Dense(32, activation=activation_function),
		tf.keras.layers.Dropout(0.1),
		tf.keras.layers.Dense(2, activation='softmax')
	])
	loss_function = tf.keras.losses.BinaryCrossentropy()
	model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])
	model.load_weights(checkpoint_path)

	# String to record results
	results_string = ''

	# First Four
	logger.info('First Four:')
	first_four_result, first_four_arrays = simulate_round(first_four_teams)
	first_four_display = '\n'.join([f'{r}, {first_four_arrays[i]}' for i, r in enumerate(first_four_result)])
	logger.info(first_four_display + '\n')
	results_string += '\n'.join(first_four_result) + '\n\n'
	first_four_split = first_four_result

	# Rest of tournament
	# Insert first four into the main tournament
	tournament_teams = tournament_string.format(*first_four_split).split('\n')

	# Simulate main tournament
	for tournament_round in range(6):
		logger.info(f'Round {tournament_round + 1}:')
		round_result, round_arrays = simulate_round(tournament_teams)
		round_display = '\n'.join([f'{r}, {round_arrays[i]}' for i, r in enumerate(round_result)])
		logger.info(round_display + '\n')

		# Record results and repeat
		results_string += '\n'.join(round_result) + '\n\n'
		tournament_teams = round_result

	results_file_name = checkpoint_path.replace('networks/', '')[:-5]

	with open(f'results/{results_file_name}.txt', 'w') as file:
		file.write(results_string)

	print('\n\n\n\n')
	import check
