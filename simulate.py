import numpy as np
import tensorflow as tf

# Checkpoint parameters
checkpoint_path = 'network.ckpt'

# Get all current team names and stats
team_names = open('current_teams.txt').read().split('\n')
npz_file = np.load('current_team_stats.npz', allow_pickle=True)
team_stats = npz_file['team_stats'].astype('float32')
print(team_stats.shape)

# Get teams for tournament
split_text = open('tournament_teams.txt').read().split('\n\n')
first_four_teams = split_text[0].split('\n')
tournament_string = split_text[1] + ''

# Get normalization value
npz_file = np.load('data.npz', allow_pickle=True)

inputs = npz_file['inputs'].astype('float32')
# outputs = npz_file['outputs_results'].astype('float32')
outputs = npz_file['outputs_scores'].astype('float32')
output_not_nan = [i for i, r in enumerate(np.isnan(outputs).any(axis=1)) if r]

inputs = [x for i, x in enumerate(inputs) if i not in output_not_nan]
outputs = [x for i, x in enumerate(outputs) if i not in output_not_nan]

input_norm = np.linalg.norm(inputs, axis=0)
output_norm = np.linalg.norm(outputs, axis=0)  # Only do if doing scores

# Load model
activation_function = 'relu'
model = tf.keras.models.Sequential([
	tf.keras.layers.Dense(64, activation=activation_function, input_shape=(234,)),
	tf.keras.layers.Dropout(0.1),
	tf.keras.layers.Dense(16, activation=activation_function),
	tf.keras.layers.Dropout(0.1),
	# tf.keras.layers.Dense(2, activation='softmax')
	tf.keras.layers.Dense(2, activation='tanh')
])
loss_function = tf.keras.losses.BinaryCrossentropy()
model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])
model.load_weights(checkpoint_path)

def simulate_game(team1, team2):
	"""Simulate a game between the two teams."""
	global model

	# Find each team stats
	team1_index = team_names.index(team1)
	team2_index = team_names.index(team2)

	team1_stats = np.expand_dims(team_stats[team1_index], axis=0)
	team2_stats = np.expand_dims(team_stats[team2_index], axis=0)

	# Concatenate stats and put it through the model
	normal_input = np.concatenate((team1_stats, team2_stats), axis=1)
	normal_result = model(np.divide(normal_input, input_norm, out=np.zeros_like(normal_input), where=input_norm!=0)).numpy()
	
	# flip_input = np.concatenate((team2_stats, team1_stats), axis=1)
	# flip_result = np.flip(model(np.divide(flip_input, input_norm, out=np.zeros_like(flip_input), where=input_norm!=0)).numpy(), axis=1)

	# result = (normal_result + flip_result) / 2
	result = normal_result

	# Pick index of the winner based on probabilities from result
	# max_index = np.random.choice(np.array([0, 1]), p=np.squeeze(result))
	max_index = int(result[0][1] > result[0][0])

	# Get score of the game
	score = result * output_norm

	return max_index, score[0]

def simulate_round(teams):
	"""Simulate each game in the round."""
	result_indices = []
	result_scores = []
	for game in range(int(len(teams) / 2)):
		result_index, result_score = simulate_game(teams[game * 2], teams[(game * 2) + 1])
		
		result_indices.append(teams[(game * 2) + result_index])
		result_scores.append(result_score)
	
	return result_indices, result_scores

# String to record results
results_string = ''

# First Four
first_four_result, first_four_scores = simulate_round(first_four_teams)
first_four_display = '\n'.join([f'{r} - {first_four_scores[i]}' for i, r in enumerate(first_four_result)])
print(first_four_display + '\n')
results_string += '\n'.join(first_four_result) + '\n\n'
first_four_split = first_four_result

# Rest of tournament
# Insert first four into the main tournament
tournament_teams = tournament_string.format(*first_four_split).split('\n')

# Simulate main tournament
for tournament_round in range(6):
	round_result, round_scores = simulate_round(tournament_teams)
	round_display = '\n'.join([f'{r} - {round_scores[i]}' for i, r in enumerate(round_result)])
	print(round_display + '\n')

	# Record results and repeat
	results_string += '\n'.join(round_result) + '\n\n'
	tournament_teams = round_result

open('tournament_results.txt', 'w').write(results_string)

print('\n\n\n\n')
import check
