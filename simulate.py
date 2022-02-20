import check
import defy_logging
import pickle
import random
import yaml


if __name__ == '__main__':
	logger = defy_logging.get_logger()

	with open('setup/config.yml') as file:
		config = yaml.safe_load(file)

	use_max_prob = config['simulate']['use_max_prob']
	checkpoint_path = config['simulate']['checkpoint_path']
	results_file_name = checkpoint_path.replace('networks/', '')[:-5]

	# Get pickled probabilities
	with open(f'results/{results_file_name}_probs.txt', 'rb') as file:
		results_probs = pickle.load(file)

	results_list = [[] for r in range(6)]

	for round in range(6):
		index = -(round + 1)
		round_probs = results_probs[index]
		
		for game in round_probs:
			# Check if any teams already picked for this round are in this game
			skip_round = False

			if index + 1 != 0:
				for team in results_list[index + 1]:
					if team in game:
						results_list[index].append(team)
						skip_round = True
						break
			
			if skip_round:
				continue

			if use_max_prob:
				win_team = max(game, key=game.get)
			else:
				win_team = random.choices(list(game.keys()), weights=list(game.values()), k=1)[0]

			results_list[index].append(win_team)

	results_string = '\n\n'.join(['\n'.join(r) for r in results_list])

	with open(f'results/{results_file_name}.txt', 'w') as file:
		file.write(results_string)

	logger.info('Running check...')
	check.check_results(logger)
