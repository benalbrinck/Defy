import defy_logging
import os
import yaml


def get_predicted_results():
	with open('setup/config.yml') as file:
		config = yaml.safe_load(file)

	use_max_prob = config['simulate']['use_max_prob']
	checkpoint_path = config['simulate']['checkpoint_path']
	results_folder_name = checkpoint_path.replace('networks/', '')
	use_epoch = config['simulate']['use_epoch']

	starts_with = 'max_' if use_max_prob else 'random_'
	starts_with += f'{use_epoch:04d}_'

	# Get all file names in folder and find which one matches starts_with
	results_file_name = ''

	for file_name in reversed(os.listdir(f'results/{results_folder_name}')):
		if starts_with in file_name:
			results_file_name = file_name + ''
			break
	
	if results_file_name == '':
		logger.info('File not found')
		return

	# Get predicted_results
	with open(f'results/{results_folder_name}/{results_file_name}') as file:
		predicted_results = [r.split('\n') for r in file.read().split('\n\n')]

	return predicted_results


def check_results(logger):
	# Get config
	with open('setup/config.yml') as file:
		config = yaml.safe_load(file)

	check_year = config['global']['check_year']

	# Get true and simulated results
	predicted_results = get_predicted_results()

	with open(f'setup/true_results/{check_year}.txt') as file:
		true_results = [r.split('\n') for r in file.read().split('\n\n')]

	# Give point total + show them both side by side w/ whether or not right there
	amount_right = 0
	points = 0
	max_points = 0

	for r in range(len(predicted_results)):
		logger.info(f'Round {r}')

		for g in range(len(predicted_results[r])):
			right_bool = int(predicted_results[r][g] == true_results[r][g])
			amount_right += right_bool

			# Add on points
			points += ((2 ** r) * 10) * right_bool
			max_points += (2 ** r) * 10

			logger.info(f'{predicted_results[r][g]}\t{right_bool}\t{true_results[r][g]}')

	logger.info(f'POINTS: {points}/{max_points}')
	logger.info(f'AMOUNT RIGHT: {amount_right}/63')


def get_round_points():
	# Get config
	with open('setup/config.yml') as file:
		config = yaml.safe_load(file)

	check_year = config['global']['check_year']

	# Get true and simulated results
	predicted_results = get_predicted_results()

	with open(f'setup/true_results/{check_year}.txt') as file:
		true_results = [r.split('\n') for r in file.read().split('\n\n')]

	# Get point total for each round
	round_points = []
	max_round_points = []

	for r in range(len(predicted_results)):
		round_points.append(0)
		max_round_points.append(0)

		for g in range(len(predicted_results[r])):
			right_bool = int(predicted_results[r][g] == true_results[r][g])

			# Add on points
			round_points[-1] += ((2 ** r) * 10) * right_bool
			max_round_points[-1] += (2 ** r) * 10
	
	return round_points, max_round_points

if __name__ == '__main__':
	logger = defy_logging.get_logger()
	check_results(logger)
