import defy_logging
import yaml


def check_results(logger):
	# Get config
	with open('setup/config.yml') as file:
		config = yaml.safe_load(file)

	checkpoint_path = config['simulate']['checkpoint_path']
	results_file_name = checkpoint_path.replace('networks/', '')[:-5]

	# Get true and simulated results
	with open(f'results/{results_file_name}.txt') as file:
		predicted_results = [r.split('\n') for r in file.read().split('\n\n')]

	with open('setup/true_results.txt') as file:
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

	checkpoint_path = config['simulate']['checkpoint_path']
	results_file_name = checkpoint_path.replace('networks/', '')[:-5]

	# Get true and simulated results
	with open(f'results/{results_file_name}.txt') as file:
		predicted_results = [r.split('\n') for r in file.read().split('\n\n')]

	with open('setup/true_results.txt') as file:
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
