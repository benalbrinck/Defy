"""Displays bracket and probabilities of predictions made from simulate.py."""


import defy_logging
import defy_model
import os
import random
import yaml
import numpy as np
import tensorflow as tf
from datetime import datetime


def get_npz(path: str) -> tuple[np.array, np.array]:
	"""Get inputs and outputs and format npz file.
	
	Parameters:
		path (str): path of the npz file
	
	Returns:
		inputs (np.array): inputs gathered from npz file
		outputs (np.array): outputs gathered from npz file
	"""
	npz_file = np.load(path, allow_pickle=True)

	inputs = np.nan_to_num(npz_file['inputs'].astype('float32'))
	outputs = np.nan_to_num(npz_file['outputs_results'].astype('float32'))

	return inputs, outputs


def get_data(end_year: int, year: int) -> tuple[np.array, np.array]:
	"""Get data from end_year to year.
	
	Parameters:
		end_year (int): earlier year in range to gather data for (inclusive)
		year (int): later yer in range to gather data for (inclusive)
	
	Returns:
		input_array (np.array): inputs gathered from year range
		output_array (np.array): outputs gathered from year range
	"""
	inputs = []
	outputs = []

	for y in range(end_year, year + 1):
		next_inputs = 0

		if os.path.exists(f'data/data_{y}.npz'):
			# Regular season
			next_inputs, next_outputs = get_npz(f'data/data_{y}.npz')

		# NCAA tournament
		if y != year and os.path.exists(f'data/data_{y}_ncaa.npz'):
			next_inputs_ncaa, next_outputs_ncaa = get_npz(f'data/data_{y}_ncaa.npz')
			
			if type(next_inputs) == int:
				# No regular season games
				next_inputs = next_inputs_ncaa
				next_outputs = next_outputs_ncaa
			else:
				# Concatenate to regular season
				next_inputs = np.concatenate((next_inputs, next_inputs_ncaa), axis=0)
				next_outputs = np.concatenate((next_outputs, next_outputs_ncaa), axis=0)
		
		input_norm = np.linalg.norm(next_inputs, axis=0)
		next_inputs = np.divide(next_inputs, input_norm, out=np.zeros_like(next_inputs), where=input_norm!=0)

		inputs.append(next_inputs)
		outputs.append(next_outputs)
	
	input_array = np.concatenate(inputs, axis=0)
	output_array = np.concatenate(outputs, axis=0)

	return input_array, output_array


def flip_data(x: np.array) -> np.array:
	"""Flip first and second half of row in array.
	
	Parameters:
		x (np.array): array to flip
	
	Returns:
		x (np.array): flipped array
	"""
	pivot = int(x.shape[1] / 2)
	half_one = x[:, :pivot]
	half_two = x[:, pivot:]

	return np.concatenate((half_two, half_one), axis=1)


def calculate_temperature(epoch: int, logs: dict={}) -> None:
	"""Get temperature value based off of the current model.
	
	Parameters:
		epoch (int): the current epoch. Is used for logging
		logs (dict): logs from Tensorflow
	"""
	global predicted_outputs
	global temperature

	# Calculate temperature value
	predicted_outputs = model(validation_inputs)

	temperature = tf.Variable(initial_value=1.0, trainable=True, dtype=tf.float32)
	optimizer = tf.optimizers.Adam(learning_rate=0.01)

	for i in range(300):
		opts = optimizer.minimize(temperature_loss, var_list=[temperature])
	
	# Log and write temperature to file
	epoch_message = ' - '.join('{}: {:0.4f}'.format(k, logs[k]) for k in logs)
	logger.info(f'Epoch {epoch + 1}: {epoch_message}')
	logger.info(f'Temperature Value: {temperature.numpy()}')

	file_name = checkpoint_path[:-5].format(epoch=epoch + 1)

	with open(f'{file_name}.temp', 'w') as file:
		file.write(str(temperature.numpy()))


def temperature_loss() -> float:
	"""Get model loss for calculating temperature.
	
	Returns:
		loss (float): loss calculated from model and temperature
	"""
	temperature_model = tf.math.divide(predicted_outputs, temperature)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(validation_outputs, temperature_model))

	return loss


if __name__ == '__main__':
	logger = defy_logging.get_logger()

	with open('setup/config.yml') as file:
		config = yaml.safe_load(file)

	year = config['global']['start_year']
	end_year = config['global']['end_year']

	validation_percent = config['train']['validation_percent']
	epochs = config['train']['epochs']

	network_name = config['train']['name']
	checkpoint_path = f'networks/{network_name}'
	tensorboard_path = f'tensorboards/{network_name}'

	if config['train']['use_timestamp']:
		timestamp = f'{datetime.now()}'.replace(':', '')
		checkpoint_path += f'_{timestamp}'
		tensorboard_path += f'_{timestamp}'

	if config['train']['auto_set_ckpt']:
		# Automatically set the checkpoint path
		config['simulate']['checkpoint_path'] = checkpoint_path

		with open('setup/config.yml', 'w') as file:
			yaml.dump(config, file, default_flow_style=False)
	
	checkpoint_path += '/{epoch:04d}.hdf5'

	# Get training data
	logger.info('Getting training data...')
	input_array, output_array = get_data(end_year, year)

	logger.info(f'{input_array.shape[0]} games retrieved from {end_year} to {year}')

	# Split into testing and validation set
	logger.info(f'Splitting data into training and validation sets ({100 - validation_percent}/{validation_percent} split)...')
	
	data_count = input_array.shape[0] + 0
	validation_indices = random.sample(range(data_count), int(data_count * (validation_percent / 100)))
	training_indices = [i for i in range(data_count) if i not in validation_indices]

	training_inputs = input_array[training_indices]
	training_outputs = output_array[training_indices]
	validation_inputs = input_array[validation_indices]
	validation_outputs = output_array[validation_indices]

	# Duplicate data
	flipped_training_inputs = flip_data(training_inputs)
	flipped_training_outputs = flip_data(training_outputs)
	flipped_validation_inputs = flip_data(validation_inputs)
	flipped_validation_outputs = flip_data(validation_outputs)

	training_inputs = np.concatenate((training_inputs, flipped_training_inputs), axis=0)
	training_outputs = np.concatenate((training_outputs, flipped_training_outputs), axis=0)
	validation_inputs = np.concatenate((validation_inputs, flipped_validation_inputs), axis=0)
	validation_outputs = np.concatenate((validation_outputs, flipped_validation_outputs), axis=0)

	# Create callbacks
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path)
	temp_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=calculate_temperature)

	if not os.path.exists(f'networks/{network_name}'):
		os.makedirs(f'networks/{network_name}')

	checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
		filepath=checkpoint_path,
		save_weights_only=True,
		verbose=1
	)

	# Create network
	logger.info('Creating network...')
	model = defy_model.get_model()

	# Train model, saving between x epochs
	logger.info(f'Training network for {epochs} epochs...')
	model.fit(
		training_inputs,
		training_outputs,
		batch_size=32,
		epochs=epochs, 
		callbacks=[checkpoint_callback, temp_callback, tensorboard_callback],
		validation_data=(validation_inputs, validation_outputs),
		shuffle=True
	)

	# Check accuracy
	validation_accuracy = model.evaluate(validation_inputs, validation_outputs, verbose=2)
	logger.info(f'Validation Loss, Accuracy: {validation_accuracy}')
