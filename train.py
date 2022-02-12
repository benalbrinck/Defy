import defy_logging
import random
import yaml
import numpy as np
import tensorflow as tf
from datetime import datetime


# Parameters
checkpoint_path = f'networks/{datetime.now()}.ckpt'.replace(':', '')  # In simulate.py, we'll get latest one unless override
tensorboard_path = f'tensorboards/{datetime.now()}'.replace(':', '')


def flip_data(x):
	pivot = int(x.shape[1] / 2)
	half_one = x[:, :pivot]
	half_two = x[:, pivot:]

	return np.concatenate((half_two, half_one), axis=1)


if __name__ == '__main__':
	logger = defy_logging.get_logger()

	with open('setup/config.yml') as file:
		config = yaml.safe_load(file)

	year = config['global']['start_year']
	end_year = config['global']['end_year']

	validation_percent = config['train']['validation_percent']
	activation_function = config['train']['activation_function']
	epochs = config['train']['epochs']

	if config['train']['auto_set_ckpt']:
		# Automatically set the checkpoint path
		config['simulate']['checkpoint_path'] = checkpoint_path

		with open('setup/config.yml', 'w') as file:
			yaml.dump(config, file, default_flow_style=False)

	# Get training data
	logger.info('Getting training data...')
	inputs = []
	outputs = []

	for y in range(end_year, year + 1):
		npz_file = np.load(f'data/data_{y}.npz', allow_pickle=True)

		next_inputs = np.nan_to_num(npz_file['inputs'].astype('float32'))
		next_outputs = np.nan_to_num(npz_file['outputs_results'].astype('float32'))

		input_norm = np.linalg.norm(next_inputs, axis=0)
		next_inputs = np.divide(next_inputs, input_norm, out=np.zeros_like(next_inputs), where=input_norm!=0)
		
		inputs.append(next_inputs)
		outputs.append(next_outputs)
	
	input_array = np.concatenate(inputs, axis=0)
	output_array = np.concatenate(outputs, axis=0)

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

	# Create network
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path)

	logger.info('Creating network...')
	model = tf.keras.models.Sequential([
		tf.keras.layers.Dense(128, activation=activation_function, input_shape=(1168,)),
		tf.keras.layers.Dropout(0.1),
		tf.keras.layers.Dense(32, activation=activation_function),
		tf.keras.layers.Dropout(0.1),
		tf.keras.layers.Dense(2, activation='softmax')
	])
	loss_function = tf.keras.losses.BinaryCrossentropy()
	model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])

	# Create checkpoint
	checkpoint = tf.keras.callbacks.ModelCheckpoint(
		filepath=checkpoint_path,
		save_weights_only=True,
		verbose=1
	)

	# Train model, saving between x epochs
	logger.info(f'Training network for {epochs} epochs...')
	model.fit(
		training_inputs,
		training_outputs,
		epochs=epochs, 
		callbacks=[checkpoint, tensorboard_callback],
		validation_data=(validation_inputs, validation_outputs)
	)

	# Check accuracy
	logger.info(model.evaluate(validation_inputs, validation_outputs, verbose=2))
