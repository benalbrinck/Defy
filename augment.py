import glob
import imageio
import os
import time
from numpy.lib.function_base import select
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from IPython import display
from tensorflow.keras import layers


class AugmentGAN:
	def __init__(self, data, labels, augment_percent, buffer_multiplier, epochs, batch_size, generator, discriminator, tracker, noise_dim=100, name=''):
		"""Creates an AugmentGAN object, which will augment the size of a dataset.
		
		Keyword arguments:
		data -- the dataset to augment.
		labels -- the labels of the dataset to augment. Should be single digits instead of one-hot encoded.
		augment_percent -- the percentage of the population to create each step (in decimal form).
		buffer_multiplier -- controls how much the dataset is tiled for training.
		epochs -- how many epochs to train the network each step.
		batch_size -- the batch size of training.
		generator -- the function to create a generator model.
		discriminator -- the function to create a discriminator model.
		tracker -- the function called to save the prediction images to a file to keep track of training.
		noise_dim -- how big the noise will be during training.
		name -- the name of the network. Is used when saving and keeping track of where in the train step the network is."""
		
		self.augment_percent = augment_percent
		self.buffer_multiplier = buffer_multiplier
		self.epochs = epochs
		self.batch_size = batch_size
		self.noise_dim = noise_dim
		self.name = name
		
		self.one_hot_size = np.max(labels) + 1
		self.data = [data[np.where(labels == i)[0]] for i in range(self.one_hot_size)]  # Labels implied
		self.data_index = 0

		self.generator_model = generator
		self.discriminator_model = discriminator
		self.tracker = tracker

		self.__create_network()
	
	def __create_network(self):
		"""Create the generator and discriminator models for the network."""
		self.generators = [self.generator_model() for i in range(len(self.data))]
		self.discriminators = [self.discriminator_model() for i in range(len(self.data))]

		# Create optimizers
		self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
		self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
		self.discriminator_optimizer = tf.keras.optimizers.Adam(4e-4)  # Originally 1e-4

		# Save checkpoints
		checkpoint_dir = './training_checkpoints'
		self.checkpoints = []
		self.checkpoint_prefixes = []

		for i in range(len(self.data)):
			self.checkpoint_prefixes.append(os.path.join(checkpoint_dir, f'ckpt{i}'))
			self.checkpoints.append(
				tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
									discriminator_optimizer=self.discriminator_optimizer,
									generator=self.generators[i],
									discriminator=self.discriminators[i])
			)
			self.checkpoints[-1].restore(tf.train.latest_checkpoint(checkpoint_dir))

	def __generator_loss(self, fake_output):
		return self.cross_entropy(tf.ones_like(fake_output), fake_output)
	
	def __discriminator_loss(self, real_output, fake_output):
		real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
		fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
		total_loss = real_loss + fake_loss
		return real_loss, total_loss

	def train_step(self, start_index=-1):
		if start_index != -1:
			self.data_index = start_index + 0

		# Training loop
		index_data = self.data[self.data_index]
		num_examples_to_generate = int(self.augment_percent * index_data.shape[0])
		self.seed = tf.random.normal([num_examples_to_generate, self.noise_dim])  # Same seed to visualize progress
		train_dataset = tf.data.Dataset.from_tensor_slices(index_data).shuffle(index_data.shape[0] * self.buffer_multiplier).batch(self.batch_size)

		self.__train(train_dataset, self.epochs)
		self.data_index += 1

		if self.data_index == len(self.data):
			self.data_index = 0
	
	def __train(self, dataset, epochs):
		for epoch in range(epochs):
			start = time.time()

			for batch in dataset:
				self.__train_step(batch)

			# Save the model every 50 epochs and on last epoch
			if ((epoch + 1) % 50 == 0) or (epoch == epochs - 1):
				if epoch != epochs - 1:
					self.__generate_and_save(self.generators[self.data_index], epoch + 1, self.seed)
				self.checkpoints[self.data_index].save(file_prefix = self.checkpoint_prefixes[self.data_index])
				print ('\nTime for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

		# Generate after the final epoch
		self.__generate_and_save(self.generators[self.data_index], epochs, self.seed, True)

	@tf.function
	def __train_step(self, data):
		noise = tf.random.normal([self.batch_size, self.noise_dim])

		with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
			generated_data = self.generators[self.data_index](noise, training=True)

			real_output = self.discriminators[self.data_index](data, training=True)
			fake_output = self.discriminators[self.data_index](generated_data, training=True)

			gen_loss = self.__generator_loss(fake_output)
			real_loss, disc_loss = self.__discriminator_loss(real_output, fake_output)

		gradients_of_generator = gen_tape.gradient(gen_loss, self.generators[self.data_index].trainable_variables)
		gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminators[self.data_index].trainable_variables)

		self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generators[self.data_index].trainable_variables))
		self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminators[self.data_index].trainable_variables))
	
	def __generate_and_save(self, model, epoch, test_input, add_data=False):
		predictions = model(test_input, training=False)
		self.tracker(predictions, self.data_index, epoch)

		# Add on new images
		if add_data:
			self.data[self.data_index] = np.concatenate((self.data[self.data_index], predictions), axis=0)
			self.save_data()

	def save_data(self):
		(save_data, save_labels) = self.return_data()
		np.savez(f'{self.name}_selected_data.npz', data=save_data, labels=save_labels)

	def return_data(self):
		# Splice it all back together with labels
		labels = [np.ones((self.data[i].shape[0],)) * i for i in range(len(self.data))]
		
		combined_data = np.concatenate(self.data, axis=0)
		combined_labels = np.concatenate(labels, axis=0)

		return combined_data, combined_labels


# Input functions for MM network
def generator_model():
	generator = tf.keras.Sequential()
	generator.add(layers.Dense(2*256, use_bias=False, input_shape=(100,)))
	generator.add(layers.BatchNormalization())
	generator.add(layers.LeakyReLU())

	generator.add(layers.Dense(300, use_bias=False))
	generator.add(layers.BatchNormalization())
	generator.add(layers.LeakyReLU())

	generator.add(layers.Dense(256, use_bias=False))
	generator.add(layers.BatchNormalization())
	generator.add(layers.LeakyReLU())

	generator.add(layers.Dense(234, use_bias=False, activation='tanh'))
	assert generator.output_shape == (None, 234)

	return generator

def discriminator_model():
	discriminator = tf.keras.Sequential()

	discriminator.add(layers.InputLayer([234]))

	discriminator.add(layers.GaussianNoise(0.2))
	# discriminator.add(layers.experimental.preprocessing.RandomRotation(0.1))
	# discriminator.add(layers.experimental.preprocessing.RandomZoom(0.1))

	discriminator.add(layers.Dense(300, use_bias=False))
	discriminator.add(layers.LeakyReLU())
	discriminator.add(layers.Dropout(0.3))

	discriminator.add(layers.Dense(150, use_bias=False))
	discriminator.add(layers.LeakyReLU())
	discriminator.add(layers.Dropout(0.3))

	discriminator.add(layers.Dense(1))

	return discriminator

def tracker(predictions, index, epoch):
	pass


if __name__ == '__main__':
	tf.config.run_functions_eagerly(True)
	name = 'MM'

	if os.path.isfile(f'{name}_selected_data.npz'):
		# This will already be normalized
		npz_file = np.load(f'{name}_selected_data.npz')
		selected_images = npz_file['data']
		selected_labels = npz_file['labels']

		print(selected_images.shape)
		print(selected_labels.shape)

		npz_file = np.load('data.npz', allow_pickle=True)
		selected_images = npz_file['inputs'].astype('float32')
		input_norm = np.linalg.norm(selected_images, axis=0)

		print((selected_images[0] * input_norm)[:117])
		print((selected_images[-1] * input_norm)[:117])
		quit()
	else:
		npz_file = np.load('data.npz', allow_pickle=True)

		selected_images = npz_file['inputs'].astype('float32')
		input_norm = np.linalg.norm(selected_images, axis=0)
		selected_images = np.divide(selected_images, input_norm, out=np.zeros_like(selected_images), where=input_norm!=0)

		selected_labels_one_hot = npz_file['outputs_results'].astype('float32')
		selected_labels = np.argmax(selected_labels_one_hot, axis=1)

	# Need a quick thing to revert single digits to one-hot and back

	network = AugmentGAN(selected_images, selected_labels, 0.1, 1, 200, 128, generator_model, discriminator_model, tracker, name=name)

	for i in range(10000):
		print(network.data_index)
		network.train_step()

# Yeah so the generated examples will be at the end of the batch
# So we can say grab a random sample from the length of the data from data.npz to the augmented length
#  then multiply by the input_norm to get back the actual data
# Then we can compare
