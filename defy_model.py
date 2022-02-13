import yaml
import tensorflow as tf
import tensorflow_addons as tfa

def get_model():
	with open('setup/config.yml') as file:
		config = yaml.safe_load(file)

	activation_function = config['train']['activation_function']

	model = tf.keras.models.Sequential([
		tf.keras.layers.Dense(128, activation=activation_function, input_shape=(1168,)),
		tf.keras.layers.Dropout(0.1),
		tf.keras.layers.Dense(32, activation=activation_function),
		tf.keras.layers.Dropout(0.1),
		tf.keras.layers.Dense(2, activation='softmax')
	])
	
	loss_function = tf.keras.losses.BinaryCrossentropy()
	optimizer = tfa.optimizers.AdamW(learning_rate=1e-1, weight_decay=1e-4)
	model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

	return model
