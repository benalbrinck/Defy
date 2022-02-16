import yaml
import tensorflow as tf
import tensorflow_addons as tfa

def get_model(simulate=False, temperature=1):
	with open('setup/config.yml') as file:
		config = yaml.safe_load(file)

	model = tf.keras.models.Sequential([
		tf.keras.layers.Dense(128, input_shape=(1168,)),
		tf.keras.layers.LeakyReLU(),
		tf.keras.layers.Dropout(0.1),
		tf.keras.layers.Dense(32),
		tf.keras.layers.LeakyReLU(),
		tf.keras.layers.Dropout(0.1),
		tf.keras.layers.Dense(2)
	])

	if simulate:
		model.add(tf.keras.layers.Lambda(lambda input: input / temperature))
		model.add(tf.keras.layers.Activation('softmax'))
	
	step = tf.Variable(0, trainable=False)
	schedule = tf.optimizers.schedules.PiecewiseConstantDecay([10000, 15000], [1e-0, 1e-1, 1e-2])

	learning_rate = 1e-1 * schedule(step)
	weight_decay = lambda: 1e-4 * schedule(step)

	# optimizer = tfa.optimizers.AdamW(weight_decay=1e-4)
	optimizer = tfa.optimizers.SGDW(learning_rate=learning_rate, weight_decay=weight_decay, momentum=0.9)

	loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
	model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

	return model
