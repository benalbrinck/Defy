import tensorflow as tf
import tensorflow_addons as tfa

def get_model(simulate=False, temperature=1):
	model = tf.keras.models.Sequential([
		tf.keras.layers.Dense(16, input_shape=(1168,)),
		tf.keras.layers.GaussianNoise(0.1),
		tf.keras.layers.LeakyReLU(),
		tf.keras.layers.Dropout(0.1),
		tf.keras.layers.Dense(32),
		tf.keras.layers.GaussianNoise(0.1),
		tf.keras.layers.LeakyReLU(),
		tf.keras.layers.Dropout(0.6),
		tf.keras.layers.Dense(2)
	])

	if simulate:
		model.add(tf.keras.layers.Lambda(lambda input: input / temperature))
		model.add(tf.keras.layers.Activation('softmax'))
	
	learning_rate = tf.optimizers.schedules.InverseTimeDecay(1e-3, 1, 1e-3)
	# weight_decay = 1e-4

	# optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
	optimizer = tfa.optimizers.LazyAdam(learning_rate=learning_rate)

	loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
	model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

	return model
