import random
import numpy as np
import tensorflow as tf

load_model = False

# Checkpoint parameters
checkpoint_path = 'network.ckpt'

# Import from non-augmented data
npz_file = np.load('data.npz', allow_pickle=True)

inputs = npz_file['inputs'].astype('float32')
# outputs = npz_file['outputs_results'].astype('float32')
outputs = npz_file['outputs_scores'].astype('float32')
output_not_nan = [i for i, r in enumerate(np.isnan(outputs).any(axis=1)) if r]

inputs = [x for i, x in enumerate(inputs) if i not in output_not_nan]
outputs = [x for i, x in enumerate(outputs) if i not in output_not_nan]

input_norm = np.linalg.norm(inputs, axis=0)
inputs = np.divide(inputs, input_norm, out=np.zeros_like(inputs), where=input_norm!=0)
output_norm = np.linalg.norm(outputs, axis=0)  # Only do if doing scores
outputs = np.divide(outputs, output_norm, out=np.zeros_like(outputs), where=output_norm!=0)

"""
# Import from augmented data
npz_file = np.load(f'MM_selected_data.npz')
inputs = npz_file['data']
outputs_integers = npz_file['labels']
outputs = np.zeros((outputs_integers.size, int(outputs_integers.max()) + 1))
outputs[np.arange(outputs_integers.size), np.array(outputs_integers, dtype=np.int)] = 1
"""

# Split into testing and validation set
validation_percent = 30

data_count = inputs.shape[0] + 0
validation_indices = random.sample(range(data_count), int(data_count * (validation_percent / 100)))
training_indices = [i for i in range(data_count) if i not in validation_indices]

training_inputs = inputs[training_indices]
training_outputs = outputs[training_indices]
validation_inputs = inputs[validation_indices]
validation_outputs = outputs[validation_indices]

# Create neural network
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

if load_model:
	model.load_weights(checkpoint_path)

# Create checkpoint
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, 
	save_weights_only=True, verbose=1)

# Train model, saving between x epochs
epochs = 10000
model.fit(training_inputs, training_outputs, epochs=epochs, 
	callbacks=[checkpoint], validation_data=(validation_inputs, validation_outputs))

# Check accuracy
print(model.evaluate(validation_inputs, validation_outputs, verbose=2))
