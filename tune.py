from random import Random
import defy_logging
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import train
import yaml
from sklearn.model_selection import RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier


def get_model(first_size, second_size, third_size,
                first_dropout, second_dropout, third_dropout,
                first_noise, second_noise, third_noise):
    model = tf.keras.models.Sequential()

    # Layers
    model.add(tf.keras.layers.Dense(first_size, input_shape=(1168,)))
    model.add(tf.keras.layers.GaussianNoise(first_noise))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(first_dropout))

    if second_size != 0:
        model.add(tf.keras.layers.Dense(second_size))
        model.add(tf.keras.layers.GaussianNoise(second_noise))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(second_dropout))
    
    if third_size != 0:
        model.add(tf.keras.layers.Dense(third_size))
        model.add(tf.keras.layers.GaussianNoise(third_noise))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(third_dropout))
    
    model.add(tf.keras.layers.Dense(2))

    # Optimizers
    step = tf.Variable(0, trainable=False)
    schedule = tf.optimizers.schedules.PiecewiseConstantDecay([10000, 15000], [1e-0, 1e-1, 1e-2])

    learning_rate = 1e-1 * schedule(step)
    weight_decay = lambda: 1e-4 * schedule(step)

    # optimizer = tfa.optimizers.AdamW(weight_decay=1e-4)
    optimizer = tfa.optimizers.SGDW(learning_rate=learning_rate, weight_decay=weight_decay, momentum=0.9)

    loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

    return model


if __name__ == '__main__':
    logger = defy_logging.get_logger()

    with open('setup/config.yml') as file:
        config = yaml.safe_load(file)

    year = config['global']['start_year']
    end_year = config['global']['end_year']

    input_array, output_array = train.get_data(end_year, year)

    # Parameters
    epochs = 75
    batch_size = 32
    check_amount = 30

    # Testing values
    sizes = [0, 8, 16, 32, 64, 128, 256, 512]
    dropouts = [0.1, 0.5, 0.6, 0.7, 0.8]
    noise = [0, 0.1]

    param_grid = dict(
        first_size=sizes[1:], second_size=sizes, third_size=sizes,
        first_dropout=dropouts, second_dropout=dropouts, third_dropout=dropouts,
        first_noise=noise, second_noise=noise, third_noise=noise
    )

    # Search
    model_cv = KerasClassifier(build_fn=get_model, verbose=1)
    grid = RandomizedSearchCV(
        estimator=model_cv,
        param_distributions=param_grid,
        cv=10,
        n_iter=check_amount
    )

    logger.info('Starting search...')
    grid_result = grid.fit(input_array, output_array)

    # Results
    logger.info(f'Best Accuracy for {grid_result.best_score_:.4} using {grid_result.best_params_}')

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):
        logger.info(f'mean={mean:.4}, std={stdev:.4} using {param}')

"""
/ Size and # of layers
/ Dropout frequency
/ Noise frequency
- Optimizers
- Learning rates/schedulers
"""
