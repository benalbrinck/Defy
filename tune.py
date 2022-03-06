from random import Random
from sched import scheduler
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
                first_noise, second_noise, third_noise,
                learning_rate, use_lr_schedule,
                weight_decay, use_wd_schedule,
                optimizer, momentum):
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

    # Learning rate and weight decay
    if use_lr_schedule:
        learning_rate = tf.optimizers.schedules.InverseTimeDecay(learning_rate, 1, 1e-3)

    if use_wd_schedule:
        weight_decay = tf.optimizers.schedules.InverseTimeDecay(weight_decay, 1, 1e-3)

    # Optimizers
    optimizers = {
        'adam': tf.keras.optimizers.Adam(learning_rate=learning_rate),
        'sgd': tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum),
        'adamw': tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay),
        'sgdw': tfa.optimizers.SGDW(learning_rate=learning_rate, weight_decay=weight_decay, momentum=momentum),
        'rmsprop': tf.keras.optimizers.RMSprop(learning_rate=learning_rate, momentum=momentum),
        'lazyadam': tfa.optimizers.LazyAdam(learning_rate=learning_rate)
    }

    loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizers[optimizer], loss=loss_function, metrics=['accuracy'])

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
    check_amount = 10
    cv = 10

    # Testing values
    sizes = [0, 8, 16, 32, 64, 128, 256, 512]
    dropouts = [0.1, 0.5, 0.6, 0.7, 0.8]
    noise = [0, 0.1]
    rates = [10 ** -(i + 1) for i in range(5)]
    momentums = [0, 0.9]
    optimizers = ['adam', 'sgd', 'adamw', 'sgdw', 'rmsprop', 'lazyadam']

    param_grid = dict(
        first_size=sizes[1:], second_size=sizes, third_size=sizes,
        first_dropout=dropouts, second_dropout=dropouts, third_dropout=dropouts,
        first_noise=noise, second_noise=noise, third_noise=noise,
        learning_rate=rates, use_lr_schedule=[True, False],
        weight_decay=rates, use_wd_schedule=[True, False],
        optimizer=optimizers, momentum=momentums
    )

    # Search
    model_cv = KerasClassifier(build_fn=get_model, verbose=1)
    grid = RandomizedSearchCV(
        estimator=model_cv,
        param_distributions=param_grid,
        cv=cv,
        n_iter=check_amount
    )

    logger.info('Starting search...')
    logger.info(f'Epochs: {epochs}, Batch size: {batch_size}, Check amount: {check_amount}, Cv: {cv}')
    grid_result = grid.fit(input_array, output_array)

    # Results
    logger.info(f'Best Accuracy for {grid_result.best_score_:.4} using {grid_result.best_params_}')

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):
        logger.info(f'mean={mean:.4}, std={stdev:.4} using {param}')


"""
I wonder if I can grab all this data and kinda fit the parameters to the accuracy to predict the best structure

And maybe have it pick best on average and max-ish accuracy (mean + stdev)
"""
