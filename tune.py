import defy_logging
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import train
import os
import yaml
from sklearn.model_selection import RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier


def get_model(first_size, second_size,
                first_dropout, second_dropout,
                first_noise, second_noise,
                learning_rate, use_lr_schedule,
                weight_decay, use_wd_schedule,
                optimizer, momentum):
    model = tf.keras.models.Sequential()

    # Layers
    model.add(tf.keras.layers.Dense(first_size, input_shape=(1168,)))
    model.add(tf.keras.layers.GaussianNoise(first_noise))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(first_dropout))

    model.add(tf.keras.layers.Dense(second_size))
    model.add(tf.keras.layers.GaussianNoise(second_noise))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(second_dropout))
    
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


def test_models(model, params, cv=10, number_models=1):
    grid = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        cv=cv,
        n_iter=number_models
    )
    
    grid_result = grid.fit(input_array, output_array)
    return grid_result


def output_results(grid_result):
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    param_keys = '\t'.join(params[0].keys())

    # Check if file exists
    if os.path.exists('tuning.tsv'):
        with open('tuning.tsv') as file:
            text = file.read()
    else:
        text = f'mean\tstdev\t{param_keys}'

    for mean, stdev, param in zip(means, stds, params):
        logger.info(f'mean={mean:.4}, std={stdev:.4} using {param}')
        param_values = '\t'.join([str(val) for val in param.values()])
        text += f'\n{mean:.4}\t{stdev:.4}\t{param_values}'

    with open('tuning.tsv', 'w') as file:
        file.write(text)


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
    check_rounds = 1
    cv = 10
    number_models = 1

    # Testing values
    sizes = [8, 16, 32, 64, 128, 256, 512]
    dropouts = [0, 0.1, 0.5, 0.6, 0.7, 0.8]
    noise = [0, 0.1]
    rates = [10 ** -(i + 1) for i in range(5)]
    momentums = [0, 0.9]
    optimizers = ['adam', 'sgd', 'adamw', 'sgdw', 'rmsprop', 'lazyadam']

    param_grid = dict(
        first_size=sizes, second_size=sizes,
        first_dropout=dropouts, second_dropout=dropouts,
        first_noise=noise, second_noise=noise,
        learning_rate=rates, use_lr_schedule=[True, False],
        weight_decay=rates, use_wd_schedule=[True, False],
        optimizer=optimizers, momentum=momentums
    )

    # Search
    logger.info('Starting search...')
    logger.info(f'Epochs: {epochs}, Batch size: {batch_size}, Check rounds: {check_rounds}, Cv: {cv}')

    model_cv = KerasClassifier(build_fn=get_model, verbose=1)

    for i in range(check_rounds):
        grid_result = test_models(model_cv, param_grid, cv, number_models)
        output_results(grid_result)
