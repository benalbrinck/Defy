# Defy

A machine learning application to predict games in the NCAA Tournament.

Defy uses a Tensorflow neural network model trained on previous regular season and NCAA Tournament games. The model takes in player statistics and outputs the probability that either team will win one game in the tournament, and Defy offers visualizations for this output.

## Table of Contents

- [Defy](#defy)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Setup](#setup)
  - [Usage](#usage)
    - [Gathering data](#gathering-data)
    - [Training the model](#training-the-model)
    - [Using the model](#using-the-model)
    - [Visualizing results](#visualizing-results)
  - [Configuration](#configuration)
    - [Global](#global)
    - [Train](#train)
    - [Simulate](#simulate)
    - [Visual](#visual)
    - [Setup](#setup-1)
      - [Conferences](#conferences)
      - [Tournament teams](#tournament-teams)
      - [Display teams](#display-teams)
      - [True results](#true-results)
      - [Removed players](#removed-players)
  - [Maintainers](#maintainers)
  - [License](#license)

## Installation

Install the requirements for the project:

```pip install requirements.txt```

The two major requirements are ```Tensorflow``` (https://www.tensorflow.org/api_docs/python/tf) for the machine learning model and ```Sportsipy``` (https://github.com/roclark/sportsipy) to gather data.

## Setup

All setup data is contained within the setup folder. There is an example setup folder in the project that can be renamed to setup. This is the example YAML config file:

```yaml
global:
  check_results: true
  check_year: 2021
  end_year: 2018
  start_year: 2021
simulate:
  checkpoint_path: 
  lpf_score: 0
  lpf_simulate: 0.2
  use_epoch: 102
  use_max_prob: false
  use_removed_players: true
train:
  auto_set_ckpt: true
  epochs: 1000
  name: example
  use_timestamp: false
  validation_percent: 30
visual:
  results_path: 
```

The example is configured to predict 2021 NCAA Tournament game with check year being set to 2021. All other settings will be explained in the [Usage](#usage) section when needed, and the [Configuration](#configuration) explains each setting in detail.

The folders each have a file named ```2021.txt``` inside of them.
- ```conferences``` lists all of the active conferences of the year
- ```tournament_teams``` lists the teams of the tournament
- ```display_teams``` lists the display names for the tournament teams
- ```removed_players``` lists all injured players for that tournament
- ```true_results``` stores the actual results of the tournament for that year

More details can be found in the [Configuration](#configuration) section.

## Usage

Before using Defy, make sure to follow the [Setup](#setup) instructions.

### Gathering data

Before gathering data, ```start_year``` and ```end_year``` in ```config.yaml``` should be set. ```start_year``` is the first year that will be collected, and it will collect years backwards up to and including ```end_year```.

```yaml
global:
  end_year: 2018
  start_year: 2021
```

In this example, Defy will collect the years 2021, 2020, 2019, and 2018. The ```conferences``` folder should also be set up for the year that is being predicted (more details in the [Configuration](#configuration) section).

Now, run ```collect.py```:

```python collect.py```

For each year, it will split all games into two files, one for regular season games and one for NCAA Tournament games. This is stored in the ```data``` folder. It also stores the teams and conferences for each year to make it easier to restart ```collect.py``` if it is stopped for any reason. It will skip all years that have data in the folder.

Defy gathers all teams for the year and then gets each team's schedule. If it detects that the game has already been collected, it will skip the game.

For each game, it collects the top 12 players that played in the game from both teams. Defy collects the statistics for each player, and it one-hot encodes the conference for each team based on the ```conferences``` setup folder. All of this is concatenated together to make one input row, and the output row is also one-hot encoded, where 1 is the team that won.

### Training the model

Before training the model, these configuration values should be set:

- ```global```
  - ```start_year```: the latest year to collect data on. This year will not include NCAA Tournament games.
  - ```end_year```: the earliest year to collect data on.
- ```train```
  - ```validation_percent```: the percent of the data to be randomly chosen to be in the validation set.
  - ```epochs```: the number of epochs to train.
  - ```name```: the name of the network. This will be used checkpoint file names.
  - ```auto_set_ckpt```: whether to automatically set ```checkpoint_path``` under ```simulate```. The path set will be ```networks/```[name of network].
  - ```use_timestamp```: whether or not to add timestamp to checkpoint path name.

If a data file is not found for a year, it will skip those files.

Now, run ```train.py```:

```python train.py```

The model checkpoints will be saved to ```networks/```[name of network]. In this folder, each epoch will have a checkpoint file and a temperature file, denoted by the epoch number. The network creates a temperature value after each epoch to adjust the output probabilities to be more accurate. The program can be stopped at any time and an earlier checkpoint can be used for simulations.

The model can be changed by modifying ```defy_model.py```. A model still has to be returned, the model has to end in a dense layer without an activation function, and these lines that apply the temperature value and softmax if simulate is true have to be kept:

```python
if simulate:
	model.add(tf.keras.layers.Lambda(lambda input: input / temperature))
	model.add(tf.keras.layers.Activation('softmax'))
```

A tensorboard will be generated during training under the tensorboards folder. To view the tensorboard:

```tensorboard --logdir tensorboard/[name of network]```

### Using the model

First, Defy needs to calculate the probability that each team will reach each round of the tournament. For each game in a round, every possible combination of teams will be run through the model, and the resulting probabilities will be multiplied by the probability that both teams will make it to that round. Then it will have the probability that each team will win the entire tournament.

The model that is used is based on ```checkpoint_path``` and ```use_epoch```. ```checkpoint_path``` is the folder that the checkpoint is located, and ```use_epoch``` is which epoch to use. ```lpf_score``` can also be set, where all probabilities at or below it will be set to 0.

Now, run ```score.py```:

```python score.py```

```score.py``` also finds the average and standard deviation of the score if each game was chosen based on those probabilities. How the score is calculated is based on the ESPN Tournament Challenge, where a correct guess in the first round is 10 points, and each subsequent round doubles the point value. The highest point value is 1920.

With the probabilities calculated, simulations can be run. In the configuration, ```use_max_prob``` will choose only the maximum probabilities if true and randomly if false, and ```lpf_simulate``` will make all probabilities at or below it 0.

Now, run ```simulate.py```:

```python simulate.py```

All results will be written to ```results/[name of network]/[max or random]_[epoch]_[timestamp].txt```. After it is run, it will set ```results_path``` under ```visual``` in the configuration to the last results file made.

### Visualizing results

The visualization will come from ```name``` and ```results_path```.

Run ```main.py```:

```python main.py```

This will show the entire bracket. If ```check_results``` is true, it will show the team as red if it was chosen incorrectly and blue if it was. If it is false, it will show blue for every game. If a game is pressed, on the right it will display the probability that each team would make it to that round. The correct team will be in blue, and the team chosen by the model will be in red if it chose incorrectly. If ```check_results``` is false, the team chosen will be blue.

## Configuration

### Global

- ```check_results```
  - Whether or not to check the model's predicted results against the true results.
  - In ```score.py```, true will result in the average and standard deviation of the score being calculated, while false will result in the average being 1920 and the standard deviation being 0.
  - In ```simulate.py```, true will result in the predicted results being checked at the end of the simlulation.
  - In ```visual.py```, true will result in incorrect results being colored in red and the score being calculated at the bottom of the screen.
- ```start_year```
  - The year for the model to predict, and the later year in the range of data to collect.
  - This affects ```collect.py```, ```train.py```, and ```score.py```.
- ```end_year```
  - The earlier year in the range of data to collect, inclusive.
  - This affects ```collect.py``` and ```train.py```.

### Train

- ```auto_set_ckpt```
  - Whether or not to automatically set ```checkpoint_path``` before training for simulations.
- ```epochs```
  - How many epochs to train for.
- ```name```
  - What to name the network.
  - This will be used in ```train.py```, ```score.py```, ```simulate.py```, and ```main.py``` to decide where to read and write data.
- ```use_timestamp```
  - Whether or not to append a timestamp to the name of the network when writing to the ```results``` folder.
- ```validation_percent```
  - What percent of the data set to use for validation.
  - This value is an integer (e.g. 30 = 30%).

### Simulate

- ```checkpoint_path```
  - The folder to look for the checkpoint files in for ```score.py```.
  - Normally, this will be ```networks/[name of network]```.
- ```use_epoch```
  - What epoch to use for ```score.py```.
  - With ```checkpoint_path```, this will create the full path to the checkpoint (e.g. ```networks/[name of network]/[epoch].hdf5```).
- ```use_removed_players```
  - Whether or not to remove injured players or not. If true, it will remove players.
- ```lpf_score```
  - Low (high) pass filter for ```score.py```.
  - While finding probabilities in ```score.py```, all probabilities for each team after each round that are below this filter will be set to 0. This is to prevent any low-probability team from being selected.
- ```lpf_simulate```
  - Low (high) pass filter for ```simulate.py```.
  - When choosing a team to win in ```simulate.py```, all probabilities for each team that are below this filter will be set to 0. This is to prevent any low-probability team from being selected.
- ```use_max_prob```
  - Whether to choose the maximum probability team or randomly select it based on the probabilities.
  - True is maximum probabilities, false is random probabilities.

### Visual

- ```results_path```
  - What results file to read for ```main.py```.
  - The results will be found in ```results/[name]/[results_path]```.

### Setup

These are the folders in the ```setup``` folder. Each year has one file in each folder that is named the year (e.g. ```2021.txt```). Every folder except for ```display_teams``` have conference, team, and player names that are compatible with ```Sportsipy```, which uses https://www.sports-reference.com/cbb/ to get its data.

#### Conferences

These are the conferences for the year that is being predicted. Go to https://www.sports-reference.com/cbb/seasons/ and select the season. This will list all the conferences for that season.

For each of the URLs, copy the conference name into the file. For example, the Big Ten conference has the URL https://www.sports-reference.com/cbb/conferences/big-ten/2021.html, so ```big-ten``` would be copied into the file. These names are not case-sensitive.

#### Tournament teams

This is a list of all 64 teams in the tournament. Like the conferences, these need to be from Sports Reference. For each team in the bracket, the name has to be taken from the team's page on Sports Reference. For example, Gonzaga's page is https://www.sports-reference.com/cbb/schools/gonzaga/2021.html, so ```gonzaga``` would be copied into the file. The easiest way to find these pages is to search for the teams in the search bar on Sports Reference.

The teams have to be listed in the correct order for the bracket to be made correctly. Most brackets online have 32 teams on either side of the bracket, so the correct order would be to copy all the teams on the left side from top to bottom, then do the same on the right side.

#### Display teams

These are the display names used in ```main.py``` for each of the teams in the tournament. Each of the lines in ```display_teams``` corresponds to the same team in ```tournament_teams```.

#### True results

These are the actual results for the given year's tournament. The same names will be used as ```tournament_teams```. Each team is delimited by a new line, and each round is delimted by two new lines. The first section in the file are the 32 teams that won the first round, the second section is the 16 teams that won the second round and so on. The final section should be one team, which is the champion of the tournament.

#### Removed players

This is the list of all injured players during the tournament of that year. Sports Reference does not have a list of injured players, so an outside source has to be used and then the names have to be converted to a format Sports Reference can use.

One source is https://www.covers.com/sport/basketball/ncaab/injuries. This only lists the current injuries, so to find historical data the Internet Archive for this page can be used.

For each injured player, search for them on Sports Reference. Many players have the same name, so make sure they are from the correct college. For example, Joshua Primo's page is https://www.sports-reference.com/cbb/players/joshua-primo-1.html, so joshua-primo-1 will be used. After their name, press tab and put the round number that they will return after based on about when they will recover from their injury. If they will most likely not return before the tournament is over, a 6 or higher can be used.

## Maintainers
- Ben Albrinck (https://github.com/benalbrinck)

## License

MIT License. Copyright (c) 2022 Ben Albrinck