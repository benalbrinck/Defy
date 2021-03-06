"""Collects data on past NCAA Tournament and regular season games."""


import defy_logging
import json
import os
import yaml
import numpy as np
from sportsreference.ncaab.conferences import Conference
from sportsreference.ncaab.teams import Teams
from sportsreference.ncaab.schedule import Schedule
from time import sleep


def get_conference_teams(conference_names: list, year: int) -> list:
    """Get all teams for each conference in the current year.
    
    Parameters:
        conference_names (list): the names of each conference to get teams for.
        year (int): the year to get teams for.

    Returns:
        conference_teams (list): a list of a list of teams. Each position in the outer 
            list corresponds with the same position in conference_names.
    """
    conference_teams = []

    for conference_name in conference_names:
        conference = Conference(conference_name.lower(), str(year))
        conference_teams.append(list(conference.teams))
    
    return conference_teams


def get_teams(year: int) -> dict:
    """Gets all teams and their player data for the specified year.

    Parameters:
        year (int): the year to get teams for.
    
    Returns:
        teams (dict): a dictionary of teams and their players, with the keys being the teams 
            and the values being a dictionary of players. This dictionary has the player name 
            as the key and the value as their data converted from a numpy array to a list.
    """
    teams = {}

    for team in Teams(str(year)):
        team_name = team.abbreviation
        roster = {}
        remove_indices = [5, 20, 28, 31, 34]  # Indices that are strings

        for player in team.roster.players:
            try:
                player_data = player.dataframe.loc[f'{year - 1}-{str(year)[2:]}'].to_numpy()
            except:
                continue
            
            player_fixed_data = np.expand_dims(np.delete(player_data, remove_indices), axis=0)
            roster[player.player_id] = player_fixed_data.tolist()
    
        teams[team_name] = roster
        sleep(1)
    
    return teams


if __name__ == '__main__':
    logger = defy_logging.get_logger()

    with open('setup/config.yml') as file:
        config = yaml.safe_load(file)
    
    year = config['global']['start_year']
    end_year = config['global']['end_year']
    check_year = config['global']['check_year']

    with open(f'setup/conferences/{check_year}.txt') as file:
        conference_names = file.read().split('\n')

    # Get data
    while True:
        logger.info(year)

        # Network input and output variables
        inputs = []
        outputs_results = []

        inputs_ncaa = []
        outputs_results_ncaa = []

        # Get all conference teams and team data
        if os.path.isfile(f'data/conferences_{year}.json'):
            logger.info('\tConference file exists, loading...')
            with open(f'data/conferences_{year}.json') as file:
                conference_teams = json.load(file)
        else:
            logger.info('\tConference file does not exist, gathering data from API...')
            conference_teams = get_conference_teams(conference_names, year)

            # Save data for future use
            with open(f'data/conferences_{year}.json', 'w') as file:
                json.dump(conference_teams, file)

        if os.path.isfile(f'data/data_{year}.npz'):
            logger.info('\tYear data exists, skipping...')

            # Check if this is the last year
            if year == end_year:
                break

            year -= 1
            continue

        if os.path.isfile(f'data/teams_{year}.json'):
            logger.info('\tTeams data file exists, loading...')
            with open(f'data/teams_{year}.json') as file:
                teams = json.load(file)
        else:
            logger.info('\tTeams data file does not exist, gathering data from API...')
            teams = get_teams(year)

            # Save data
            with open(f'data/teams_{year}.json', 'w') as file:
                json.dump(teams, file)

        # For each team, get schedule and add to data
        game_index = []  # Ensures every game is counted once instead of twice

        for team in teams:
            logger.info(f'\tGetting schedule for {team}...')

            team_players = teams[team]
            team_conference = np.expand_dims(np.array([int(team.lower() in c) for c in conference_teams]), axis=0)

            try:
                schedule = Schedule(team.lower(), str(year))
            except Exception as e:
                logger.info(f'\t\tNo schedule.')
                continue

            for game in schedule:
                if game.points_for == None:
                    # If the game hasn't been played yet, skip it
                    continue

                if game.datetime in game_index:
                    # If the game has already been counted, skip it
                    logger.info(f'\t\t\tSkipping {game.datetime}')
                    continue

                logger.info(f'\t\t\t{game.datetime}, {game.type}')
                game_index.append(game.datetime)

                win_index = 0 if game.result == 'Win' else 1
                opponent = game.opponent_abbr

                if opponent.upper() in teams:
                    opponent_players = teams[opponent.upper()]
                else:
                    logger.info(f'\t\t\t\tSkipping, opponent not in teams')
                    continue

                try:
                    boxscore = game.boxscore
                except Exception as e:
                    logger.info('\t\t\t\tSkipping, can\'t get boxscore')
                    continue

                home_players = [list(p.dataframe.index)[0] for p in boxscore.home_players]
                away_players = [list(p.dataframe.index)[0] for p in boxscore.away_players]

                # Find if team is home or away
                try:
                    if home_players[0] in team_players:
                        game_team_players = [team_players[p] for p in home_players]
                        game_opponent_players = [opponent_players[p] for p in away_players]
                    else:
                        game_team_players = [team_players[p] for p in away_players]
                        game_opponent_players = [opponent_players[p] for p in home_players]
                except:
                    # This is for players like Travonta Doolittle
                    # For some reason on one of the pages his name is written Travonte
                    # And there's Quoiren Waldon, where the boxscore had it as Walden
                    # Instead of dealing with each individually, we're going to throw out the game
                    logger.info('\t\t\t\tSkipping, bad player name')
                    continue

                # Pad with zeros if they don't have 12 players
                for i in range(12 - len(game_team_players)):
                    game_team_players.append(np.zeros_like(game_team_players[0]))
                
                for i in range(12 - len(game_opponent_players)):
                    game_opponent_players.append(np.zeros_like(game_opponent_players[0]))
                
                # Concatenate player data, then add on conference id
                team_concat = np.concatenate(game_team_players[:12], axis=1)
                opponent_concat = np.concatenate(game_opponent_players[:12], axis=1)

                opponent_conference = np.expand_dims(np.array([int(opponent.lower() in c) for c in conference_teams]), axis=0)

                team_concat = np.concatenate((team_concat, team_conference), axis=1)
                opponent_concat = np.concatenate((opponent_concat, opponent_conference), axis=1)

                # Make input and output
                game_input = np.concatenate((team_concat, opponent_concat), axis=1)
                game_output = np.zeros((1, 2))
                game_output[0][win_index] = 1

                if game.type == None or game.type.lower() != 'ncaa':
                    # Regular season games
                    inputs.append(game_input)
                    outputs_results.append(game_output)
                else:
                    # NCAA tournament games
                    inputs_ncaa.append(game_input)
                    outputs_results_ncaa.append(game_output)
                
                sleep(0.5)
            sleep(1)

        year_inputs = np.concatenate(inputs, axis=0)
        year_outputs = np.concatenate(outputs_results, axis=0)

        np.savez(f'data/data_{year}.npz', inputs=year_inputs, outputs_results=year_outputs)

        if len(inputs_ncaa) != 0:
            year_inputs_ncaa = np.concatenate(inputs_ncaa, axis=0)
            year_outputs_ncaa = np.concatenate(outputs_results_ncaa, axis=0)

            np.savez(f'data/data_{year}_ncaa.npz', inputs=year_inputs_ncaa, outputs_results=year_outputs_ncaa)

        # Check if this is the last year
        if year == end_year:
            break

        year -= 1
        sleep(1)
