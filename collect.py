import defy_logging
import json
import os
import numpy as np
from sportsreference.ncaab.conferences import Conference
from sportsreference.ncaab.teams import Teams
from sportsreference.ncaab.schedule import Schedule
from time import sleep


year = 2021
end_year = 2016


def get_conference_teams():
    """Get all teams for each conference in the current year"""
    conference_teams = []

    for conference_name in conference_names:
        conference = Conference(conference_name.lower(), str(year))
        conference_teams.append(list(conference.teams))
    
    return conference_teams


def get_teams():
    teams = {}

    for team in Teams(str(year)):
        team_name = team.abbreviation
        roster = {}
        remove_indices = [5, 20, 28, 31, 34]  # Indices that are strings

        for player in team.roster.players:
            player_data = player.dataframe.loc[f'{year - 1}-{str(year)[2:]}'].to_numpy()
            player_fixed_data = np.expand_dims(np.delete(player_data, remove_indices), axis=0)
            roster[player.player_id] = player_fixed_data.tolist()
    
        teams[team_name] = roster
        sleep(1)
    
    return teams


if __name__ == '__main__':
    logger = defy_logging.get_logger()

    # Network input and output variables
    inputs = []
    outputs_results = []

    with open('setup/conferences.txt') as file:
        conference_names = file.read().split('\n')

    # Get data
    while True:
        logger.info(year)

        # Get all conference teams and team data
        if os.path.isfile(f'data/conferences_{year}.json'):
            logger.info('\tConference file exists, loading...')
            with open(f'data/conferences_{year}.json') as file:
                conference_teams = json.load(file)
        else:
            logger.info('\tConference file does not exist, gathering data from API...')
            conference_teams = get_conference_teams()

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
            teams = get_teams()

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

                if year == 2021 and game.type.lower() == 'ncaa':
                    # For testing purposes, we won't include March Madness games of 2021
                    continue

                logger.info(f'\t\t\t{game.datetime}')
                game_index.append(game.datetime)

                win_index = 0 if game.result == 'Win' else 1
                opponent = game.opponent_abbr

                if opponent.upper() in teams:
                    opponent_players = teams[opponent.upper()]
                else:
                    logger.info(f'\t\t\t\tSkipping, opponent not in teams')
                    continue

                boxscore = game.boxscore
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

                inputs.append(game_input)
                outputs_results.append(game_output)
                sleep(0.5)
            sleep(1)

        year_inputs = np.concatenate(inputs, axis=0)
        year_outputs = np.concatenate(outputs_results, axis=0)

        np.savez(f'data/data_{year}.npz', inputs=year_inputs, outputs_results=year_outputs)

        # Check if this is the last year
        if year == end_year:
            break

        year -= 1
        sleep(1)
