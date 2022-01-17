import numpy as np
from sportsreference.ncaab.conferences import Conference
from sportsreference.ncaab.teams import Teams
from sportsreference.ncaab.schedule import Schedule
from statistics import mean, stdev
from time import sleep

# Parameters
year = 2021
end_year = 2016

use_conference_id = True
use_n_players_stats = True
use_players_summary = True

n_players = 10
roster_vars = ['assist_percentage', 'assists', 'block_percentage', 'blocks', 
				'defensive_rebound_percentage', 'defensive_rebounds', 
				'effective_field_goal_percentage', 'field_goal_attempts', 
				'field_goal_percentage', 'field_goals', 'free_throw_attempt_rate', 
				'free_throw_attempts', 'free_throw_percentage', 'free_throws', 
				'minutes_played', 'offensive_rebound_percentage', 
				'offensive_rebounds', 'personal_fouls', 'points', 'steal_percentage', 
				'steals', 'three_point_attempt_rate', 'three_point_attempts', 
				'three_point_percentage', 'three_pointers', 'total_rebounds', 
				'true_shooting_percentage', 'turnover_percentage', 'turnovers', 
				'two_point_attempts', 'two_point_percentage', 'two_pointers', 
				'usage_percentage']

# Network input and output variables
inputs = []
outputs_results = []
outputs_scores = []

conference_names = open('conferences.txt').read().split('\n')

# Loop through each year
while True:
	print(year)
	game_index = []  # Keep track of which games were played

	if use_conference_id:
		# Get all conference teams
		conference_teams = []

		for conference_name in conference_names:
			conference = Conference(conference_name.lower(), str(year))
			conference_teams.append(list(conference.teams.values()))

	# Get all team statistics and append one-hot vector of conference id
	all_team_names = []
	all_team_stats_list = []

	for team in Teams(str(year)):
		team_name = team.name
		team_values_list = []

		# Team stats
		try:
			team_stats = team.dataframe.to_numpy()
			team_stats = np.where(team_stats == None, 0, team_stats)
			team_stats[0][0] = 0
			team_stats[0][7] = 0
			team_stats[0][25] = 0
		except Exception as e:
			continue

		all_team_names.append(team.abbreviation.lower())
		team_values_list.append(team_stats)

		# Conference id
		if use_conference_id:
			conference_id = len(conference_names) + 0
			for i in range(len(conference_teams)):
				if team_name in conference_teams[i]:
					conference_id = i + 0
					break
				continue

			conference_id_array = np.zeros((1, len(conference_names) + 1))
			conference_id_array[0][conference_id] = 1
			team_values_list.append(conference_id_array)

		if use_n_players_stats or use_players_summary:
			roster = team.roster.players
			
			roster_stats = []
			for p in roster:
				roster_stats.append([])
				for v in roster_vars:
					roster_stats[-1].append(eval(f'p.{v}'))

			if use_n_players_stats:
				n_players_stats_list = roster_stats[:n_players]
				n_players_stats = np.concatenate(n_players_stats_list, axis=1)
				team_values_list.append(n_players_stats)
			
			if use_players_summary:
				player_summary_list = []
				for s in range(len(roster_stats[0])):
					stats = [p[s] for p in roster_stats]
					mean_value = mean(stats)
					stdev_value = stdev(stats)
					player_summary_list.extend([mean_value, stdev_value])
				team_values_list.append(np.array(player_summary_list))

		# Concatenate all team values
		all_team_stats_list.append(np.concatenate(team_values_list, axis=1))

	quit()

	# Normalize team stats and convert back to list
	# all_team_stats_array = normalize(np.concatenate(all_team_stats_list, axis=0))
	all_team_stats_array = np.concatenate(all_team_stats_list, axis=0)
	all_team_stats_list = [np.expand_dims(i, axis=0) for i in all_team_stats_array]

	if year == 2021:
		# Save current teams and their statistics
		open('current_teams.txt', 'w').write('\n'.join(all_team_names))
		np.savez('current_team_stats.npz', team_stats=all_team_stats_array)

	# Go through each team's schedule to create data
	for i in range(len(all_team_names)):
		try:
			schedule = Schedule(all_team_names[i].lower(), str(year))
			games_data = schedule.dataframe.to_numpy()
		except Exception as e:
			print(all_team_names[i])
			continue

		opponents = [game[6].lower() for game in games_data]
		team_score = [game[12] for game in games_data]
		opponent_score = [game[11] for game in games_data]
		results = [0 if game[13] == 'Win' else 1 for game in games_data]
		indices = [str(game[3]) for game in games_data]

		# Scrub data
		while True:
			if team_score[-1] == None:
				opponents.pop()
				team_score.pop()
				opponent_score.pop()
				results.pop()
			else:
				break
			continue

		# Create inputs and outputs
		for j in range(len(opponents)):
			try:
				opponent_index = all_team_names.index(opponents[j])
			except Exception as e:
				sleep(1)
				continue

			if indices[j] in game_index:
				# Game already in data
				continue

			inputs.append(np.concatenate((all_team_stats_list[i], all_team_stats_list[opponent_index]), axis=1))
			outputs_scores.append(np.array([[team_score[j], opponent_score[j]]]))
			outputs_results.append(np.zeros((1, 2)))
			outputs_results[-1][0][results[j]] = 1
			game_index.append(indices[j] + '')
		continue

	# Check if this is the last year
	if year == end_year:
		break

	year -= 1

# Concatenate each list
input_array = np.concatenate(inputs, axis=0)
output_results_array = np.concatenate(outputs_results, axis=0)
output_scores_array = np.concatenate(outputs_scores, axis=0)

# Save the network's inputs and outputs
np.savez('data.npz', inputs=input_array, outputs_results=output_results_array, outputs_scores=output_scores_array)
