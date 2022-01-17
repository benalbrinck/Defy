
with open('tournament_results.txt') as file:
	predicted_results = [r.split('\n') for r in file.read().split('\n\n')[:-1]]

with open('true_results.txt') as file:
	true_results = [r.split('\n') for r in file.read().split('\n\n')]

# Give point total + show them both side by side w/ whether or not right there
amount_right = 0
points = 0
max_points = 0

for r in range(len(predicted_results)):
	for g in range(len(predicted_results[r])):
		right_bool = int(predicted_results[r][g] == true_results[r][g])
		amount_right += right_bool

		if r != 0:
			# Add on points if not the play-in games
			points += ((2 ** (r - 1)) * 10) * right_bool
			max_points += (2 ** (r - 1)) * 10

		print(f'{predicted_results[r][g]}\t{right_bool}\t{true_results[r][g]}')
	print('')

print(f'\nPOINTS: {points}/{max_points}\nAMOUNT RIGHT: {amount_right}/63')
