import yaml
import PySimpleGUI as sg
from math import ceil

def display_tournament(teams, results, first_four=True):
    teams = teams.strip().split('\n\n')
    results = results.strip().split('\n\n')

    sg.theme('DarkBlack1')
    initial_teams = teams[0].split('\n')
    layout = [[] for i in range(int(len(initial_teams) / 2))]
    text_width = 15

    # Left side
    for t in range(int(len(initial_teams) / 2)):
        layout[t].append(sg.Text(initial_teams[t].capitalize(), size=(text_width, 1)))

    for r in range(len(results)):
        next_layer = results[r].split('\n')
        buffer = int((len(initial_teams) / 4) - (len(next_layer) / 2))

        # Buffer top
        for b in range(buffer):
            layout[b].append(sg.Text('', size=(text_width, 1)))

        # Results
        for t in range(ceil(len(next_layer) / 2)):
            layout[(t * 2) + buffer].append(sg.Text('', size=(text_width, 1)))
            layout[(t * 2) + buffer + 1].append(sg.Text(next_layer[t].capitalize(), 
                size=(text_width, 1)))

        # Buffer bottom
        for b in range(buffer):
            layout[-(b + 1)].append(sg.Text('', size=(text_width, 1)))

    # Right side
    for r in reversed(range(len(results) - 1)):
        next_layer = results[r].split('\n')
        half_layer = int(len(next_layer) / 2)
        buffer = int((len(initial_teams) / 4) - half_layer)

        # Buffer top
        for b in range(buffer):
            layout[b].append(sg.Text('', size=(text_width, 1)))

        # Results
        for t in range(ceil(len(next_layer) / 2)):
            layout[(t * 2) + buffer].append(sg.Text('', size=(text_width, 1)))
            layout[(t * 2) + buffer + 1].append(sg.Text(next_layer[t + half_layer].capitalize(), 
                size=(text_width, 1)))

        # Buffer bottom
        for b in range(buffer):
            layout[-(b + 1)].append(sg.Text('', size=(text_width, 1)))

    for t in range(int(len(initial_teams) / 2)):
        layout[t].append(sg.Text(initial_teams[t + int(len(initial_teams) 
            / 2)].capitalize(), size=(text_width, 1), justification='right'))

    # Create window
    window = sg.Window("DEFYTHEODDS Visualization", layout)

    # Run the Event Loop
    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED:
            break

    window.close()

if __name__ == '__main__':
    # Get config
    with open('setup/config.yml') as file:
        config = yaml.safe_load(file)

    use_true_results = config['visual']['use_true_results']
    checkpoint_path = config['simulate']['checkpoint_path']
    results_file_name = checkpoint_path.replace('networks/', '')[:-5]

    # Get results and teams
    if use_true_results:
        with open('setup/true_results.txt') as file:
            results = file.read()
    else:
        with open(f'results/{results_file_name}.txt') as file:
            results = file.read()

    with open('setup/tournament_teams.txt') as file:
        teams = file.read()

    display_tournament(teams, results)
