import check
import pickle
import sys
import yaml

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvas

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtWidgets import QWidget
from PyQt6.QtWidgets import QLabel
from PyQt6.QtWidgets import QPushButton
from PyQt6.QtWidgets import QDialog
from PyQt6.QtWidgets import QVBoxLayout
from PyQt6.QtWidgets import QHBoxLayout


class MainWindow(QMainWindow):
    def __init__(self, results, true_results, results_probs, round_points, max_round_points) -> None:
        super().__init__()

        self.setWindowTitle('Defy')
        self.setMinimumSize(800, 450)

        main_widget = QWidget()
        layout = QHBoxLayout()
        main_widget.setLayout(layout)

        graph = Graph()
        bracket = Bracket(results, true_results, results_probs, round_points,
            max_round_points, graph.set_data)

        layout.addWidget(bracket, 2)
        layout.addWidget(graph)

        self.setCentralWidget(main_widget)


class Bracket(QWidget):
    def __init__(self, results, true_results, results_probs, round_points, max_round_points, set_data):
        super().__init__()
        main_layout = QVBoxLayout()

        # Add bracket
        bracket_layout = QHBoxLayout()

        # Left side
        for r in range(6):
            round_layout = QVBoxLayout()

            for g in range(2 ** (5 - r)):
                round_layout.addWidget(
                    Team(
                        results[r][g],
                        true_results[r][g],
                        results_probs[r][g],
                        set_data
                    )
                )

                if r != 0 and g != (2 ** (5 - r) - 1):
                    round_layout.addWidget(Space(''))

            # Add on round points
            bracket_layout.addLayout(round_layout)
        
        # Champion
        round_layout = QVBoxLayout()
        round_layout.addWidget(
            Team(
                results[-1][0],
                true_results[-1][0],
                results_probs[-1][0],
                set_data
            )
        )
        bracket_layout.addLayout(round_layout)

        # Right side
        for r in range(6):
            round_layout = QVBoxLayout()

            for g in range(2 ** r):
                round_layout.addWidget(
                    Team(
                        results[5 - r][g + (2 ** r)],
                        true_results[5 - r][g + (2 ** r)],
                        results_probs[5 - r][g + (2 ** r)],
                        set_data
                    )
                )

                if r != 5 and g != (2 ** r - 1):
                    round_layout.addWidget(Space(''))

            bracket_layout.addLayout(round_layout)

        main_layout.addLayout(bracket_layout)

        # Add points
        point_layout = QHBoxLayout()
        point_layout.addWidget(Space('---/---'))

        for r in range(6):
            point_layout.addWidget(Space(f'{round_points[r]}/{max_round_points[r]}'))
        
        for r in range(6):
            point_layout.addWidget(Space('---/---'))
        
        main_layout.addLayout(point_layout)
        
        self.setLayout(main_layout)
        

class Graph(QDialog):
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout()
        self.canvas = None
        self.setLayout(self.layout)
    
    def set_data(self, results_probs, team, true_team):
        if self.canvas is not None:
            self.layout.removeWidget(self.canvas)
            plt.close()

        # If results_probs is empty, initialize nothing
        if len(results_probs) == 0:
            return

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        # Get data points
        reversed_results = dict((prob, team) for team, prob in results_probs.items())
        sorted_probs = list(reversed(sorted(list(reversed_results))))
        top_probs = sorted_probs[:10]

        x = list(range(len(top_probs)))
        tick_names = [display_names[reversed_results[prob]] for prob in top_probs]

        # Plot values
        ax = self.figure.add_subplot(111)
        ax.clear()
        bar_list = ax.bar(x, top_probs, color='black', width=0.75)

        # Change colors based off of what was the correct pick
        team_index = tick_names.index(team)
        bar_list[team_index].set_color('darkred')

        true_team_index = tick_names.index(true_team)
        bar_list[true_team_index].set_color('steelblue')

        plt.xticks(x, tick_names, fontsize=12, rotation=45)

        # Set the layout
        self.layout.addWidget(self.canvas)


class Team(QPushButton):
    def __init__(self, team, true_team, results_probs, set_data) -> None:
        # Check if result is correct
        correct_result = team == true_team

        # Find display name
        if team in display_names:
            team = display_names[team]
        
        if true_team in display_names:
            true_team = display_names[true_team]

        self.team = team
        self.true_team = true_team

        super().__init__(team)
        self.setAutoFillBackground(True)
        self.setMaximumSize(120, 40)

        # Style button and text
        self.setStyleSheet(f'''
            font-size: 12px;
            background-color: {'steelblue' if correct_result else 'darkred'};
            color: white;
        ''')

        # Set up graphing probabilities
        self.results_probs = results_probs
        self.set_data = set_data
        self.pressed.connect(self.display_probs)
    
    def display_probs(self):
        self.set_data(self.results_probs, self.team, self.true_team)


class Space(QLabel):
    def __init__(self, text) -> None:
        super().__init__(text)

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet('font-weight: bold')
        self.setMaximumSize(120, 40)


if __name__ == '__main__':
    # Get config and lookup table
    with open('setup/config.yml') as file:
        config = yaml.safe_load(file)

    # use_true_results = config['visual']['use_true_results']
    checkpoint_path = config['simulate']['checkpoint_path']
    results_folder_name = checkpoint_path.replace('networks/', '')
    results_file_name = config['visual']['results_path']
    use_epoch = config['simulate']['use_epoch']

    with open('setup/display_teams.txt') as file:
        display_names = {r.split('\t')[0]: r.split('\t')[1] for r in file.read().split('\n')}

    # Get results and teams
    results = check.get_predicted_results()

    with open('setup/true_results.txt') as file:
        true_results = [r.split('\n') for r in file.read().split('\n\n')]

    with open('setup/tournament_teams.txt') as file:
        teams = file.read().split('\n')
    
    # Get pickled probabilities
    with open(f'results/{results_folder_name}/{use_epoch:04d}_probs.txt', 'rb') as file:
        results_probs = pickle.load(file)

    # Add on pre-first round
    results.insert(0, teams)
    true_results.insert(0, teams)

    initial_probs = [{team: 1} for team in results[0]]
    results_probs.insert(0, initial_probs)

    # Get round points
    round_points, max_round_points = check.get_round_points()

    # Create main window
    app = QApplication.instance()

    if not app:
        app = QApplication(sys.argv)

    window = MainWindow(results, true_results, results_probs, round_points, max_round_points)
    window.show()

    sys.exit(app.exec())
