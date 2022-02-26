import pickle
import random
import sys
import yaml

import matplotlib.pyplot as plt
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtWidgets import QWidget
from PyQt6.QtWidgets import QLabel
from PyQt6.QtWidgets import QPushButton
from PyQt6.QtWidgets import QDialog
from PyQt6.QtWidgets import QCheckBox
from PyQt6.QtWidgets import QVBoxLayout
from PyQt6.QtWidgets import QHBoxLayout
from PyQt6.QtWidgets import QGridLayout
from PyQt6.QtGui import QPalette
from PyQt6.QtGui import QColor


class MainWindow(QMainWindow):
    def __init__(self, results, true_results, results_probs) -> None:
        super().__init__()

        self.setWindowTitle('Defy')
        self.setMinimumSize(800, 450)

        main_widget = QWidget()
        layout = QHBoxLayout()
        main_widget.setLayout(layout)

        graph = Graph()
        bracket = Bracket(results, true_results, results_probs, graph.set_data)

        layout.addWidget(bracket, 2)
        layout.addWidget(graph)

        self.setCentralWidget(main_widget)


class Bracket(QWidget):
    def __init__(self, results, true_results, results_probs, set_data):
        super().__init__()
        bracket_layout = QHBoxLayout()

        # Left side
        for r in range(6):
            round_layout = QVBoxLayout()
            margin = 0
            round_layout.setContentsMargins(0, margin, 0, margin)

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
                    round_layout.addWidget(Space())

            bracket_layout.addLayout(round_layout)
        
        # Champion
        round_layout = QVBoxLayout()
        round_layout.setContentsMargins(0, 2 ** 5, 0, 2 ** 5)
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
            round_layout.setContentsMargins(0, 2 ** (6 - r), 0, 2 ** (6 - r))

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
                    round_layout.addWidget(Space())

            bracket_layout.addLayout(round_layout)

        self.setLayout(bracket_layout)
        

class Graph(QDialog):
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout()
        self.canvas = None
        self.setLayout(self.layout)
    
    def set_data(self, results_probs):
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
        ax.bar(x, top_probs, color='b', width=0.75)
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
        self.set_data(self.results_probs)


class Space(QLabel):
    def __init__(self) -> None:
        super().__init__('')
        self.setMaximumSize(120, 40)


if __name__ == '__main__':
    # Get config and lookup table
    with open('setup/config.yml') as file:
        config = yaml.safe_load(file)

    # use_true_results = config['visual']['use_true_results']
    checkpoint_path = config['simulate']['checkpoint_path']
    results_file_name = checkpoint_path.replace('networks/', '')[:-5]

    with open('setup/display_teams.txt') as file:
        display_names = {r.split('\t')[0]: r.split('\t')[1] for r in file.read().split('\n')}

    # Get results and teams
    with open(f'results/{results_file_name}.txt') as file:
        results = [r.split('\n') for r in file.read().split('\n\n')]

    with open('setup/true_results.txt') as file:
        true_results = [r.split('\n') for r in file.read().split('\n\n')]

    with open('setup/tournament_teams.txt') as file:
        teams = file.read().split('\n')
    
    # Get pickled probabilities
    with open(f'results/{results_file_name}_probs.txt', 'rb') as file:
        results_probs = pickle.load(file)

    results.insert(0, teams)
    true_results.insert(0, teams)

    initial_probs = [{team: 1} for team in results[0]]
    results_probs.insert(0, initial_probs)

    # Create main window
    app = QApplication.instance()

    if not app:
        app = QApplication(sys.argv)

    window = MainWindow(results, true_results, results_probs)
    window.show()

    sys.exit(app.exec())

# Move graph to its own class?
# Recolor the plot
# Display scores (for each round?)
