# https://www.youtube.com/watch?v=SelawmXHtPg
# Modern: https://github.com/Wanderson-Magalhaes/Modern_GUI_PyDracula_PySide6_or_PyQt6

# steelblue
# darkslategray
# slategray
# black
# white
# darkred


import sys
import yaml
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtWidgets import QWidget
from PyQt6.QtWidgets import QLabel
from PyQt6.QtWidgets import QPushButton
from PyQt6.QtWidgets import QTextEdit
from PyQt6.QtWidgets import QVBoxLayout
from PyQt6.QtWidgets import QHBoxLayout
from PyQt6.QtGui import QPalette
from PyQt6.QtGui import QColor


class MainWindow(QMainWindow):
    def __init__(self, results, true_results) -> None:
        super().__init__()

        self.setWindowTitle('Defy')
        self.setMinimumSize(800, 450)

        main_layout = QHBoxLayout()

        # Left side
        for r in range(6):
            round_layout = QVBoxLayout()
            margin = 0
            round_layout.setContentsMargins(0, margin, 0, margin)

            for g in range(2 ** (5 - r)):
                round_layout.addWidget(Team(results[r][g], true_results[r][g]))

                if r != 0 and g != (2 ** (5 - r) - 1):
                    round_layout.addWidget(Space())

            main_layout.addLayout(round_layout)
        
        round_layout = QVBoxLayout()
        round_layout.setContentsMargins(0, 2 ** 5, 0, 2 ** 5)
        round_layout.addWidget(Team(results[-1][0], true_results[-1][0]))
        main_layout.addLayout(round_layout)

        # Right side
        for r in range(6):
            round_layout = QVBoxLayout()
            round_layout.setContentsMargins(0, 2 ** (6 - r), 0, 2 ** (6 - r))

            for g in range(2 ** r):
                round_layout.addWidget(Team(results[5 - r][g + (2 ** r)], true_results[5 - r][g + (2 ** r)]))

                if r != 5 and g != (2 ** r - 1):
                    round_layout.addWidget(Space())

            main_layout.addLayout(round_layout)

        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)


class Team(QLabel):
    def __init__(self, team, true_team) -> None:
        # Check if result is correct
        correct_result = team == true_team

        # Find display name
        if team in display_names:
            team = display_names[team]

        super().__init__(team)
        self.setAutoFillBackground(True)
        self.setMaximumSize(120, 40)

        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor('steelblue' if correct_result else 'darkred'))
        self.setPalette(palette)

        # Add text
        self.setStyleSheet('color: white;')
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)


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

    results.insert(0, teams)
    true_results.insert(0, teams)

    app = QApplication(sys.argv)

    window = MainWindow(results, true_results)
    window.show()

    sys.exit(app.exec())
