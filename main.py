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


# QtCore.Qt.AlignmentFlag

class MainWindow(QMainWindow):
    def __init__(self, results) -> None:
        super().__init__()

        self.setWindowTitle('Defy')
        self.setMinimumSize(800, 450)

        main_layout = QHBoxLayout()

        # Left side
        for r in range(6):
            round_layout = QVBoxLayout()
            round_layout.setContentsMargins(0, 2 ** r, 0, 2 ** r)

            for g in range(2 ** (5 - r)):
                round_layout.addWidget(Team(results[r][g]))

            main_layout.addLayout(round_layout)
        
        round_layout = QVBoxLayout()
        round_layout.setContentsMargins(0, 2 ** 5, 0, 2 ** 5)
        round_layout.addWidget(Team(results[-1][0]))
        main_layout.addLayout(round_layout)

        # Right side
        for r in range(6):
            round_layout = QVBoxLayout()
            round_layout.setContentsMargins(0, 2 ** (6 - r), 0, 2 ** (6 - r))

            for g in range(2 ** r):
                round_layout.addWidget(Team(results[5 - r][g + (2 ** r)]))

            main_layout.addLayout(round_layout)

        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)


class Team(QLabel):
    def __init__(self, team) -> None:
        super().__init__(team)
        self.setAutoFillBackground(True)
        self.setMaximumSize(120, 40)

        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor('steelblue'))
        self.setPalette(palette)

        # Add text
        # label = QLabel(team, self)
        self.setStyleSheet('color: white;')
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)


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
            results = [r.split('\n') for r in file.read().split('\n\n')]
    else:
        with open(f'results/{results_file_name}.txt') as file:
            results = [r.split('\n') for r in file.read().split('\n\n')]

    with open('setup/tournament_teams.txt') as file:
        teams = file.read().split('\n')

    results.insert(0, teams)
    print(results)
    print(len(results))

    app = QApplication(sys.argv)

    window = MainWindow(results)
    window.show()

    sys.exit(app.exec())
