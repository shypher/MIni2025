from matplotlib import pyplot as plt
import torch
from src.colour import Colour
from src.compare_all_moves_strategy import CompareAllMovesWeightingDistanceAndSinglesWithEndGame
from src.game2 import Game
from src.strategies import MoveRandomPiece
from random import randint
from scipy.special import erf
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from src.RL import BackgammonNet
from src.strategies import MoveRandomPiece
from src.RL_player_old import RL_player_old
from src.RL_old import BackgammonNet 
from random import randint
import numpy as np


class part_1_stat_maker:
    def init(self, target_boards=200, boards_per_game=50, min_games=4):
        self.target_boards = target_boards
        self.boards_per_game = boards_per_game
        self.min_games = min_games
        # נניח שיש לנו רשת ניורונים לאימון שמחושבת על לוח.
        # יש לממש את המימוש המתאים או להשתמש במחלקה קיימת
        self.network = BackgammonNet() # יש לשנות בהתאם למימוש הקיים שלכם
    def evaluate_network_value(self, board_state):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BackgammonNet().to(device)
        try:
            model.load_state_dict(torch.load("RLNN_finalLvlA.pth", map_location=device))
            model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
            return np.random.random()

    def run_evaluation(self):
        differences = []  
        game_count = 0
        while (len(differences) < self.target_boards) or (game_count < self.min_games):
            game_count += 1
            for idx, entry in enumerate(game_history[:self.boards_per_game]):
                heuristic_value = entry.get('heuristic_score', None)
                if heuristic_value is None:
                    print(f"אזהרה: אין heuristic_score עבור לוח מספר {idx} במשחק {game_count}")
                    continue
                board_state = entry.get('board_state', None)
                if board_state is None:
                    print(f"אזהרה: אין board_state עבור לוח מספר {idx} במשחק {game_count}")
                    continue
                network_value = self.evaluate_network_value(board_state)
                diff = heuristic_value - network_value
                differences.append(diff)
                if len(differences) >= self.target_boards:
                    break
            print(f" {game_count}  {len(differences)}")
        differences_np = np.array(differences)
        avg_diff = np.mean(differences_np)
        var_diff = np.var(differences_np)

        print("\n-------------------------------------")
        print(f"Total boards {len(differences_np)}")
        print(f"Averege: {avg_diff}")
        print(f"Difference Variance: {var_diff}")
        print("-------------------------------------\n")
        return avg_diff, var_diff




if __name__ == "main":
    eval_runner = part_1_stat_maker(target_boards=200, boards_per_game=50, min_games=4)
    avg, var = eval_runner.run_evaluation()

""""
if __name__ == '__main__':
    num_of_board = 0 
    while (num_of_board<=200):
        game = Game(
            white_strategy= RL_player_old(),
            black_strategy= MoveRandomPiece(),
            first_player=Colour(randint(0, 1)),
            time_limit=-1
        )
        game.run_game(verbose=False)
        game_history = game.get_game_history()[1:]
     """   

        