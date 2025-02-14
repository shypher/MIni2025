from src.strategies import Strategy
from src.piece import Piece
from fractions import Fraction
import time
import threading
from itertools import permutations
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from src.strategy_factory import StrategyFactory
class BackgammonNet(Strategy):
    
    def __init__(self, input_size):
        super(BackgammonNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 40)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(40, 40)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(40, 1)
        self.sigmoid = nn.Sigmoid() 
        self.softmax = nn.Softmax(dim=1)
    
    @staticmethod
    def get_difficulty():
        return "shimishimiya"

    def move(self, board, colour, dice_roll, make_move, opponents_activity):
        #wait 2 seconds before moving
        
        could_not_move_first_roll = False

        for i, die_roll in enumerate(dice_roll):
            moved = self.move_die_roll(board, colour, die_roll, make_move)
            if not moved and i == 0:
                could_not_move_first_roll = True

        if could_not_move_first_roll:
            self.move_die_roll(board, colour, dice_roll[0], make_move)

    @staticmethod
    def move_die_roll(board, colour, die_roll, make_move):
        valid_pieces = board.get_pieces(colour)
        valid_pieces.sort(key=Piece.spaces_to_home, reverse=True)
        for piece in valid_pieces:
            if board.is_move_possible(piece, die_roll):
                make_move(piece.location, die_roll)
                return True

        return False
