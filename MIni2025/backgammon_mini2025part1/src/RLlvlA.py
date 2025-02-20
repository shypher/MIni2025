from itertools import permutations
import random
from src.strategies import Strategy
from src.piece import Piece
from fractions import Fraction
import time
import torch
import numpy as np
import time
from src.strategies import Strategy
from src.piece import Piece
from fractions import Fraction
import time
import threading
from itertools import permutations
import numpy as np 
import torch
import torch.nn as nn
from torch.utils.data import random_split
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from src.RL import BackgammonNet

class RLlvlA(Strategy):
    @staticmethod
    def get_difficulty():
        return "shimy"
    def __init__(self):
        self.saved_tree = None
        


 
    def move(self, board, color, dice_roll, make_move, opponents_activity):

        # Record starting time and compute a time limit (if necessary)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BackgammonNet().to(device)

        try:
            model.load_state_dict(torch.load("RLNNlvlA.pth", map_location=device))
            model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")

        # Assume there is a function that returns all legal move sequences given board, color, dice_roll.
        # Each candidate is (for example) a list of moves, where each move is stored as a dict.
        candidate_moves = self.get_all_possible_moves(board, color, dice_roll)
        if not candidate_moves:
            return
        best_score = -float('inf')
        best_move_sequence = None
        for simulated_board,move_seq in candidate_moves.items():
            simulated_board = board.export_state()
            input_data = np.concatenate([simulated_board, np.array([color.value], dtype=np.float32)])
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
            input_tensor = input_tensor.to(next(model.parameters()).device)
            with torch.no_grad():
                output = model(input_tensor).item()
            if output > best_score:
                best_score = output
                best_move_sequence = move_seq
        if best_move_sequence and isinstance(best_move_sequence, list):
            for move_dict in best_move_sequence:
                make_move(move_dict['piece_at'], move_dict['die_roll'])
     
    def get_all_possible_moves(self, board, color, dice_roll):
        if len(dice_roll) == 0:
            return {}
        all_possible_moves = {}
        pieces_to_try = [x.location for x in board.get_pieces(color)]
        pieces_to_try = list(set(pieces_to_try))
            
        for dice in (set(permutations(dice_roll))):
            dice_roll = dice[0]
            remaining_dice = dice[1:]
            for piece_curr in pieces_to_try:
                piece = board.get_piece_at(piece_curr)
                if board.is_move_possible(piece, dice_roll):
                    board_copy = board.create_copy()
                    new_piece = board_copy.get_piece_at(piece.location)
                    board_copy.move_piece(new_piece, dice_roll)
                    rest_dice_board = self.get_all_possible_moves(board_copy, color, remaining_dice)
                    if len(rest_dice_board) == 0:
                        all_possible_moves[board_copy] = [{'piece_at': piece.location, 'die_roll': dice_roll}]
                    else:
                        for new_board, moves in rest_dice_board.items():
                            all_possible_moves[new_board] = [{'piece_at': piece.location, 'die_roll': dice_roll}] + moves
        return all_possible_moves
    
