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
from src.RL_old import BackgammonNet

class RL_player_old(Strategy):
    @staticmethod
    def get_difficulty():
        return "shimy"
    def __init__(self):
        self.boardHistory= None

        
            
    def move(self, board, color, dice_roll, make_move, opponents_activity):
        #self.boardHistory_opp = np.append(board.export_state(),self.boardHistory_opp)

        # Record starting time and compute a time limit (if necessary)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BackgammonNet().to(device)

        try:
            model.load_state_dict(torch.load("RLNN_old.pth", map_location=device))
            model.eval()
            #print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")

        # Assume there is a function that returns all legal move sequences given board, color, dice_roll.
        # Each candidate is (for example) a list of moves, where each move is stored as a dict.
        candidate_moves = self.get_all_possible_moves(board, color, dice_roll)
        if not candidate_moves:
            return
        best_score = -float('inf')
        best_move_sequence = None
        move_scores = []
        for simulated_board, move_seq in candidate_moves.items():
            simulated_board = simulated_board.export_state()
            input_data = np.concatenate([simulated_board, np.array([(color.value+1)%2], dtype=np.float32)])
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
            input_tensor = input_tensor.to(next(model.parameters()).device)
            with torch.no_grad():
                output = model(input_tensor).item()
            move_scores.append((output, move_seq))

        # Sort moves by score in descending order and select the top 4
        move_scores.sort(reverse=True, key=lambda x: x[0])
        top_moves = move_scores[:4]
        if top_moves:
            scores = [np.exp(score) for score, _ in top_moves]
            total_score = sum(scores)
            probabilities = [score / total_score for score in scores]
            best_move_sequence = random.choices([move_seq for _, move_seq in top_moves], weights=probabilities, k=1)[0]
            

            if best_move_sequence and isinstance(best_move_sequence, list):
                for move_dict in best_move_sequence:
                    make_move(move_dict['piece_at'], move_dict['die_roll'])
       # self.boardHistory_my = np.append(board.export_state(),self.boardHistory_my)
    
                    
        
                
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
    
