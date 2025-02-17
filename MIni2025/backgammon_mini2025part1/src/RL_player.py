from itertools import permutations
from src.strategies import Strategy
from src.piece import Piece
from fractions import Fraction
import time
import torch
import numpy as np
import time

class RL_player(Strategy):
    @staticmethod
    def get_difficulty():
        return "shimy"
    def __init__(self):
        self.saved_tree = None  

    def RL_move(self, board, colour, dice_rolls, depth, maximizing, start_time, time_limit):
        if time.time() - start_time > time_limit:
            return {'best_value': self.evaluate_board(board, colour), 'best_moves': []}
        if depth == 0 or not dice_rolls:
            return {'best_value': self.evaluate_board(board, colour), 'best_moves': []}

        best_value = float('-inf') if maximizing else float('inf')
        best_moves = []

        pieces_to_try = [x.location for x in board.get_pieces(colour if maximizing else colour.other())]
        pieces_to_try = list(set(pieces_to_try))

        valid_pieces = [board.get_piece_at(loc) for loc in pieces_to_try]
        valid_pieces.sort(key=Piece.spaces_to_home, reverse=maximizing)

        dice_rolls_left = dice_rolls.copy()
        die_roll = dice_rolls_left.pop(0)

        for piece in valid_pieces:
            if board.is_move_possible(piece, die_roll):
                board_copy = board.create_copy()
                new_piece = board_copy.get_piece_at(piece.location)
                board_copy.move_piece(new_piece, die_roll)
                result = self.minimax(board_copy, colour, dice_rolls_left, depth - 1, not maximizing, start_time, time_limit)
                if maximizing and result['best_value'] > best_value:
                    best_value = result['best_value']
                    best_moves = [{'die_roll': die_roll, 'piece_at': piece.location}] + result['best_moves']
                elif not maximizing and result['best_value'] < best_value:
                    best_value = result['best_value']
                    best_moves = [{'die_roll': die_roll, 'piece_at': piece.location}] + result['best_moves']

        return {'best_value': best_value, 'best_moves': best_moves}


 
    def move(self, board, colour, dice_roll, make_move, opponents_activity):

        # Record starting time and compute a time limit (if necessary)
        start_time = time.time()
        time_limit = board.getTheTimeLim() - 0.05

        # Assume there is a function that returns all legal move sequences given board, colour, dice_roll.
        # Each candidate is (for example) a list of moves, where each move is stored as a dict.
        candidate_moves = self.get_all_possible_moves(board, colour, dice_roll)

        if not candidate_moves:
            # No legal move was found, so you might decide to pass or return an empty move.
            return

        best_score = -float('inf')
        best_move_sequence = None

        # Ensure the RL network is in evaluation mode; it is assumed that self.model is your trained RL network.
        self.model.eval()
        
        # Iterate over each candidate move sequence and evaluate the resulting board state using the RL network.
        for move_seq in candidate_moves:
            # Make a copy of the board to simulate moves.
            simulated_board = board.copy()

            # Simulate each move for the candidate sequence.
            for mv in move_seq:
                # It's assumed the board handle the simulation of individual moves.
                simulated_board.simulate_move(mv)  # Alternatively: simulated_board.make_move(move) if no real game change is intended

            # Generate an input vector for the RL network.
            # This should be exactly the same format you used when training your network.
            # For example, if your training data was generated like:
            #   np.concatenate([entry['board'], [entry['color']]])
            # then do the same here.
            board_representation = simulated_board.get_board_representation()  # e.g., a numpy array of shape (28,)
            input_vector = np.concatenate([board_representation, np.array([colour])]).astype(np.float32)

            # Convert the input to a torch tensor; add a batch dimension.
            input_tensor = torch.tensor(input_vector).unsqueeze(0)

            # Optionally, ensure the tensor is on the same device as your model.
            input_tensor = input_tensor.to(next(self.model.parameters()).device)

            # Get the RL network's predicted value for this board state.
            with torch.no_grad():
                score = self.model(input_tensor).item()  # Assumes the network returns a tensor of shape [1,1]

            # Check if this candidate produced a better score.
            if score > best_score:
                best_score = score
                best_move_sequence = move_seq

        # Finally, if you found a best move sequence, apply all the moves.
        if best_move_sequence:
            for move_dict in best_move_sequence:
                # Here it is assumed that each move_dict contains at least the keys "piece_at" and "die_roll".
                make_move(move_dict['piece_at'], move_dict['die_roll'])
                
    def get_all_possible_moves(self, board, colour, dice_roll):
        if len(dice_roll) == 0:
            return {}
        all_possible_moves = {}
        pieces_to_try = [x.location for x in board.get_pieces(colour)]
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
                    rest_dice_board = self.get_all_possible_moves(board_copy, colour, remaining_dice)
                    if len(rest_dice_board) == 0:
                        all_possible_moves[board_copy] = [{'piece_at': piece.location, 'die_roll': dice_roll}]
                    else:
                        for new_board, moves in rest_dice_board.items():
                            all_possible_moves[new_board] = [{'piece_at': piece.location, 'die_roll': dice_roll}] + moves
        return all_possible_moves