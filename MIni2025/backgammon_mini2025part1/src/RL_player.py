from src.strategies import Strategy
from src.piece import Piece
from fractions import Fraction
import time
import torch
import numpy as np
import time

class RL_player(Strategytr):
    @staticmethod
    def get_difficulty():
        return "shimy"
    def __init__(self):
        self.saved_tree = None  
    def assess_board(self, colour, myboard):
        pieces = myboard.get_pieces(colour)
        opponent_pieces = myboard.get_pieces(colour.other())
        pieces_on_board = len(pieces)
        sum_distances = sum(piece.spaces_to_home() for piece in pieces)
        number_of_singles = sum(1 for loc in range(1, 25) if len(myboard.pieces_at(loc)) == 1)
        number_occupied_spaces =sum(1 for loc in range(1, 25) if len(myboard.pieces_at(loc)) > 1)
        sum_single_distance_away_from_home = sum(25 - pieces[0].spaces_to_home() for location in range(1, 25) if len((pieces := myboard.pieces_at(location)))\
            == 1 and pieces[0].colour == colour)
        sum_distances_to_endzone = sum(max(0, piece.spaces_to_home() - 6) for piece in pieces)
        threat_level = sum(1 for piece in pieces for opponent_piece in opponent_pieces if abs(piece.location - opponent_piece.location) < 8)
        """
        for piece in pieces:
            for opponent_piece in opponent_pieces:
                distance = abs(piece.location- opponent_piece.location)
                if distance < 8:   
                    threat_level += 1 
                #threat_level = chance_map(distance)*10 """
        opponents_taken_pieces = len(myboard.get_taken_pieces(colour.other()))
        opponent_pieces = myboard.get_pieces(colour.other())
        sum_distances_opponent = sum(piece.spaces_to_home() for piece in opponent_pieces)
        return {
            'number_occupied_spaces': number_occupied_spaces,
            'opponents_taken_pieces': opponents_taken_pieces,
            'sum_distances': sum_distances,
            'sum_distances_opponent': sum_distances_opponent,
            'number_of_singles': number_of_singles,
            'sum_single_distance_away_from_home': sum_single_distance_away_from_home,
            'pieces_on_board': pieces_on_board,
            'sum_distances_to_endzone': sum_distances_to_endzone,
            'threat level' : threat_level,
        }
        
    def move(self, board, colour, dice_roll, make_move, opponents_activity):
        start_time = time.time()
        time_limit = board.getTheTimeLim()-0.05
        best_move = self.minimax(board, colour, dice_roll, depth=3, maximizing=True,start_time=start_time, time_limit=time_limit)

        if best_move['best_moves']:
            for move in best_move['best_moves']:
                make_move(move['piece_at'], move['die_roll'])
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

    def move2(self, board, colour, dice_roll, make_move, opponents_activity):

        result = self.move_recursively(board, colour, dice_roll)
        not_a_double = len(dice_roll) == 2
        if not_a_double:
            new_dice_roll = dice_roll.copy()
            new_dice_roll.reverse()
            result_swapped = self.move_recursively(board, colour,
                                                   dice_rolls=new_dice_roll)
            if result_swapped['best_value'] < result['best_value'] and \
                    len(result_swapped['best_moves']) >= len(result['best_moves']):
                result = result_swapped

        if len(result['best_moves']) != 0:
            for move in result['best_moves']:
                make_move(move['piece_at'], move['die_roll'])
 
def move(self, board, colour, dice_roll, make_move, opponents_activity):

# Record starting time and compute a time limit (if necessary)
  start_time = time.time()
  time_limit = board.getTheTimeLim() - 0.05

  # Assume there is a function that returns all legal move sequences given board, colour, dice_roll.
  # Each candidate is (for example) a list of moves, where each move is stored as a dict.
  candidate_moves = board.get_legal_moves(colour, dice_roll)

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

      # Optional: You can also check elapsed time to avoid running over your limit.
      if (time.time() - start_time) > time_limit:
          print("Time limit reached during move selection.")
          break

  # Finally, if you found a best move sequence, apply all the moves.
  if best_move_sequence:
      for move_dict in best_move_sequence:
          # Here it is assumed that each move_dict contains at least the keys "piece_at" and "die_roll".
          make_move(move_dict['piece_at'], move_dict['die_roll'])