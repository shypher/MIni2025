from src.strategies import Strategy
from src.piece import Piece
from fractions import Fraction
import time
def log(message, file_path="tournament_log.txt"):
    with open(file_path, "a") as log_file:
        log_file.write(message + "\n")
    print(message)
class MinMax_shayOren(Strategy):
    
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
        """for piece in pieces:
            for opponent_piece in opponent_pieces:
                distance = abs(piece.location- opponent_piece.location)
                if distance < 8:   
                    threat_level += 1 """
                #threat_level = chance_map(distance)*10 
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
        log("minimax")
        start_time = time.time()
        time_limit = board.getTheTimeLim() - 0.01
        
        best_move = self.minmax(board, colour, dice_roll, depth=0, maximizing_player=True,alpha=float('-inf'), beta=float('inf'), start_time=start_time, time_limit=time_limit)
        if len(best_move['best_moves']) != 0:
            for move in best_move['best_moves']:
                make_move(move['piece_at'], move['die_roll'])
        else:
            log("No valid moves found.")   
        log(f"Best move: {best_move}")
        
    def minmax(self, board, colour, dice_rolls, depth, maximizing_player, alpha, beta, start_time, time_limit):
        if time.time() - start_time > time_limit:
            return {'best_value': self.evaluate_board(board, colour), 'best_moves': []}
        
        dice_rolls_left = dice_rolls.copy()  # Ensure we work with a copy of the list
        if not dice_rolls_left:
            return {'best_value': self.evaluate_board(board, colour), 'best_moves': []}
        
        # Process each die roll in the copied list
        for i, die_roll in enumerate(dice_rolls_left):
            # Create a new list of remaining dice rolls
            remaining_rolls = dice_rolls_left[:i] + dice_rolls_left[i+1:]
            new_dice_rolls = dice_rolls_left.copy()
            new_dice_rolls.remove(die_roll)
            valid_moves = self.get_valid_moves(board, colour, die_roll)
            if maximizing_player:
                best_value = float('-inf')
                best_moves = []
                for move in valid_moves:
                    board_copy = board.create_copy()
                    piece = board_copy.get_piece_at(move['piece_at'])
                    board_copy.move_piece(piece, die_roll)
                    
                    result = self.minmax(
                        board=board_copy,
                        colour=colour,
                        dice_rolls=new_dice_rolls,  # Pass remaining rolls
                        depth=depth + 1,
                        maximizing_player=False,
                        alpha=alpha,
                        beta=beta,
                        start_time=start_time,
                        time_limit=time_limit
                    )

                    if result['best_value'] > best_value:
                        best_value = result['best_value']
                        best_moves = [move] + result['best_moves']
                    alpha = max(alpha, best_value)
                    if beta <= alpha:
                        break
                return {'best_value': best_value, 'best_moves': best_moves}
            else:
                best_value = float('inf')
                best_moves = []
                for move in valid_moves:
                    board_copy = board.create_copy()
                    piece = board_copy.get_piece_at(move['piece_at'])
                    board_copy.move_piece(piece, die_roll)
                    
                    result = self.minmax(
                        board=board_copy,
                        colour=colour,
                        dice_rolls=remaining_rolls,  # Pass remaining rolls
                        depth=depth + 1,
                        maximizing_player=True,
                        alpha=alpha,
                        beta=beta,
                        start_time=start_time,
                        time_limit=time_limit
                    )

                    if result['best_value'] < best_value:
                        best_value = result['best_value']
                        best_moves = [move] + result['best_moves']
                    beta = min(beta, best_value)
                    if beta <= alpha:
                        break
                return {'best_value': best_value, 'best_moves': best_moves}
    def minmax2(self, board, colour, dice_rolls, depth, maximizing_player, alpha, beta, start_time, time_limit):
        if time.time() - start_time > time_limit:
            return {'best_value': self.evaluate_board(board, colour), 'best_moves': []}
        dice_rolls_left = dice_rolls.copy()
        if not dice_rolls_left:
            return {'best_value': self.evaluate_board(board, colour), 'best_moves': []}
        die_roll = dice_rolls_left.pop(0)
        valid_moves = self.get_valid_moves(board, colour, die_roll)
        if maximizing_player:
            best_value = float('-inf')
            best_moves = []
            for move in valid_moves:
                board_copy = board.create_copy()
                Piece = board_copy.get_piece_at(move['piece_at'])
                board_copy.move_piece(Piece, move['die_roll'])
                for move in valid_moves:
                    board_copy = board.create_copy()
                    piece = board_copy.get_piece_at(move['piece_at'])
                    board_copy.move_piece(piece, die_roll)
                    result = self.minmax(board_copy, colour, dice_rolls_left, depth + 1, False, alpha, beta, start_time, time_limit)
                    if result is not None:
                        if result['best_value'] > best_value:
                            best_value = result['best_value']
                            best_moves = [move] + result['best_moves']
                        alpha = max(alpha, best_value)
                        if beta <= alpha:
                            break
            return {'best_value': best_value, 'best_moves': best_moves}
        else:
            best_value = float('inf')
            best_moves = []
            for move in valid_moves:
                board_copy = board.create_copy()
                piece = board_copy.get_piece_at(move['piece_at'])
                board_copy.move_piece(piece, die_roll)
                result = self.minmax(board_copy, colour, dice_rolls_left, depth + 1, True, alpha, beta, start_time, time_limit)
                if result is not None:
                    if result['best_value'] < best_value:
                        best_value = result['best_value']
                        best_moves = [move] + result['best_moves']
                    beta = min(beta, best_value)
                    if beta <= alpha:
                        break
            return {'best_value': best_value, 'best_moves': best_moves}
    def get_valid_moves(self, board, colour, die_roll):
        valid_moves = []
        pieces_to_try = [x.location for x in board.get_pieces(colour)]
        pieces_to_try = list(set(pieces_to_try))

        for piece_location in pieces_to_try:
            piece = board.get_piece_at(piece_location)
            if board.is_move_possible(piece, die_roll):
                valid_moves.append({'piece_at': piece_location, 'die_roll': die_roll})
        return valid_moves            
            
    def minimax(self, board, colour, dice_rolls, depth, maximizing, start_time, time_limit):
        if time.time() - start_time > time_limit:
            return {'best_value': self.evaluate_board(board, colour), 'best_moves': []}

        # Base case: Depth or game end
        if depth == 0 or board.has_game_ended():
            eval_value = self.evaluate_board(board, colour)
            log(f"Evaluating board: value={eval_value}, depth={depth}")
            return {'best_value': eval_value, 'best_moves': []}

        best_value = float('-inf') if maximizing else float('inf')
        best_moves = []

        dice_rolls_left = dice_rolls.copy()
        die_roll = dice_rolls_left.pop(0)

        pieces_to_try = [x.location for x in board.get_pieces(colour)]
        pieces_to_try = list(set(pieces_to_try))

        valid_pieces = []
        for piece_location in pieces_to_try:
            piece = board.get_piece_at(piece_location)
            if piece and board.is_move_possible(piece, die_roll):
                valid_pieces.append(piece)

        if not valid_pieces:
            log(f"No valid pieces found for die roll {die_roll}")
            return {'best_value': best_value, 'best_moves': best_moves}

        for piece in valid_pieces:
            # Debugging valid moves
            log(f"Trying piece at {piece.location} with die roll {die_roll}")

            board_copy = board.create_copy()
            new_piece = board_copy.get_piece_at(piece.location)
            board_copy.move_piece(new_piece, die_roll)

            result = self.minimax(board_copy, colour, dice_rolls_left, depth - 1, not maximizing, start_time, time_limit)
            if result['best_value'] is None:
                continue

            if maximizing and result['best_value'] > best_value:
                best_value = result['best_value']
                best_moves = [{'die_roll': die_roll, 'piece_at': piece.location}] + result['best_moves']
            elif not maximizing and result['best_value'] < best_value:
                best_value = result['best_value']
                best_moves = [{'die_roll': die_roll, 'piece_at': piece.location}] + result['best_moves']

        log(f"Returning from minimax: best_value={best_value}, best_moves={best_moves}")
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
   
    def move_recursively(self, board, colour, dice_rolls):
        best_board_value = float('inf')
        best_pieces_to_move = []

        pieces_to_try = [x.location for x in board.get_pieces(colour)]
        pieces_to_try = list(set(pieces_to_try))

        valid_pieces = []
        for piece_location in pieces_to_try:
            valid_pieces.append(board.get_piece_at(piece_location))
        valid_pieces.sort(key=Piece.spaces_to_home, reverse=True)

        dice_rolls_left = dice_rolls.copy()
        die_roll = dice_rolls_left.pop(0)

        for piece in valid_pieces:
            if board.is_move_possible(piece, die_roll):
                board_copy = board.create_copy()
                new_piece = board_copy.get_piece_at(piece.location)
                board_copy.move_piece(new_piece, die_roll)
                if len(dice_rolls_left) > 0:
                    result = self.move_recursively(board_copy, colour, dice_rolls_left)
                    if len(result['best_moves']) == 0:
                        # we have done the best we can do
                        board_value = self.evaluate_board(board_copy, colour)
                        if board_value < best_board_value and len(best_pieces_to_move) < 2:
                            best_board_value = board_value
                            best_pieces_to_move = [{'die_roll': die_roll, 'piece_at': piece.location}]
                    elif result['best_value'] < best_board_value:
                        new_best_moves_length = len(result['best_moves']) + 1
                        if new_best_moves_length >= len(best_pieces_to_move):
                            best_board_value = result['best_value']
                            move = {'die_roll': die_roll, 'piece_at': piece.location}
                            best_pieces_to_move = [move] + result['best_moves']
                else:
                    board_value = self.evaluate_board(board_copy, colour)
                    if board_value < best_board_value and len(best_pieces_to_move) < 2:
                        best_board_value = board_value
                        best_pieces_to_move = [{'die_roll': die_roll, 'piece_at': piece.location}]

        return {'best_value': best_board_value,
                'best_moves': best_pieces_to_move}
        
    def evaluate_board(self, myboard, colour):
        board_stats = self.assess_board(colour, myboard)

        board_value = board_stats['sum_distances'] - float(board_stats['sum_distances_opponent']) / 3 + \
                      float(board_stats['sum_single_distance_away_from_home']) / 6 - \
                      board_stats['number_occupied_spaces'] - board_stats['opponents_taken_pieces'] + \
                      3 * board_stats['pieces_on_board'] + float(board_stats['sum_distances_to_endzone']) / 6\
                      + board_stats['threat level']/ 6
        return board_value




distance_odds_fraction_map = {
    1: Fraction(11, 36),
    2: Fraction(1, 3),
    3: Fraction(7, 18),
    4: Fraction(5, 12),
    5: Fraction(5, 12),
    6: Fraction(17, 36),
    7: Fraction(1, 6),
    8: Fraction(1, 6),
    9: Fraction(5, 36),
    10: Fraction(1, 12),
    11: Fraction(1, 18),
    12: Fraction(1, 12),
    15: Fraction(1, 36),
    16: Fraction(1, 36),
    18: Fraction(1, 36),
    20: Fraction(1, 36),
    24: Fraction(1, 36)
}
