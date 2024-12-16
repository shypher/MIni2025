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
        time_limit = board.getTheTimeLim() - 0.2
        with open("tree.txt", "a") as tree_log:
            best_move = self.minmax(board, colour, dice_roll, depth=4, maximizing_player=True,\
                alpha=float('-inf'), beta=float('inf'), start_time=start_time, time_limit=time_limit, tree_log=tree_log)
            tree_log.write(f"\n-----------------------------------------------------\n")
        if len(best_move['best_moves']) >0:
            for move in best_move['best_moves']:
                log(f"Moving piece at {move['piece_at']} to {move['piece_at'] + move['die_roll']}")
                make_move(move['piece_at'], move['die_roll'])
        else:
            log("No valid moves found.")   
        log(f"Best move: {best_move}")

    def minmax(self, board, colour, dice_rolls, depth, maximizing_player, alpha, beta, start_time, time_limit, tree_log=None):
        if  time.time() - start_time > time_limit:
            if tree_log:
                    tree_log.write(f"{'|   ' * depth}Depth {depth}: {'Max' if maximizing_player else 'Min'} Player\n")
            return {'best_moves': [], 'value': self.evaluate_board(board, colour)}
        if depth == 0:
            if tree_log:
                tree_log.write(f"{'|   ' * depth}Depth {depth}: {'Max' if maximizing_player else 'Min'} Player\n")
            return {'best_moves': [], 'value': self.evaluate_board(board, colour)}
        current_colour = colour if maximizing_player else colour.other()
        best_moves = []
        value = float('-inf') if maximizing_player else float('inf')
        for dice1, dice2 in self.generate_all_dice_combinations():
            new_dice_rolls = [dice1, dice2] if dice1 != dice2 else [dice1, dice1, dice1, dice1]
            dice_probability = 1/36 if dice1 != dice2 else 1/18
            possible_moves = self.get_all_possible_moves2(board, current_colour, dice1)+self.get_all_possible_moves2(board, current_colour, dice2)
            for move in possible_moves:
                new_board = board.create_copy()
                piece = new_board.get_piece_at(move['piece_at'])
                new_board.move_piece(piece, move['die_roll'])
                opponent_result = self.minmax(new_board, colour.other(), new_dice_rolls, depth - 1, not maximizing_player, alpha, beta, start_time, time_limit, tree_log)
                if maximizing_player:
                    if opponent_result['value'] > value:
                        value = opponent_result['value']
                        best_moves = [move]
                    elif opponent_result['value'] == value:
                        best_moves.append(move)
                    alpha = max(alpha, value)
                else:
                    if opponent_result['value'] < value:
                        value = opponent_result['value']
                        best_moves = [move]
                    elif opponent_result['value'] == value:
                        best_moves.append(move)
                    beta = min(beta, value)
                if beta <= alpha:
                    break
        return {'best_moves': best_moves, 'value': value}
                                    
    def minmax7200(self, board, colour, dice_rolls, depth, maximizing_player, alpha, beta, start_time, time_limit, tree_log=None):
        if time.time() - start_time > time_limit:
            return {'best_moves': [], 'value': self.evaluate_board(board, colour)}
        
        if depth == 0:
            if tree_log:
                tree_log.write(f"{'|   ' * depth}Depth {depth}: {'Max' if maximizing_player else 'Min'} Player\n")
            return {'best_moves': [], 'value': self.evaluate_board(board, colour)} 
        
        current_colour = colour if maximizing_player else colour.other()
        
        if maximizing_player:
            max_value = float('-inf')
            best_moves = []
            
            # Try all combinations of current player's dice rolls
            for dice_combination in self.generate_dice_combinations(dice_rolls):
                remaining_dice = [die for die in dice_rolls if die not in dice_combination]
                
                # Get all possible moves for the current dice combination
                possible_moves = []
                for dice in dice_combination:
                    moves = self.get_all_possible_moves2(board, current_colour, dice)
                    possible_moves.extend(moves)
                
                for move in possible_moves:
                    new_board = board.create_copy()
                    piece = new_board.get_piece_at(move['piece_at'])
                    new_board.move_piece(piece, move['die_roll'])
                    
                    # If no more dice, consider all possible opponent dice rolls
                    if not remaining_dice:
                        worst_value = float('inf')
                        # Generate all possible dice roll combinations for the opponent
                        for opponent_dice_rolls in self.generate_all_dice_combinations():
                            value = self.minmax(new_board, colour.other(), opponent_dice_rolls, depth - 1, False, alpha, beta, start_time, time_limit, tree_log)
                            worst_value = min(worst_value, value['value'])
                        
                        # Use the worst (minimum) value for the maximizing player
                        current_value = worst_value
                    else:
                        # Continue with remaining dice
                        value = self.minmax(new_board, colour, remaining_dice, depth, False, alpha, beta, start_time, time_limit, tree_log)
                        current_value = value['value']
                    
                    # Update best moves
                    if current_value > max_value:
                        max_value = current_value
                        best_moves = [move]
                    elif current_value == max_value:
                        best_moves.append(move)
                    
                    # Alpha-beta pruning
                    alpha = max(alpha, current_value)
                    if beta <= alpha:
                        break
            
            return {'best_moves': best_moves, 'value': max_value}
        
        else:  # Minimizing player
            min_value = float('inf')
            best_moves = []
            
            # Try all combinations of current player's dice rolls
            for dice_combination in self.generate_dice_combinations(dice_rolls):
                remaining_dice = [die for die in dice_rolls if die not in dice_combination]
                
                # Get all possible moves for the current dice combination
                possible_moves = []
                for dice in dice_combination:
                    moves = self.get_all_possible_moves2(board, current_colour, dice)
                    possible_moves.extend(moves)
                
                for move in possible_moves:
                    new_board = board.create_copy()
                    piece = new_board.get_piece_at(move['piece_at'])
                    new_board.move_piece(piece, move['die_roll'])
                    
                    # If no more dice, consider all possible opponent dice rolls
                    if not remaining_dice:
                        best_value = float('-inf')
                        # Generate all possible dice roll combinations for the opponent
                        for opponent_dice_rolls in self.generate_all_dice_combinations():
                            value = self.minmax(new_board, colour.other(), opponent_dice_rolls, depth - 1, True, alpha, beta, start_time, time_limit, tree_log)
                            best_value = max(best_value, value['value'])
                        
                        # Use the best (maximum) value for the minimizing player
                        current_value = best_value
                    else:
                        # Continue with remaining dice
                        value = self.minmax(new_board, colour, remaining_dice, depth, True, alpha, beta, start_time, time_limit, tree_log)
                        current_value = value['value']
                    
                    # Update best moves
                    if current_value < min_value:
                        min_value = current_value
                        best_moves = [move]
                    elif current_value == min_value:
                        best_moves.append(move)
                    
                    # Alpha-beta pruning
                    beta = min(beta, current_value)
                    if beta <= alpha:
                        break
            
            return {'best_moves': best_moves, 'value': min_value}

    def generate_all_dice_combinations(self):
        dice_combinations = []
        # Generate all dice combinations, ensuring all possibilities are included
        for first in range(1, 7):
            for second in range(first, 7):  # Include dice combinations in a sorted way
                dice_combinations.append([first, second])
        return dice_combinations

    def generate_dice_combinations(self, dice_rolls):
        """Generate all possible ways to use the given dice rolls."""
        combinations = []
        if len(dice_rolls) == 1:
            combinations = [[dice] for dice in dice_rolls]
        else:
            # For two dice, generate all permutations
            combinations = [[dice_rolls[0], dice_rolls[1]], [dice_rolls[1], dice_rolls[0]]]
        return combinations
        
    def minmax2(self, board, colour, dice_rolls, depth, maximizing_player, alpha, beta, start_time, time_limit, tree_log=None):
        if time.time() - start_time > time_limit:
            return {'best_moves': [], 'value': self.evaluate_board(board, colour)}
        if depth == 0:
            return {'best_moves': [], 'value': self.evaluate_board(board, colour)}
        if tree_log:
            tree_log.write(f"{'|   ' * depth}Depth {depth}: {'Max' if maximizing_player else 'Min'} Player\n")
        
        if maximizing_player:
            max_value = float('-inf')
            best_moves = []
            for dice in dice_rolls:
                remaining_dice_rolls = dice_rolls[:]
                remaining_dice_rolls.remove(dice)
                for move in self.get_all_possible_moves2(board, colour, dice):
                    new_board = board.create_copy()
                    piece = new_board.get_piece_at(move['piece_at'])
                    new_board.move_piece(piece, move['die_roll'])
                    if(len(remaining_dice_rolls) != 0):
                        value = self.minmax(new_board, colour, remaining_dice_rolls, depth, False, alpha, beta, start_time, time_limit, tree_log)
                    else:
                        for diceA in range(1,7):
                            for diceB in range(diceA, 7):
                                if (diceA!=diceB):
                                    new_dice_rolls = [diceA, diceB]
                                else:
                                    new_dice_rolls = [diceA, diceA, diceA, diceA]
                                value = self.minmax(new_board, colour.other(), new_dice_rolls, depth - 1, False, alpha, beta, start_time, time_limit, tree_log)
                    if value['value'] > max_value:
                        max_value = value['value']
                        best_moves = [move]
                    elif value['value'] == max_value:
                        best_moves.append(move)
                    alpha = max(alpha, value['value'])
                    if beta <= alpha:
                        break
            return {'best_moves': best_moves, 'value': max_value}
        else:
            min_value = float('inf')
            best_moves = []
            for dice in dice_rolls:
                remaining_dice_rolls = dice_rolls[:]
                remaining_dice_rolls.remove(dice)
                for move in self.get_all_possible_moves2(board, colour.other(), dice):
                    new_board = board.create_copy()
                    piece = new_board.get_piece_at(move['piece_at'])
                    new_board.move_piece(piece, move['die_roll'])
                    if(len(remaining_dice_rolls) != 0):
                        value = self.minmax(new_board, colour, remaining_dice_rolls, depth, True, alpha, beta, start_time, time_limit, tree_log)
                    else:
                        for diceA in range(1,7):
                            for diceB in range(diceA, 7):
                                if (diceA!=diceB):
                                    new_dice_rolls = [diceA, diceB]
                                else:
                                    new_dice_rolls = [diceA, diceA, diceA, diceA]
                                value = self.minmax(new_board, colour.other(), new_dice_rolls, depth - 1, True, alpha, beta, start_time, time_limit, tree_log)
                    if value['value'] < min_value:
                        min_value = value['value']
                        best_moves = [move]
                    elif value['value'] == min_value:
                        best_moves.append(move)
                    beta = min(beta, value['value'])
                    if beta <= alpha:
                        break
            return {'best_moves': best_moves, 'value': min_value}
            
    def backgummon_minimax(self, board, colour, dice_rolls, depth, maximizing_player, alpha, beta, start_time, time_limit, tree_log=None):
        if time.time() - start_time > time_limit:
            return {'best_moves': [], 'value': self.evaluate_board(board, colour)}
        if depth == 0:
            return {'best_moves': [], 'value': self.evaluate_board(board, colour)}
        if tree_log:
            tree_log.write(f"{'|   ' * depth}Depth {depth}: {'Max' if maximizing_player else 'Min'} Player\n")
        
        if maximizing_player:
            
            max_value = float('-inf')
            best_moves = []
            all_possible_moves = self.get_all_possible_moves(board, colour, dice_rolls)
            for move in all_possible_moves:
                new_board = board.create_copy()
                piece= new_board.get_piece_at(move['piece_at'])
                new_board.move_piece(piece, move['die_roll'])
                value = self.backgummon_minimax(new_board, colour, dice_rolls, depth - 1, False, alpha, beta, start_time, time_limit, tree_log)['value']
                if value > max_value:
                    max_value = value
                    best_moves = [move]
                elif value == max_value:
                    best_moves.append(move)
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return {'best_moves': best_moves, 'value': max_value}
        else:
            min_value = float('inf')
            best_moves = []
            for diceA in range(1,6):
                for diceB in range(diceA, 6):
                    if (diceA!=diceB):
                        dice_rolls = [diceA, diceB]
                    else:
                        dice_rolls = [diceA, diceA, diceA, diceA]
                    for move in self.get_all_possible_moves(board, colour.other(), dice_rolls):
                        new_board = board.create_copy()
                        piece = new_board.get_piece_at(move['piece_at'])
                        new_board.move_piece(piece, move['die_roll'])
                        #log(f"Moving piece at {piece.location} to {piece.location + move['die_roll']}")
                        value = self.backgummon_minimax(new_board, colour, dice_rolls, depth - 1, True, alpha, beta, start_time, time_limit, tree_log)['value']
                        if value < min_value:
                            min_value = value
                            best_moves = [move]
                        elif value == min_value:
                            best_moves.append(move)
                        beta = min(beta, value)
                        if beta <= alpha:
                            break
            return {'best_moves': best_moves, 'value': min_value}
    def get_all_possible_moves2(self, board, colour, dice):
        all_possible_moves = []
        pieces_to_try = [x.location for x in board.get_pieces(colour)]
        pieces_to_try = list(set(pieces_to_try))
        
        for piece_location in pieces_to_try:
            piece = board.get_piece_at(piece_location)
            if board.is_move_possible(piece, dice):
                all_possible_moves.append({'piece_at': piece_location, 'die_roll': dice})
        
        return all_possible_moves
    
    def get_all_possible_moves3(self, board, colour, dice):
        all_possible_moves = []
        pieces_to_try = [x.location for x in board.get_pieces(colour)]
        pieces_to_try = list(set(pieces_to_try))
        for piece_location in pieces_to_try:
            piece = board.get_piece_at(piece_location)
            if board.is_move_possible(piece, dice):
                all_possible_moves.append({'piece_at': piece_location, 'die_roll': dice})
        print(all_possible_moves)
        return all_possible_moves

    def get_all_possible_moves(self, board, colour, dice_rolls):
        all_possible_moves = []
        pieces_to_try = [x.location for x in board.get_pieces(colour)]
        pieces_to_try = list(set(pieces_to_try))
        for piece_location in pieces_to_try:
            piece = board.get_piece_at(piece_location)
            for dice in dice_rolls:
                if board.is_move_possible(piece, dice):
                    all_possible_moves.append({'piece_at': piece_location, 'die_roll': dice})
            

       
                    
        
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
