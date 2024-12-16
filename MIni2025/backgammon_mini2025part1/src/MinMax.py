from src.strategies import Strategy
from src.piece import Piece
from fractions import Fraction
import time
import threading
from itertools import permutations
def log(message, file_path="tournament_log.txt"):
    with open(file_path, "a") as log_file:
        log_file.write(message + "\n")
    print(message)
class MinMax_shayOren(Strategy):
    
    @staticmethod
    def get_difficulty():
        return "shimy"
    def __init__(self):
        tree = None
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

        self.time_limit = board.getTheTimeLim() - 0.2
        possible_state =  self.get_all_possible_moves(board, colour, dice_roll , start_time, self.time_limit)
        best_val = float('-inf')
        best_moves = []
        for new_board, moves in possible_state.items():
            if time.time() - start_time > self.time_limit:
                break
            val = self.minmax(new_board, colour, depth=4, maximizing_player=True, alpha=float('-inf'), beta=float('inf'), start_time=start_time, time_limit=self.time_limit)
            if val > best_val:
                best_val = val
                best_moves = moves
        if len(best_moves) != 0:
            for move in best_moves:
                log(f"Moving piece at {move['piece_at']} to {move['piece_at'] + move['die_roll']}")
                make_move(move['piece_at'], move['die_roll'])
        return
    def minmax(self, board, colour, depth, maximizing_player, alpha, beta, start_time, time_limit):
        if time.time() - start_time > time_limit:
            return float('-inf') if maximizing_player else float('inf') #meybe self.evaluate_board(board, colour)
        if depth == 0:
            return self.evaluate_board(board, colour)
        dice_rolls = []
        for diceA in range(1,7):
            for diceB in range(diceA, 7):
                if (diceA!=diceB):
                    dice_rolls_prop = [diceA, diceB]
                else:
                    dice_rolls = [diceA, diceA, diceA, diceA]
                posible_states = self.get_all_possible_moves(board, colour, dice_rolls, start_time, time_limit)
        
        if maximizing_player:
            best_val = float('-inf')
            for new_board in posible_states.keys():
                val = self.minmax(new_board, colour, depth-1, False, alpha, beta, start_time, time_limit)
                if dice_rolls[0] != dice_rolls[1]:
                    prop = 1/18
                else:
                    prop = 1/36

                best_val = max(best_val, val)
                alpha = max(alpha, val)
                if beta <= alpha:
                    break
        else:
            best_val = float('inf')
            for new_board in posible_states.keys():
                val = self.minmax(new_board, colour, depth-1, True, alpha, beta, start_time, time_limit)
                if dice_rolls[0] != dice_rolls[1]:
                    prop = 1/18
                else:
                    prop = 1/36
                val = val * prop
                best_val = min(best_val, val)
                beta = min(beta, val)
                if beta <= alpha:
                    break
        return best_val
    
            



        


    
    def create_tree(self, tree, borad):
        for child in range(1,16):
            node = tree.add_children(tree, child)
            node.value = self

    def get_all_possible_moves(self, board, colour, dice_roll, start_time, time_limit):
        if time.time() - start_time > time_limit:
            return {}
        if(len(dice_roll) == 0):
            return []
        if len(dice_roll) == 0:
            return {}
        all_possible_moves = {}
        pieces_to_try = [x.location for x in board.get_pieces(colour)]
        pieces_to_try = list(set(pieces_to_try))
        """
        if (dice_roll[0] != dice_roll[1]):
        #get 2 order, 1 the first dice is the first and 2 the second dice is the first
            dice_order = [dice_roll, [dice_roll[1], dice_roll[0]]]
        else:
            dice_order = dice_roll
        """
        for dice in (set(permutations(dice_roll))):
            dice_roll = dice[0]
            remeining_dice = dice[1:]
            for piece_curr in pieces_to_try:
                piece = board.get_piece_at(piece_curr)
                if board.is_move_possible(piece, dice_roll):
                    board_copy = board.create_copy()
                    new_piece = board_copy.get_piece_at(piece.location)
                    board_copy.move_piece(new_piece, dice_roll)
                    rest_dice_board = self.get_all_possible_moves(board_copy, colour, remeining_dice, start_time, time_limit)
                    if len(rest_dice_board) == 0:
                        all_possible_moves[board_copy] = [{'piece_at': piece.location, 'die_roll': dice_roll}]
                    else:
                        for new_board, moves in rest_dice_board.items():
                            all_possible_moves[new_board] = [{'piece_at': piece.location, 'die_roll': dice_roll}] + moves
        return all_possible_moves


                        

                    

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


class TreeNode:
    def __init__(self, value):
        self.value = value
        self.children = [None]*15

    def add_child(self, child_node):
        self.children.append(child_node)

    def get_min(self):
        min_value = float('inf')
        for child in self.children:
            child_min = child.get_value()
            min_value = min(min_value, child_min)
        return min_value
        
    
    def get_max(self):
        max_value = float('inf')
        for child in self.children:
            child_max = child.value
            max_value = max(max_value, child_max)
        return max_value
    def get_value(self):
        return self.value
    def dice_to_index(self, diceA, diceB):
        return (diceA-1)+(diceB-1)*6

        

class Tree:
    def __init__(self, root_value):
        self.root = TreeNode(root_value)
        

    def add_children(self, parent_node, children_values):
        for value in children_values:
            child_node = TreeNode(value)
            parent_node.add_child(child_node)

    def get_min(self):
        if self.root:
            return self.root.get_min_max()
        else:
            return None, None
    def cut_tree_by_child(self, child_index):
        self.root = self.root.children[child_index]

        
        

