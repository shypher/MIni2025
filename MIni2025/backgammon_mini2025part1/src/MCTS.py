from src.strategies import Strategy
from src.piece import Piece
from fractions import Fraction
import time
import threading
from itertools import permutations
import math
import random
from random import randint, shuffle
from src.compare_all_moves_strategy import CompareAllMovesSimple

 
def log(message, file_path="tournament_log.txt"):
    with open(file_path, "a") as log_file:
        log_file.write(message + "\n")
    print(message)
def log_tree(message, file_path="tree.txt"):
    with open(file_path, "a") as log_file:
        log_file.write(message + "\n")
    
class MCTS_shayOren(Strategy):
    @staticmethod
    def get_difficulty():
        return "shimeh"
    def __init__(self):
        self.tree = None

    def move(self, board, colour, dice_roll, make_move, opponents_activity):
        log("Starting MCTS move calculation")
        start_time = time.time()
        time_limit = board.getTheTimeLim() - 0.75 if board.getTheTimeLim() != -1 else float('inf')
        self.tree = MCTSNode(state=board, colour=colour, dice_roll=dice_roll)

        root = self.tree
        iterations = 0

        while time.time() - start_time < time_limit:
            iterations += 1
            node = root

            # Selection
            while not node.untried_moves and node.children:
                node = node.best_child(exploration_weight = 1.0 / (1 + iterations / 100)
)
                #log(f"Selecting child node: Visits={node.visits}, Total Reward={node.total_reward}")

            # Expansion
            if node.untried_moves:
                #print(node.untried_moves)
                board = list(node.untried_moves.keys())[0]
                moves = node.untried_moves[board]
                node.untried_moves.pop(board)
                #log(f"Expanding node with move: {move}")
                child_node = node.add_child(moves, node.state)
                node = child_node

            # Simulation
            reward = self.simulate_game2(node.state.create_copy(), colour)
            #log(f"Simulation completed with reward={reward}")

            # Backpropagation
            while node:
                node.update(reward)
                #log(f"Backpropagating reward={reward} for node with move={node.move}")
                node = node.parent
            #log(f"Iteration: {iterations}")
        if root.best_child(exploration_weight=0) is not None:
            best_move = root.best_child(exploration_weight=0).move
        #log(f"Selected best move: {best_move}")
            log(f"Selected best move: {best_move}")
            for move in best_move:
                if colour !=  0:
                    log(f"Moving piece at {move['piece_at']} to {move['piece_at'] + move['die_roll']}")
                make_move(move['piece_at'], move['die_roll'])
        self.tree.print_tree()
        log_tree(f"_____________________________________________")
                
    def simulate_game(self, board, colour):
        game_color = colour
        while not board.has_game_ended():
            dice_roll = [randint(1, 6), randint(1, 6)]
            if dice_roll[0] == dice_roll[1]:
                dice_roll = [dice_roll[0]] * 4
            for die_roll in dice_roll:
                valid_pieces = board.get_pieces(game_color)
                shuffle(valid_pieces)
                for piece in valid_pieces:
                    if board.is_move_possible(piece, die_roll):
                        #print(f"Moving piece {piece.location} {die_roll}")
                        #board.print_board()
                        board.move_piece(piece, die_roll)
                        break
            game_color = game_color.other()
                
        if board.who_won() == colour:
            return 1
        else:
            return -1

    def simulate_game_Heuristic(self, board, starting_colour):
        heuristic_player = CompareAllMovesSimple()
        current_colour = starting_colour

        # Variables to track dice rolls and moves
        moves = []
        previous_dice_roll = []

        while not board.has_game_ended():
            # Roll the dice
            dice_roll = [randint(1, 6), randint(1, 6)]
            if dice_roll[0] == dice_roll[1]:
                dice_roll *= 2  # Handle doubles

            # Copy the dice roll for manipulation
            dice_roll_copy = dice_roll.copy()

            # Define the make_move function for simulation
            def make_move(location, die_roll):
                self.game.make_move_simulation(
                    board,
                    location,
                    die_roll,
                    dice_roll_copy,
                    moves,
                    previous_dice_roll
                )

                # Use the heuristic player to make moves
                heuristic_player.move(
                    board,
                    current_colour,
                    dice_roll_copy,
                    make_move,
                    {'dice_roll': previous_dice_roll.copy(), 'opponents_move': moves.copy()}
                )

                # Reset move tracking for the next player
                moves.clear()
                previous_dice_roll.clear()

                # Switch player
                current_colour = current_colour.other()

                # Determine the reward based on the game outcome
                if board.who_won() == starting_colour:
                    return 1  # Win
                else:
                    return -1  # Lose

    def simulate_game2(self, board, starting_colour):
        cur_colour = starting_colour
        while not board.has_game_ended():
            dice_roll = [randint(1, 6), randint(1, 6)]
            if dice_roll[0] == dice_roll[1]:
                dice_roll = [dice_roll[0]] * 4
            result = self.move_recursively(board, cur_colour, dice_roll)
            not_a_double = len(dice_roll) == 2
            if not_a_double:
                new_dice_roll = dice_roll.copy()
                new_dice_roll.reverse()
                result_swapped = self.move_recursively(board, cur_colour,
                                                    dice_rolls=new_dice_roll)
                if result_swapped['best_value'] < result['best_value'] and \
                        len(result_swapped['best_moves']) >= len(result['best_moves']):
                    result = result_swapped
            if len(result['best_moves']) != 0:
                for move in result['best_moves']:
                    piece = board.get_piece_at(move['piece_at'])
                    board.move_piece(piece, move['die_roll'])
            cur_colour = cur_colour.other()
        if board.who_won() == starting_colour:
            return 1
        else:
            return -1

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

    def evaluate_board(self, board, colour):
        board_stats = self.assess_board(colour, board)
        board_value = board_stats['sum_distances'] - float(board_stats['sum_distances_opponent']) / 3 + \
                      float(board_stats['sum_single_distance_away_from_home']) / 6 - \
                      board_stats['number_occupied_spaces'] - board_stats['opponents_taken_pieces'] + \
                      3 * board_stats['pieces_on_board']

        return board_value
    def assess_board(self, colour, myboard):
        pieces = myboard.get_pieces(colour)
        pieces_on_board = len(pieces)
        sum_distances = 0
        number_of_singles = 0
        number_occupied_spaces = 0
        sum_single_distance_away_from_home = 0
        sum_distances_to_endzone = 0
        for piece in pieces:
            sum_distances = sum_distances + piece.spaces_to_home()
            if piece.spaces_to_home() > 6:
                sum_distances_to_endzone += piece.spaces_to_home() - 6
        for location in range(1, 25):
            pieces = myboard.pieces_at(location)
            if len(pieces) != 0 and pieces[0].colour == colour:
                if len(pieces) == 1:
                    number_of_singles = number_of_singles + 1
                    sum_single_distance_away_from_home += 25 - pieces[0].spaces_to_home()
                elif len(pieces) > 1:
                    number_occupied_spaces = number_occupied_spaces + 1
        opponents_taken_pieces = len(myboard.get_taken_pieces(colour.other()))
        opponent_pieces = myboard.get_pieces(colour.other())
        sum_distances_opponent = 0
        for piece in opponent_pieces:
            sum_distances_opponent = sum_distances_opponent + piece.spaces_to_home()
        return {
            'number_occupied_spaces': number_occupied_spaces,
            'opponents_taken_pieces': opponents_taken_pieces,
            'sum_distances': sum_distances,
            'sum_distances_opponent': sum_distances_opponent,
            'number_of_singles': number_of_singles,
            'sum_single_distance_away_from_home': sum_single_distance_away_from_home,
            'pieces_on_board': pieces_on_board,
            'sum_distances_to_endzone': sum_distances_to_endzone,
        }

class MCTSNode:
    def __init__(self, state, colour, dice_roll, parent=None, move=None):
        self.state = state
        self.colour = colour
        self.dice_roll = dice_roll
        self.parent = parent
        self.move = move  # The move that led to this state
        self.children = []
        self.visits = 0
        self.total_reward = 0
        self.untried_moves = self.get_all_possible_moves(state, colour, dice_roll)


    def get_all_possible_moves(self, board, colour, dice_roll):
        if(len(dice_roll) == 0):
            return []
        if(len(dice_roll) == 0):
            return {}
        all_possible_moves = {}
        pieces_to_try = [x.location for x in board.get_pieces(colour)]
        pieces_to_try = list(set(pieces_to_try))
        for dice in (set(permutations(dice_roll))):
            dice_roll = dice[0]
            remeining_dice = dice[1:]
            for piece_curr in pieces_to_try:
                piece = board.get_piece_at(piece_curr)
                if board.is_move_possible(piece, dice_roll):
                    board_copy = board.create_copy()
                    new_piece = board_copy.get_piece_at(piece.location)
                    board_copy.move_piece(new_piece, dice_roll)
                    rest_dice_board = self.get_all_possible_moves(board_copy, colour, remeining_dice)
                    if len(rest_dice_board) == 0:
                        all_possible_moves[board_copy] = [{'piece_at': piece.location, 'die_roll': dice_roll}]
                    else:
                        for new_board, moves in rest_dice_board.items():
                            all_possible_moves[new_board] = [{'piece_at': piece.location, 'die_roll': dice_roll}] + moves
        return all_possible_moves



    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, exploration_weight=1.0):
        try:
            best = max(
                self.children,
                key=lambda child: child.total_reward / (child.visits + 1e-6) +
                                  exploration_weight * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6))
            )
        except ValueError:
            best = None
        #log(f"Best child selected with move={best.move}, Visits={best.visits}, Total Reward={best.total_reward}")
        return best

    def add_child(self, move, new_state):
        #log(f"Adding child node for move={move}")
        
        child = MCTSNode(new_state, self.colour.other(), self.dice_roll, parent=self, move=move)
        self.children.append(child)
        return child

    def update(self, reward):
        self.visits += 1
        self.total_reward += reward
        #log(f"Node updated: Visits={self.visits}, Total Reward={self.total_reward}")
    def print_tree(self, depth=0):
        log_tree(f"  " * depth + str(self.move)+ f"Total Reward={self.total_reward}")
        for child in self.children:
            child.print_tree(depth + 1)