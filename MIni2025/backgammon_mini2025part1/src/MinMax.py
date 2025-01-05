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
        minmax_deap =0
        endgame = False
        
    def assess_board0(self, colour, myboard):
        pieces = myboard.get_pieces(colour)
        opponent_pieces = myboard.get_pieces(colour.other())
        pieces_on_board = len(pieces)
        sum_distances = sum(piece.spaces_to_home() for piece in pieces)
        sum_single_distance_away_from_home = 0
        number_of_singles=0
        number_occupied_spaces=0
        stack_penalty=0
        end_game = False

        for location in range(1, 25):
            pieces = myboard.pieces_at(location)
            if len(pieces) != 0 and pieces[0].colour == colour:
                if len(pieces) == 1:
                    number_of_singles = number_of_singles + 1
                    sum_single_distance_away_from_home += 25 - pieces[0].spaces_to_home()
                elif len(pieces) > 5:
                    stack_penalty+= 1
                elif len(pieces) > 1:
                    number_occupied_spaces = number_occupied_spaces + 1
        """number_of_singles = sum(1 for loc in range(1, 25) if len(myboard.pieces_at(loc)) == 1)
        number_occupied_spaces =sum(1 for loc in range(1, 25) if len(myboard.pieces_at(loc)) > 1)
        sum_single_distance_away_from_home = sum(25 - pieces[0].spaces_to_home() for location in range(1, 25) if len((pieces := myboard.pieces_at(location)))
            == 1 and pieces[0].colour == colour)"""
        
        sum_distances_to_endzone = sum(max(0, piece.spaces_to_home() - 6) for piece in pieces)
        opponents_taken_pieces = len(myboard.get_taken_pieces(colour.other()))
        threat_level = sum(1 for piece in pieces for opponent_piece in opponent_pieces if abs(piece.location - opponent_piece.location) < 8)
        """for piece in pieces:
            for opponent_piece in opponent_pieces:
                distance = abs(piece.location- opponent_piece.location)
                if distance < 8:   
                    threat_level += 1 """
                #threat_level = chance_map(distance)*10 
        opponents_taken_pieces = len(myboard.get_taken_pieces(colour.other()))
        my_taken_pieces = len(myboard.get_taken_pieces(colour))
        opponent_pieces = myboard.get_pieces(colour.other())
        #piece in endzone
        endzone_pieces = sum(1 for piece in pieces if piece.spaces_to_home() < 7)
        sum_distances_opponent = sum(piece.spaces_to_home() for piece in opponent_pieces)
        pieces_on_other_endzone = sum(1 for piece in pieces if piece.spaces_to_home() < 19)
        
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
            'my_taken_pieces' : my_taken_pieces,
            'pieces_on_other_endzone' : pieces_on_other_endzone,
            'stack_penalty': stack_penalty,
            'endzone_pieces': endzone_pieces
        }
        
    def assess_board3(self, colour, myboard):
        pieces = myboard.get_pieces(colour)
        opponent_pieces = myboard.get_pieces(colour.other())
        pieces_on_board = len(pieces)
        sum_distances = sum(piece.spaces_to_home() for piece in pieces)
        sum_single_distance_away_from_home = 0
        number_of_singles = 0
        number_occupied_spaces = 0
        home_board_control = 0
        prime_value = 0
        connectivity = 0

        # Calculate number of singles and occupied spaces
        for location in range(1, 25):
            pieces_at_location = myboard.pieces_at(location)
            if len(pieces_at_location) != 0 and pieces_at_location[0].colour == colour:
                if len(pieces_at_location) == 1:
                    number_of_singles += 1
                    sum_single_distance_away_from_home += 25 - pieces_at_location[0].spaces_to_home()
                elif len(pieces_at_location) > 1:
                    number_occupied_spaces += 1

                # Update home board control (locations 1-6)
                if 1 <= location <= 6:
                    home_board_control += 1

        # Calculate prime value
        primes = 0
        current_prime_length = 0
        for location in range(1, 25):
            pieces_at_location = myboard.pieces_at(location)
            if len(pieces_at_location) > 0 and pieces_at_location[0].colour == colour:
                current_prime_length += 1
            else:
                if current_prime_length > 1:
                    primes += current_prime_length
                current_prime_length = 0
        prime_value = primes

        # Calculate connectivity
        piece_locations = [piece.location for piece in pieces]
        for location in piece_locations:
            if any(abs(location - other) <= 6 for other in piece_locations if location != other):
                connectivity += 1

        # Additional metrics
        sum_distances_to_endzone = sum(max(0, piece.spaces_to_home() - 6) for piece in pieces)
        opponents_taken_pieces = len(myboard.get_taken_pieces(colour.other()))
        threat_level = sum(1 for piece in pieces for opponent_piece in opponent_pieces if abs(piece.location - opponent_piece.location) < 8)
        my_taken_pieces = len(myboard.get_taken_pieces(colour))
        sum_distances_opponent = sum(piece.spaces_to_home() for piece in opponent_pieces)
        pieces_on_other_endzone = sum(1 for piece in pieces if piece.spaces_to_home() < 19) 
        return {
            'number_occupied_spaces': number_occupied_spaces,
            'opponents_taken_pieces': opponents_taken_pieces,
            'sum_distances': sum_distances,
            'sum_distances_opponent': sum_distances_opponent,
            'number_of_singles': number_of_singles,
            'sum_single_distance_away_from_home': sum_single_distance_away_from_home,
            'pieces_on_board': pieces_on_board,
            'sum_distances_to_endzone': sum_distances_to_endzone,
            'threat level': threat_level,
            'my_taken_pieces': my_taken_pieces,
            'pieces_on_other_endzone': pieces_on_other_endzone,
            'prime_value': prime_value,
            'home_board_control': home_board_control,
            'connectivity': connectivity,
        }
    def assess_board2(self, colour, myboard):
        pieces = myboard.get_pieces(colour)
        pieces_opponent = myboard.get_pieces(colour.other())
        taken_pieces = myboard.get_taken_pieces(colour)
        taken_pieces_opponent = myboard.get_taken_pieces(colour.other())
        pieces_out = 15-len(pieces)
        pieces_out_opponent = 15-len(pieces_opponent)
        sum_power_our = 0
        sum_power_opponent = 0
        threat_level = sum(1 for piece in pieces for opponent_piece in pieces_opponent if abs(piece.location - opponent_piece.location) < 8)
        for location in range(1, 25):
            pieces = myboard.pieces_at(location)
            if len(pieces) >= 1:
                power =25- pieces[0].spaces_to_home()

                if power >22:
                     power = power 
                elif power > 18:
                     power = power 
                elif power > 12:
                    power = power 
                elif power > 6:
                    power = power * 0.5
                elif power > 0:
                    power = power 
                if len(pieces) == 1:
                    power = power * 0.0001
                if len(pieces) > 3:
                    power = power * 0.8
                if len(pieces) > 5:
                    power = power * 0.5
                if pieces[0].colour == colour:
                    sum_power_our += power
                else:
                    sum_power_opponent += power
        return {
            'pieces_out': pieces_out,
            'pieces_out_opponent': pieces_out_opponent,
            'sum_power_our': sum_power_our,
            'sum_power_opponent': sum_power_opponent,
            'taken_pieces': len(taken_pieces),
            'taken_pieces_opponent': len(taken_pieces_opponent),
            'threat_level': threat_level,
        }
                
    def assess_board_0(self, colour, myboard):
        pieces = myboard.get_pieces(colour)
        pieces_on_board = len(pieces)
        sum_distances = 0
        number_of_singles = 0
        number_occupied_spaces = 0
        home_control = 0
        sum_single_distance_away_from_home = 0
        sum_distances_to_endzone = 0
        board_control =0
        sum_distance_far_from_home=0
        building_of_two = 0
        tower = 0
        the_most_far =0
        op_most_far = 0
        taken_pieces = len(myboard.get_taken_pieces(colour))
        end_game = False
        first_tower = 0
        if sum(1 for piece in myboard.get_pieces(colour) if piece.spaces_to_home() > 7) ==0:
            end_game = True
        for piece in pieces:
            the_most_far = max(the_most_far, piece.spaces_to_home())
            sum_distances = sum_distances + piece.spaces_to_home()
            if piece.spaces_to_home() > 6:
                sum_distances_to_endzone += piece.spaces_to_home() - 6
            if piece.spaces_to_home()>18:
                sum_distance_far_from_home += piece.spaces_to_home() -6
            if piece.spaces_to_home() ==1 and not end_game:
                first_tower += 1
                
        if first_tower>3:
            first_tower = 0 
         
        sum_distances_opponent = 0
        opponent_pieces = myboard.get_pieces(colour.other())
        for piece in opponent_pieces:
            op_most_far = max(op_most_far, piece.spaces_to_home())
            sum_distances_opponent = sum_distances_opponent + piece.spaces_to_home()
        for location in range(1, 25):
            pieces = myboard.pieces_at(location)
            if len(pieces) != 0 and pieces[0].colour == colour:
                if len(pieces) == 1:
                    if op_most_far + pieces[0].spaces_to_home()<25:
                        number_of_singles = number_of_singles + 0.1
                    else:
                        number_of_singles = number_of_singles + 1
                    sum_single_distance_away_from_home += 25 - pieces[0].spaces_to_home()
                elif len(pieces) > 1:
                    number_occupied_spaces = number_occupied_spaces + 1
                if len(pieces) > 1:
                    board_control+= 1
                    if 1 <= pieces[0].spaces_to_home() <= 6:
                        home_control += 1
                if len(pieces) == 2:
                    building_of_two += 1
                elif len(pieces) > 3:
                    tower += 1
        opponents_taken_pieces = len(myboard.get_taken_pieces(colour.other()))
        threat_level = sum(1 for piece in pieces for opponent_piece in opponent_pieces if abs(piece.location - opponent_piece.location) < 8)


        pieces_off_board = 15 - pieces_on_board
        if the_most_far+op_most_far < 25:
            number_of_singles=0
            number_occupied_spaces=0
            pieces_off_board = pieces_off_board *300
            home_control=0
            board_control=0
            building_of_two=0    
        return {
            'number_occupied_spaces': number_occupied_spaces,
            'opponents_taken_pieces': opponents_taken_pieces,
            'taken_pieces': taken_pieces,
            'sum_distances': sum_distances,
            'sum_distances_opponent': sum_distances_opponent,
            'number_of_singles': number_of_singles,
            'sum_single_distance_away_from_home': sum_single_distance_away_from_home,
            'pieces_off_board': pieces_off_board,
            'sum_distances_to_endzone': sum_distances_to_endzone,
            'home_control': home_control,
            'board_control': board_control,
            'threat_level': threat_level,
            'sum_distance_far_from_home': sum_distance_far_from_home,
            'building_of_two':building_of_two,
            'tower': tower,
            'first_tower': first_tower
        }
              
    def assess_board(self, colour, myboard):
        pieces = myboard.get_pieces(colour)
        pieces_on_board = len(pieces)
        sum_distances = 0
        number_of_singles = 0
        number_occupied_spaces = 0
        home_control = 0
        sum_single_distance_away_from_home = 0
        sum_distances_to_endzone = 0
        board_control =0
        sum_distance_far_from_home=0
        building_of_two = 0
        tower = 0
        taken_pieces = len(myboard.get_taken_pieces(colour))
        end_game = False
        first_tower = 0
        if sum(1 for piece in myboard.get_pieces(colour) if piece.spaces_to_home() > 7) ==0:
            end_game = True
        for piece in pieces:
            sum_distances = sum_distances + piece.spaces_to_home()
            if piece.spaces_to_home() > 6:
                sum_distances_to_endzone += piece.spaces_to_home() - 6
            if piece.spaces_to_home()>12:
                sum_distance_far_from_home += piece.spaces_to_home() -6
            if piece.spaces_to_home() ==1 and not end_game:
                first_tower += 1
                
        if first_tower>3:
            first_tower = 0 
         
        for location in range(1, 25):
            pieces = myboard.pieces_at(location)
            if len(pieces) != 0 and pieces[0].colour == colour:
                if len(pieces) == 1:
                    number_of_singles = number_of_singles + 1
                    sum_single_distance_away_from_home += 25 - pieces[0].spaces_to_home()
                elif len(pieces) > 1:
                    number_occupied_spaces = number_occupied_spaces + 1
                if len(pieces) > 1:
                    board_control+= 1
                    if 1 <= pieces[0].spaces_to_home() <= 6:
                        home_control += 1
                if len(pieces) == 2:
                    building_of_two += 1
                elif len(pieces) > 3:
                    tower += 1
        opponents_taken_pieces = len(myboard.get_taken_pieces(colour.other()))
        opponent_pieces = myboard.get_pieces(colour.other())
        threat_level = sum(1 for piece in pieces for opponent_piece in opponent_pieces if abs(piece.location - opponent_piece.location) < 8)
        
        sum_distances_opponent = 0
        for piece in opponent_pieces:
            sum_distances_opponent = sum_distances_opponent + piece.spaces_to_home()
    
        return {
            'number_occupied_spaces': number_occupied_spaces,
            'opponents_taken_pieces': opponents_taken_pieces,
            'taken_pieces': taken_pieces,
            'sum_distances': sum_distances,
            'sum_distances_opponent': sum_distances_opponent,
            'number_of_singles': number_of_singles,
            'sum_single_distance_away_from_home': sum_single_distance_away_from_home,
            'pieces_on_board': pieces_on_board,
            'sum_distances_to_endzone': sum_distances_to_endzone,
            'home_control': home_control,
            'board_control': board_control,
            'threat_level': threat_level,
            'sum_distance_far_from_home': sum_distance_far_from_home,
            'building_of_two':building_of_two,
            'tower': tower,
            'first_tower': first_tower
        }
              




    def move(self, board, colour, dice_roll, make_move, opponents_activity):
        self.minmax_deap=0
        log("minimax")
        start_time = time.time()
        if board.getTheTimeLim() != -1:
            self.time_limit = board.getTheTimeLim() - 0.2
        else:
            self.time_limit = -1
        possible_state =  self.get_all_possible_moves(board, colour, dice_roll , start_time, self.time_limit)
        best_val = float('-inf')
        best_moves = []
        #chack if all the pieces are in the endzone
        self.endgame = False
        if sum(1 for piece in board.get_pieces(colour) if piece.spaces_to_home() > 7) ==0:
            self.endgame = True
        #print(possible_state)
            lest_pieces = board.get_pieces(colour)
            lest_rows = list(set(lest_pieces))
            log(f"Possible states: {(possible_state).items()}")
            #sort by spaces to home
            lest_rows.sort(key=lambda x: x.spaces_to_home(), reverse=True)
            if len(lest_rows) <=len (dice_roll):
                dice= sorted(dice_roll, reverse=True)
                for piece in lest_pieces:
                    if board.is_move_possible(piece, dice[0]):
                        make_move(piece.location, dice[0])
                        dice_roll.remove(dice[0])
                        break
                    if len(dice_roll) == 0:
                        return

        for new_board, moves in possible_state.items():
            if self.time_limit != -1:   
                if time.time() - start_time > self.time_limit:
                    break
            if not self.endgame:
                val = self.minmax(new_board, colour, depth=4, maximizing_player=True, alpha=float('-inf'), beta=float('inf'), start_time=start_time, time_limit=self.time_limit)
            else:
                val = self.minmax(new_board, colour, depth=2, maximizing_player=True, alpha=float('-inf'), beta=float('inf'), start_time=start_time, time_limit=self.time_limit)

            if val > best_val:
                best_val = val
                best_moves = moves
        if len(best_moves) != 0:
            for move in best_moves:
                if colour !=  0:
                    log(f"Moving piece at {move['piece_at']} to {move['piece_at'] + move['die_roll']}")
                make_move(move['piece_at'], move['die_roll'])
        
        log(f"Best value: {best_val}")
        board_stat = self.assess_board(colour, board)
        log(f"board_stat: {board_stat}")      
        log(f"minmax_deap: {self.minmax_deap}")
        self.minmax_deap=0
        return
     
        
    def minmax(self, board, colour, depth, maximizing_player, alpha, beta, start_time, time_limit):
        self.minmax_deap = max(self.minmax_deap, 4-depth)
        if self.time_limit != -1:  
            if time.time() - start_time > time_limit:
                return self.evaluate_board(board, colour)
        if depth == 0:
            return self.evaluate_board(board, colour)
        dice_rolls = []
        for diceA in range(1,7):
            for diceB in range(diceA, 7):
                if (diceA!=diceB):
                    dice_rolls.append(([diceA, diceB], 1/36))
                else:
                    dice_rolls.append(([diceA]*4, 1/18))
        
        if maximizing_player:
            max_eval = float('-inf') 
            for dice_roll, prob in dice_rolls:
                possible_states = self.get_all_possible_moves(board, colour, dice_roll, 
                                                            start_time, time_limit)
                
                for new_board in possible_states.keys():
                    eval = self.minmax(new_board, colour, depth-1, False, alpha, beta, 
                                     start_time, time_limit)
                    eval = eval * prob
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    
                    if beta <= alpha:
                        break
                        
            return max_eval
        else:
            min_eval = float('inf')
            
            for dice_roll, prob in dice_rolls:
                possible_states = self.get_all_possible_moves(board, colour.other(), dice_roll,
                                                            start_time, time_limit)
                
                for new_board in possible_states.keys():
                    eval = self.minmax(new_board, colour, depth-1, True, alpha, beta,
                                     start_time, time_limit)
                    eval = eval * prob
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    
                    if beta <= alpha:
                        break
            return min_eval
    
            



        


    
    def create_tree(self, tree, borad):
        for child in range(1,16):
            node = tree.add_children(tree, child)
            node.value = self

    def get_all_possible_moves(self, board, colour, dice_roll, start_time, time_limit):
        if self.time_limit != -1:  
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
        
        # Updated weights based on strategic importance

        weights = {
            'number_occupied_spaces': 3.0 ,
            'opponents_taken_pieces': 2.0,
            'taken_pieces': -5.0,
            'sum_distances': -1.0,
            'sum_distances_opponent': 2.0,
            'number_of_singles': -5000.0 ,
            'sum_single_distance_away_from_home': -3.0,
            'pieces_off_board': 30.0,
            'sum_distances_to_endzone': -15.5,
            'home_control': 300.0 ,
            'board_control': 30.0,
            'threat_level': 0.0,
            'sum_distance_far_from_home': -100.0    
            ,'building_of_two': 0.0,
            'tower': -150.0
            ,'first_tower': -150.0
        }
        # Compute weighted board value
        board_value = sum(weights[key] * board_stats.get(key, 0) for key in weights)
        return board_value


    def evaluate_board1(self, myboard, colour):
        board_stats = self.assess_board(colour, myboard)
        weights = {
            'number_occupied_spaces': 3.0 ,
            'opponents_taken_pieces': 2.0,
            'taken_pieces': -5.0,
            'sum_distances': -1.0,
            'sum_distances_opponent': 0.8,
            'number_of_singles': -525.0 ,
            'sum_single_distance_away_from_home': -3.0,
            'pieces_on_board': -30.0,
            'sum_distances_to_endzone': -0.5,
            'home_control': 300.0 ,
            'board_control': 30.0,
            'threat_level': 0.0,
            'sum_distance_far_from_home': -10.0    
            ,'building_of_two': 0.0,
            'tower': -150.0,
            'first_tower': -70.0
        }
        board_value = sum(weights[key] * board_stats.get(key, 0) for key in weights)
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

        
        

