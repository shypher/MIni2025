
    def minmax1(self, board, colour, dice_rolls, depth, maximizing_player, alpha, beta, start_time, time_limit, tree_log=None):
        if time.time() - start_time > time_limit:
            return {'best_value': self.evaluate_board(board, colour), 'best_moves': []}

        dice_rolls_left = dice_rolls.copy()
        if not dice_rolls_left:
            eval_value = self.evaluate_board(board, colour)
            if tree_log:
                tree_log.write(f"{'|   ' * depth}Leaf: Evaluation = {eval_value}\n")
            return {'best_value': eval_value, 'best_moves': []}

        die_roll = dice_rolls_left.pop(0)
        valid_moves = self.get_valid_moves(board, colour, die_roll)

        if tree_log:
            tree_log.write(f"{'|   ' * depth}Depth {depth}: {'Max' if maximizing_player else 'Min'} Player, Die Roll = {die_roll}\n")

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
                    dice_rolls=dice_rolls_left,
                    depth=depth + 1,
                    maximizing_player=False,
                    alpha=alpha,
                    beta=beta,
                    start_time=start_time,
                    time_limit=time_limit,
                    tree_log=tree_log
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
                    dice_rolls=dice_rolls_left,
                    depth=depth + 1,
                    maximizing_player=True,
                    alpha=alpha,
                    beta=beta,
                    start_time=start_time,
                    time_limit=time_limit,
                    tree_log=tree_log
                )

                if result['best_value'] < best_value:
                    best_value = result['best_value']
                    best_moves = [move] + result['best_moves']

                beta = min(beta, best_value)
                if beta <= alpha:
                    break
            return {'best_value': best_value, 'best_moves': best_moves}

    def minmax(self, board, colour, dice_rolls, depth, maximizing_player, alpha, beta, start_time, time_limit, tree_log=None):
        if time.time() - start_time > time_limit:
            return {'best_value': self.evaluate_board(board, colour), 'best_moves': []}
        dice_rolls_left = dice_rolls.copy()
        if not dice_rolls_left:
            eval_value = self.evaluate_board(board, colour)
            if tree_log:
                tree_log.write(f"{'|   ' * depth}Leaf: Evaluation = {eval_value}\n")
            return {'best_value': self.evaluate_board(board, colour), 'best_moves': []}
        die_roll = dice_rolls_left.pop(0)
        valid_moves = self.get_valid_moves(board, colour, die_roll)
        if tree_log:
            tree_log.write(f"{'|   ' * depth}Depth {depth}: {'Max' if maximizing_player else 'Min'} Player, Die Roll = {die_roll}\n")
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
                    result = self.minmax(board_copy, colour, dice_rolls_left, depth + 1, False, alpha, beta, start_time, time_limit,tree_log=tree_log)
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
                result = self.minmax(board_copy, colour, dice_rolls_left, depth + 1, True, alpha, beta, start_time, time_limit,tree_log=tree_log)
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
        """