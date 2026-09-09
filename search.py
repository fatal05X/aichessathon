import time
import chess
import config
import tt
import state
import eval as evaluation

# ==========================================
# SEARCH STATE & GLOBALS
# ==========================================
_abort_search = False
_start_time = 0.0
_time_budget_sec = 0.0
_nodes_searched = 0

# Move Ordering Heuristics Tables
_killer_moves = [[None, None] for _ in range(config.MAX_PLY)]
_history_table = [[0] * 64 for _ in range(64)]

# MVV-LVA (Most Valuable Victim - Least Valuable Attacker) table base values
_PIECE_VALUES = {
    chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000
}


# ==========================================
# TIME MANAGEMENT
# ==========================================
def _check_time():
    """Polls wall-clock time every 1024 nodes to prevent flagging."""
    global _abort_search
    if _abort_search:
        return
    if time.time() - _start_time >= _time_budget_sec:
        _abort_search = True


# ==========================================
# MOVE ORDERING
# ==========================================
def _score_move(board: chess.Board, move: chess.Move, hash_move: chess.Move, ply: int) -> int:
    """Assigns a numerical score to a move to optimize Alpha-Beta pruning."""
    if move == hash_move:
        return 2000000  # Highest priority: TT Best Move

    if board.is_capture(move):
        # MVV-LVA Scoring
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)

        # Handle en passant explicitly
        if victim is None and board.is_en_passant(move):
            v_val = _PIECE_VALUES[chess.PAWN]
        else:
            v_val = _PIECE_VALUES.get(victim.piece_type, 0) if victim else 0

        a_val = _PIECE_VALUES.get(attacker.piece_type, 0) if attacker else 0
        return 1000000 + v_val * 10 - a_val

    else:
        # Quiet moves
        if ply < config.MAX_PLY:
            if move == _killer_moves[ply][0]:
                return 900000
            elif move == _killer_moves[ply][1]:
                return 800000

        # History Heuristic
        return _history_table[move.from_square][move.to_square]


def _order_moves(board: chess.Board, moves, hash_move: chess.Move, ply: int):
    """Sorts legal moves in-place based on their heuristic score."""
    moves_list = list(moves)
    moves_list.sort(key=lambda m: _score_move(board, m, hash_move, ply), reverse=True)
    return moves_list


# ==========================================
# QUIESCENCE SEARCH
# ==========================================
def _quiescence(board: chess.Board, alpha: int, beta: int, ply: int) -> int:
    """Extends search along tactical lines (captures) to avoid the horizon effect."""
    global _nodes_searched, _abort_search

    _nodes_searched += 1
    if _nodes_searched & 1023 == 0:
        _check_time()
    if _abort_search:
        return 0

    if state.is_draw_approaching(board) or board.is_repetition(2):
        return config.DRAW_SCORE

    stand_pat = evaluation.evaluate(board)

    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat

    captures = (m for m in board.legal_moves if board.is_capture(m) or m.promotion)
    ordered_captures = _order_moves(board, captures, None, ply)

    for move in ordered_captures:
        board.push(move)
        score = -_quiescence(board, -beta, -alpha, ply + 1)
        board.pop()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha


# ==========================================
# NEGAMAX ALPHA-BETA SEARCH
# ==========================================
def _negamax(board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
    """Core recursive tree search with advanced pruning techniques."""
    global _nodes_searched, _abort_search

    _nodes_searched += 1
    if _nodes_searched & 1023 == 0:
        _check_time()
    if _abort_search:
        return 0

    if state.is_draw_approaching(board) or board.is_repetition(2):
        return config.DRAW_SCORE

    # Mate distance pruning
    alpha = max(alpha, -config.MATE_BOUND + ply)
    beta = min(beta, config.MATE_BOUND - ply)
    if alpha >= beta:
        return alpha

    zobrist_key = board._transposition_key()
    tt_score, hash_move_uci = tt.probe(zobrist_key, depth, alpha, beta)

    hash_move = chess.Move.from_uci(hash_move_uci) if hash_move_uci else None
    if tt_score is not None:
        return tt_score

    in_check = board.is_check()

    if depth <= 0:
        return _quiescence(board, alpha, beta, ply)

    # 1. Null Move Pruning (NMP)
    # If not in check, deep enough, and we still have non-pawn pieces
    if not in_check and depth >= 3:
        has_non_pawns = any(
            board.pieces(pt, board.turn) for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
        if has_non_pawns:
            board.push(chess.Move.null())
            # Search with reduced depth (R=2) and a zero window
            null_score = -_negamax(board, depth - 3, -beta, -beta + 1, ply + 1)
            board.pop()
            if null_score >= beta:
                return beta

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        if in_check:
            return -config.MATE_SCORE + ply
        return config.DRAW_SCORE

    ordered_moves = _order_moves(board, legal_moves, hash_move, ply)
    best_move = None
    best_score = -config.INFINITY
    tt_flag = tt.TT_UPPERBOUND
    move_index = 0

    # Check Extension
    extension = 1 if in_check else 0

    for move in ordered_moves:
        move_index += 1
        is_capture = board.is_capture(move)
        gives_check = board.gives_check(move)

        board.push(move)

        new_depth = depth - 1 + extension

        # 2. Late Move Reductions (LMR)
        if depth >= 3 and move_index >= 4 and not is_capture and not in_check and not gives_check and (
                ply >= config.MAX_PLY or move not in _killer_moves[ply]):
            # Search with reduced depth
            score = -_negamax(board, new_depth - 1, -alpha - 1, -alpha, ply + 1)
            if score > alpha:
                # Re-search at full depth if it beats alpha
                score = -_negamax(board, new_depth, -beta, -alpha, ply + 1)
        else:
            # Normal Principal Variation or Tactical Search
            score = -_negamax(board, new_depth, -beta, -alpha, ply + 1)

        board.pop()

        if _abort_search:
            return 0

        if score > best_score:
            best_score = score
            best_move = move

        if score > alpha:
            alpha = score
            tt_flag = tt.TT_EXACT

            if score >= beta:
                # Alpha-Beta Cutoff (Fail-High)
                tt.store(zobrist_key, depth, tt.TT_LOWERBOUND, beta, move.uci())

                # Update heuristics for quiet moves
                if not is_capture:
                    if ply < config.MAX_PLY:
                        _killer_moves[ply][1] = _killer_moves[ply][0]
                        _killer_moves[ply][0] = move
                    _history_table[move.from_square][move.to_square] += depth * depth

                return beta

    if best_move:
        tt.store(zobrist_key, depth, tt_flag, alpha, best_move.uci())

    return alpha


# ==========================================
# ITERATIVE DEEPENING & ENTRY POINT
# ==========================================
def get_best_move(board: chess.Board, time_left_ms: int) -> str:
    """
    Main entry point for calculating a move.
    Applies dynamic time budgeting and Aspiration Windows.
    """
    global _abort_search, _start_time, _time_budget_sec, _nodes_searched
    global _killer_moves, _history_table

    _start_time = time.time()
    _abort_search = False
    _nodes_searched = 0

    # 1. Dynamic Time Budgeting
    expected_moves_left = max(config.TIME_MIN_DIVISOR, 40 - (state.get_plies_played() // 2))
    allocated_ms = time_left_ms / expected_moves_left

    if allocated_ms > time_left_ms * 0.8:
        allocated_ms = time_left_ms * 0.8

    _time_budget_sec = allocated_ms / 1000.0

    # Age heuristics slightly
    for i in range(64):
        for j in range(64):
            _history_table[i][j] //= 2

    best_move_global = list(board.legal_moves)[0].uci()
    best_score_global = 0

    # 2. Iterative Deepening Loop with Aspiration Windows
    for depth in range(1, config.MAX_DEPTH + 1):

        # Setup Aspiration Window
        if depth >= 4:
            alpha = best_score_global - 35
            beta = best_score_global + 35
        else:
            alpha = -config.INFINITY
            beta = config.INFINITY

        while True:
            _, root_hash_move_uci = tt.probe(board._transposition_key(), 0, alpha, beta)
            root_hash_move = chess.Move.from_uci(root_hash_move_uci) if root_hash_move_uci else None
            ordered_root_moves = _order_moves(board, list(board.legal_moves), root_hash_move, 0)

            best_move_this_iteration = None
            best_score_this_iteration = -config.INFINITY

            for move in ordered_root_moves:
                board.push(move)
                score = -_negamax(board, depth - 1, -beta, -alpha, 1)
                board.pop()

                if _abort_search:
                    break

                if score > best_score_this_iteration:
                    best_score_this_iteration = score
                    best_move_this_iteration = move.uci()

                if score > alpha:
                    alpha = score

            if _abort_search:
                break

            # 3. Check Aspiration Window Bounds
            if depth >= 4 and (
                    best_score_this_iteration <= best_score_global - 35 or best_score_this_iteration >= best_score_global + 35):
                # Fell outside the window (Fail-High or Fail-Low), re-search at full window
                alpha = -config.INFINITY
                beta = config.INFINITY
                continue

            break  # Window held, proceed to next depth

        if _abort_search:
            break

        if best_move_this_iteration:
            best_move_global = best_move_this_iteration
            best_score_global = best_score_this_iteration

        # Optional: Early exit if we find a forced mate
        if best_score_this_iteration >= config.MATE_BOUND - 100:
            break

    return best_move_global