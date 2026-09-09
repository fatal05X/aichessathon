import os
import sys
import chess
import chess.polyglot
import torch

# Ensure strictly single-threaded execution on the 1 dedicated CPU core
torch.set_num_threads(1)

# Import local flat modules
import config
import state
import tt
import eval as evaluation
import search

# ==========================================
# 90-SECOND INITIALIZATION WARMUP
# ==========================================
# Executes at module import time inside the 90s pre-clock window.
# Compiles all @numba.njit functions before the match clock starts.
evaluation.warmup()

# Optional: Polyglot book reader (if book.bin is packaged in the root)
_BOOK_PATH = "book.bin"
_polyglot_reader = None

if os.path.exists(_BOOK_PATH):
    try:
        _polyglot_reader = chess.polyglot.open_reader(_BOOK_PATH)
    except Exception:
        _polyglot_reader = None


# ==========================================
# AGENT API ENTRY POINT
# ==========================================
def get_move(fen: str, time_left_ms: int) -> str:
    """
    Mandatory platform API called on each of our turns.

    Args:
        fen: Current position FEN string.
        time_left_ms: Remaining clock in milliseconds.

    Returns:
        A legal move in UCI string format (e.g., 'e2e4' or 'e7e8q').
    """
    board = chess.Board(fen)

    # 1. Update Match State
    # Register the position we were just presented with
    state.register_position(board)

    # Safety check: if no legal moves exist (game over), return empty fallback
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return ""

    # 2. Check Polyglot Opening Book
    # Even though rated games start from curated positions, if the FEN matches
    # our book or transposes into known theory, take the 0ms move.
    if _polyglot_reader is not None:
        try:
            entry = _polyglot_reader.get(board)
            if entry is not None and entry.move in board.legal_moves:
                chosen_move_uci = entry.move.uci()

                # Register our played move into the position state tracker
                board.push(entry.move)
                state.register_position(board)
                return chosen_move_uci
        except Exception:
            pass

    # 3. Tree Search
    # Iterative deepening Alpha-Beta with dynamic time budgeting
    chosen_move_uci = search.get_best_move(board, time_left_ms)

    # 4. Strict Legality Verification
    # Ensure the returned move parses and is 100% legal
    try:
        move_obj = chess.Move.from_uci(chosen_move_uci)
        if move_obj not in board.legal_moves:
            chosen_move_uci = legal_moves[0].uci()
            move_obj = legal_moves[0]
    except Exception:
        chosen_move_uci = legal_moves[0].uci()
        move_obj = legal_moves[0]

    # 5. Register Resulting Board State
    # Push our move to track the resulting position for repetition avoidance
    board.push(move_obj)
    state.register_position(board)

    return chosen_move_uci