import os
import sys
import chess
import chess.polyglot
import chess.syzygy
import torch
from pathlib import Path

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

# 1. Polyglot Book Reader (Optional midgame/trap book)
_BOOK_PATH = "book.bin"
_polyglot_reader = None

if os.path.exists(_BOOK_PATH):
    try:
        _polyglot_reader = chess.polyglot.open_reader(_BOOK_PATH)
    except Exception:
        _polyglot_reader = None

# 2. Syzygy Tablebase Reader (Optional 3- and 4-man endgames)
# We look in the current flat root directory for .rtbw and .rtbz files
_SYZYGY_PATH = str(Path(__file__).resolve().parent)
_tablebase = None

try:
    _tablebase = chess.syzygy.open_tablebase(_SYZYGY_PATH)
except Exception:
    _tablebase = None


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

    # 2. Check Polyglot Book (0ms)
    # If the FEN matches our book or transposes into known theory, take it.
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

    # 3. Probe Syzygy Tablebases (0ms endgame conversion)
    # If we have 4 or fewer pieces on the board, try for a perfect tablebase move.
    if _tablebase is not None and len(board.piece_map()) <= 4:
        try:
            best_move = None
            best_wdl = -2
            best_dtz = -99999

            for move in legal_moves:
                board.push(move)
                try:
                    # 1. Probe the tables from the opponent's perspective
                    wdl = -_tablebase.probe_wdl(board)
                    dtz = _tablebase.probe_dtz(board)
                finally:
                    board.pop()  # ALWAYS restore the board

                # 2. Maximize WDL first
                if wdl > best_wdl:
                    best_wdl = wdl
                    best_dtz = dtz
                    best_move = move

                # 3. If WDL is tied, maximize the opponent's DTZ to force fast wins / slow losses
                elif wdl == best_wdl and dtz > best_dtz:
                    best_dtz = dtz
                    best_move = move

            if best_move is not None:
                board.push(best_move)
                state.register_position(board)
                return best_move.uci()
        except Exception as e:
            print(e)  # Fall back to standard search if a table is missing

    # 4. Tree Search
    # Iterative deepening Alpha-Beta with dynamic time budgeting and LMR/NMP
    chosen_move_uci = search.get_best_move(board, time_left_ms)

    # 5. Strict Legality Verification
    # Ensure the returned move parses and is 100% legal
    try:
        move_obj = chess.Move.from_uci(chosen_move_uci)
        if move_obj not in board.legal_moves:
            chosen_move_uci = legal_moves[0].uci()
            move_obj = legal_moves[0]
    except Exception:
        chosen_move_uci = legal_moves[0].uci()
        move_obj = legal_moves[0]

    # 6. Register Resulting Board State
    # Push our move to track the resulting position for repetition avoidance
    board.push(move_obj)
    state.register_position(board)

    return chosen_move_uci