import chess
import config

# ==========================================
# GLOBAL MATCH STATE
# ==========================================
# The container starts fresh for every game, so global variables
# safely persist for exactly one match without needing a reset mechanism.
_position_counts = {}
_plies_played = 0
_initial_fen = None


def register_position(board: chess.Board):
    """
    Registers a board state in the match history.
    This must be called twice per turn:
    1. Once when receiving the new FEN from the opponent.
    2. Once after we apply our chosen move, to track the opponent's upcoming turn.
    """
    global _position_counts, _plies_played, _initial_fen

    if _initial_fen is None:
        _initial_fen = board.fen()

    key = board._transposition_key()
    _position_counts[key] = _position_counts.get(key, 0) + 1
    _plies_played += 1


def get_repetition_count(board: chess.Board) -> int:
    """
    Returns the number of times the current board's exact piece configuration,
    castling rights, and en passant rights have occurred.
    """
    key = board._transposition_key()
    return _position_counts.get(key, 0)


def is_draw_approaching(board: chess.Board) -> bool:
    """
    Evaluates if the current position is dangerously close to an automatic referee draw.
    The referee strictly enforces the 50-move rule (100 halfmoves) and threefold repetition.
    """
    # 1. 50-move rule check (referee draws at 100 halfmoves)
    if board.halfmove_clock >= 100:
        return True

    # 2. Repetition check (if it has happened 2 or more times, the next one is a draw)
    if get_repetition_count(board) >= 2:
        return True

    # 3. Hard game cap check (games still running at 600 plies are drawn)
    if _plies_played >= config.MAX_GAME_PLIES:
        return True

    return False


def get_plies_played() -> int:
    """Returns the total number of plies (half-moves) tracked so far."""
    return _plies_played