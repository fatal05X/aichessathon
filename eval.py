import chess
import numpy as np
from numba import njit
import config
import state

# ==========================================
# PESTO PIECE-SQUARE TABLES & WEIGHTS
# ==========================================
# Piece mapping:
# 0: Empty, 1: wP, 2: wN, 3: wB, 4: wR, 5: wQ, 6: wK
#           7: bP, 8: bN, 9: bB, 10: bR, 11: bQ, 12: bK

# Base piece values (Midgame, Endgame)
MG_VALUE = np.array([0, 82, 337, 365, 477, 1025, 0], dtype=np.int32)
EG_VALUE = np.array([0, 94, 281, 297, 512, 936, 0], dtype=np.int32)

# Game phase limits for tapering
PHASE_INC = np.array([0, 0, 1, 1, 2, 4, 0], dtype=np.int32)
MAX_PHASE = 24

# We define a simplified but highly effective symmetrical PeSTO table set for compilation limits.
# Arrays are pre-expanded to (13, 64) for instant Numba lookups.
_mg_table = np.zeros((13, 64), dtype=np.int32)
_eg_table = np.zeros((13, 64), dtype=np.int32)


def _init_tables():
    """Populate the flat lookup arrays dynamically to keep the source compact."""
    # Simplified center-weighted tables for minor pieces, edge-weighted for rooks
    center_bonus = np.array([
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10, 5, 5, 5, 5, 5, -10, -10,
        -10, 5, 10, 15, 15, 10, 5, -10,
        -10, 5, 15, 20, 20, 15, 5, -10,
        -10, 5, 15, 20, 20, 15, 5, -10,
        -10, 5, 10, 15, 15, 10, 5, -10,
        -10, 5, 5, 5, 5, 5, -10, -10,
        -20, -10, -10, -10, -10, -10, -10, -20
    ], dtype=np.int32)

    pawn_rank_mg = np.array([0, 5, 10, 20, 30, 50, 70, 0])
    pawn_rank_eg = np.array([0, 10, 20, 30, 50, 70, 90, 0])

    for sq in range(64):
        rank, file = sq // 8, sq % 8
        flip_sq = (7 - rank) * 8 + file

        # White Pieces
        _mg_table[1][sq] = MG_VALUE[1] + pawn_rank_mg[rank] + center_bonus[sq] // 2
        _eg_table[1][sq] = EG_VALUE[1] + pawn_rank_eg[rank]

        for p in range(2, 6):  # N, B, R, Q
            _mg_table[p][sq] = MG_VALUE[p] + center_bonus[sq]
            _eg_table[p][sq] = EG_VALUE[p] + (center_bonus[sq] // 2)

        _mg_table[6][sq] = 0 - center_bonus[sq]  # King safety in MG
        _eg_table[6][sq] = center_bonus[sq]  # King active in EG

        # Black Pieces (Mirrored)
        _mg_table[7][flip_sq] = -_mg_table[1][sq]
        _eg_table[7][flip_sq] = -_eg_table[1][sq]
        for p in range(2, 7):
            _mg_table[p + 6][flip_sq] = -_mg_table[p][sq]
            _eg_table[p + 6][flip_sq] = -_eg_table[p][sq]


_init_tables()


# ==========================================
# NUMBA JIT EVALUATION
# ==========================================
@njit(cache=False)
def _evaluate_array(board_array: np.ndarray, mg_table: np.ndarray, eg_table: np.ndarray, phase_inc: np.ndarray) -> int:
    """
    Highly optimized static evaluation executed in machine code.
    Evaluates the board purely from White's absolute perspective.
    """
    mg_score = 0
    eg_score = 0
    phase = MAX_PHASE

    for sq in range(64):
        p = board_array[sq]
        if p != 0:
            mg_score += mg_table[p][sq]
            eg_score += eg_table[p][sq]

            # Map black pieces back to 1-6 range for phase calculation
            pt = p if p < 7 else p - 6
            phase -= phase_inc[pt]

    # Cap phase to prevent out-of-bounds in weird promotions
    if phase < 0: phase = 0
    if phase > MAX_PHASE: phase = MAX_PHASE

    # Tapered evaluation interpolation
    score = (mg_score * (MAX_PHASE - phase) + eg_score * phase) // MAX_PHASE
    return score


# ==========================================
# PYTHON INTERFACE
# ==========================================
def evaluate(board: chess.Board) -> int:
    """
    Evaluates the current board state.
    Returns the score relative to the side to move (positive = good for side to move).
    """
    # 1. Translate python-chess board to flat NumPy integer array
    board_array = np.zeros(64, dtype=np.int32)
    for sq, piece in board.piece_map().items():
        # Map: White = 1..6, Black = 7..12
        board_array[sq] = piece.piece_type + (0 if piece.color else 6)

    # 2. Compute absolute score (from White's perspective) using Numba
    score = _evaluate_array(board_array, _mg_table, _eg_table, PHASE_INC)

    # 3. Flip score if it is Black's turn (Negamax requirement)
    if board.turn == chess.BLACK:
        score = -score

    # 4. Anti-Draw / Contempt Adjustment
    # If the score is positive (we are winning), but we are about to trigger
    # a referee draw (600 plies or 3fold rep), heavily penalize the position.
    if score > 0 and state.is_draw_approaching(board):
        score += config.CONTEMPT_FACTOR

    return score


def warmup():
    """
    Must be called during the 90s init window in agent.py to force Numba
    to compile the JIT function before the game clock starts.
    """
    dummy_board = np.zeros(64, dtype=np.int32)
    dummy_board[0] = 1  # Just place one piece to prevent out-of-bounds optimization
    _evaluate_array(dummy_board, _mg_table, _eg_table, PHASE_INC)