import os
import chess
import numpy as np
from numba import njit
import config
import state

# ==========================================
# CONSTANTS & WEIGHT SPECIFICATION
# ==========================================
_WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), "weights.npy")

# Base piece values (Midgame, Endgame)
# Index: 0: Empty, 1: P, 2: N, 3: B, 4: R, 5: Q, 6: K
MG_VALUE = np.array([0, 82, 337, 365, 477, 1025, 0], dtype=np.int32)
EG_VALUE = np.array([0, 94, 281, 297, 512, 936, 0], dtype=np.int32)

# Phase increments for non-pawn material (max phase = 24)
PHASE_INC = np.array([0, 0, 1, 1, 2, 4, 0], dtype=np.int32)
MAX_PHASE = 24

# Tables: 13 piece types (0=Empty, 1..6=White, 7..12=Black) x 64 squares
_mg_table = np.zeros((13, 64), dtype=np.int32)
_eg_table = np.zeros((13, 64), dtype=np.int32)
_is_warmed_up = False


def _init_default_tables():
    """Populate default PeSTO piece-square tables with rich positional heuristics."""
    global _mg_table, _eg_table

    center_bonus = np.array([
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10,   5,   5,   5,   5,   5, -10, -10,
        -10,   5,  10,  15,  15,  10,   5, -10,
        -10,   5,  15,  20,  20,  15,   5, -10,
        -10,   5,  15,  20,  20,  15,   5, -10,
        -10,   5,  10,  15,  15,  10,   5, -10,
        -10,   5,   5,   5,   5,   5, -10, -10,
        -20, -10, -10, -10, -10, -10, -10, -20
    ], dtype=np.int32)

    pawn_rank_mg = np.array([0, 5, 10, 20, 30, 50, 70, 0], dtype=np.int32)
    pawn_rank_eg = np.array([0, 10, 20, 30, 50, 70, 90, 0], dtype=np.int32)

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
        _eg_table[6][sq] = center_bonus[sq]      # King active in EG

        # Black Pieces (Mirrored & Negated)
        _mg_table[7][flip_sq] = -_mg_table[1][sq]
        _eg_table[7][flip_sq] = -_eg_table[1][sq]
        for p in range(2, 7):
            _mg_table[p + 6][flip_sq] = -_mg_table[p][sq]
            _eg_table[p + 6][flip_sq] = -_eg_table[p][sq]


def _load_weights():
    """Loads weights.npy if present; falls back to PeSTO defaults otherwise."""
    global _mg_table, _eg_table
    if os.path.exists(_WEIGHTS_FILE):
        try:
            data = np.load(_WEIGHTS_FILE)
            if data.shape == (2, 13, 64):
                _mg_table = data[0].astype(np.int32)
                _eg_table = data[1].astype(np.int32)
                return
        except Exception:
            pass
    _init_default_tables()


# Initialize tables on import
_load_weights()


# ==========================================
# NUMBA JIT EVALUATION
# ==========================================
@njit(cache=False)
def _evaluate_array(board_array: np.ndarray, mg_table: np.ndarray, eg_table: np.ndarray, phase_inc: np.ndarray) -> int:
    """
    Highly optimized static evaluation executed in machine code.
    Computes piece scores and game phase simultaneously in a single pass.
    """
    mg_score = 0
    eg_score = 0
    phase = MAX_PHASE

    for sq in range(64):
        p = board_array[sq]
        if p != 0:
            mg_score += mg_table[p][sq]
            eg_score += eg_table[p][sq]

            pt = p if p < 7 else p - 6
            phase -= phase_inc[pt]

    if phase < 0:
        phase = 0
    elif phase > MAX_PHASE:
        phase = MAX_PHASE

    # Interpolate: phase=0 (all non-pawns present) -> mg_score; phase=24 (no non-pawns) -> eg_score
    score = (mg_score * (MAX_PHASE - phase) + eg_score * phase) // MAX_PHASE
    return score


# ==========================================
# PYTHON INTERFACE
# ==========================================
def evaluate(board: chess.Board) -> int:
    """
    Evaluates current board state relative to the side to move.
    """
    # 1. Translate board to flat array
    board_array = np.zeros(64, dtype=np.int32)
    for sq, piece in board.piece_map().items():
        board_array[sq] = piece.piece_type + (0 if piece.color else 6)

    # 2. Score via JIT Numba function
    score = _evaluate_array(board_array, _mg_table, _eg_table, PHASE_INC)

    # 3. Flip score for side to move (Negamax requirement)
    if board.turn == chess.BLACK:
        score = -score

    # 4. Anti-Draw / Repetition Contempt Adjustment
    if score > 0 and state.is_draw_approaching(board):
        score += config.CONTEMPT_FACTOR

    return score


def warmup():
    """
    Runs during the 90s init window via agent.py.
    Ensures weights are loaded and JIT compilation overhead is paid before move 1.
    """
    global _is_warmed_up
    if _is_warmed_up:
        return

    _load_weights()

    dummy_board = np.zeros(64, dtype=np.int32)
    dummy_board[0] = 1
    _evaluate_array(dummy_board, _mg_table, _eg_table, PHASE_INC)

    _is_warmed_up = True