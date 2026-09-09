import argparse
import os
import sys
import chess
import chess.pgn
import numpy as np

# Phase increment values matching eval.py
PHASE_INC = np.array([0, 0, 1, 1, 2, 4, 0], dtype=np.int32)
MAX_PHASE = 24


def is_quiet(board: chess.Board) -> bool:
    """
    Checks if a position is tactically quiescent (no checks and no captures).
    Avoids positions mid-exchange that skew static evaluation.
    """
    if board.is_check():
        return False
    # If any legal move is a capture or promotion, skip this position
    for move in board.legal_moves:
        if board.is_capture(move) or move.promotion:
            return False
    return True


def board_to_array(board: chess.Board) -> np.ndarray:
    """Encodes a board into the 64-element array format matching eval.py."""
    arr = np.zeros(64, dtype=np.int8)
    for sq, piece in board.piece_map().items():
        arr[sq] = piece.piece_type + (0 if piece.color == chess.WHITE else 6)
    return arr


def compute_phase(board_array: np.ndarray) -> int:
    """Computes game phase matching eval.py logic."""
    phase = MAX_PHASE
    for sq in range(64):
        p = board_array[sq]
        if p != 0:
            pt = p if p < 7 else p - 6
            phase -= PHASE_INC[pt]
    return max(0, min(MAX_PHASE, phase))


def extract_pgn(pgn_path: str, max_positions: int = 500_000, min_ply: int = 16):
    """
    Parses games from a PGN file and compiles quiet positions into an NPZ dataset.
    """
    if not os.path.exists(pgn_path):
        print(f"Error: PGN file not found at {pgn_path}")
        sys.exit(1)

    print(f"Extracting up to {max_positions:,} positions from {pgn_path}...")

    boards_list = []
    phases_list = []
    targets_list = []

    total_games = 0
    total_positions = 0

    outcome_map = {
        "1-0": 1.0,      # White win
        "1/2-1/2": 0.5,  # Draw
        "0-1": 0.0       # Black win
    }

    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as pgn_file:
        while total_positions < max_positions:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            total_games += 1
            result_str = game.headers.get("Result", "*")
            if result_str not in outcome_map:
                continue

            target = outcome_map[result_str]
            board = game.board()

            for ply, move in enumerate(game.mainline_moves()):
                board.push(move)

                # Skip early opening moves and noisy tactical positions
                if ply < min_ply:
                    continue

                if is_quiet(board):
                    arr = board_to_array(board)
                    phase = compute_phase(arr)

                    boards_list.append(arr)
                    phases_list.append(phase)
                    targets_list.append(target)
                    total_positions += 1

                    if total_positions % 25_000 == 0:
                        print(f"  Extracted {total_positions:,} quiet positions from {total_games:,} games...")

                    if total_positions >= max_positions:
                        break

    print(f"Extraction complete! Total positions: {len(boards_list):,}")

    out_path = os.path.join(os.path.dirname(__file__), "dataset.npz")
    np.savez_compressed(
        out_path,
        boards=np.array(boards_list, dtype=np.int8),
        phases=np.array(phases_list, dtype=np.int8),
        targets=np.array(targets_list, dtype=np.float32),
    )
    print(f"Saved dataset to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract quiet chess positions for Texel tuning.")
    parser.add_argument("pgn", type=str, help="Path to input PGN file.")
    parser.add_argument("--max", type=int, default=300_000, help="Max positions to extract.")
    parser.add_argument("--min-ply", type=int, default=16, help="Skip moves before this ply.")
    args = parser.parse_args()

    extract_pgn(args.pgn, max_positions=args.max, min_ply=args.min_ply)