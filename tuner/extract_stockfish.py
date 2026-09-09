import argparse
import glob
import os
import chess
import numpy as np
import pandas as pd

# Game phase constants matching eval.py
PHASE_INC = np.array([0, 0, 1, 1, 2, 4, 0], dtype=np.int32)
MAX_PHASE = 24

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

def process_parquet_file(file_path: str, max_positions_per_file: int):
    """
    Reads a single parquet file, parses FENs and centipawns,
    and returns boards, phases, and win-probability targets.
    """
    print(f"\nProcessing {os.path.basename(file_path)}...")
    try:
        sample = pd.read_parquet(file_path)
        eval_col = next((col for col in ["cp", "evaluation", "eval"] if col in sample.columns), None)

        if eval_col is None or "fen" not in sample.columns:
            print(f"  [Skipped] Required columns not found in {file_path}. Columns: {list(sample.columns)}")
            return [], [], []

        df = sample[["fen", eval_col]].copy()
        del sample
    except Exception as e:
        print(f"  [Error] Could not read {file_path}: {e}")
        return [], [], []

    # Clean numeric centipawns (removes string-based mate scores)
    df[eval_col] = pd.to_numeric(df[eval_col], errors="coerce")
    df = df.dropna(subset=[eval_col])

    if max_positions_per_file > 0 and len(df) > max_positions_per_file:
        df = df.sample(n=max_positions_per_file, random_state=42)

    boards, phases, targets = [], [], []

    for idx, row in enumerate(df.itertuples(index=False)):
        try:
            fen_str = getattr(row, "fen")
            board = chess.Board(fen_str)
        except Exception:
            continue

        white_eval = float(getattr(row, eval_col))

        # Win probability mapping: P(Win) = 1 / (1 + 10^(-eval / 400))
        target_prob = 1.0 / (1.0 + 10.0 ** (-white_eval / 400.0))

        arr = board_to_array(board)
        phase = compute_phase(arr)

        boards.append(arr)
        phases.append(phase)
        targets.append(target_prob)

        if (idx + 1) % 50_000 == 0:
            print(f"  Extracted {idx + 1:,} / {len(df):,} positions...")

    print(f"  Finished {os.path.basename(file_path)}: {len(boards):,} valid samples.")
    return boards, phases, targets

def run_pipeline(data_dir: str, max_per_file: int):
    """Scans data_dir for .parquet files and extracts records into dataset.npz."""
    parquet_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not parquet_files:
        print(f"No .parquet files found in directory: {data_dir}")
        return

    all_boards, all_phases, all_targets = [], [], []

    for file_path in parquet_files:
        boards, phases, targets = process_parquet_file(file_path, max_per_file)
        all_boards.extend(boards)
        all_phases.extend(phases)
        all_targets.extend(targets)

    if not all_boards:
        print("No valid positions extracted.")
        return

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.npz")
    print(f"\nSaving {len(all_boards):,} positions to {out_path}...")
    np.savez_compressed(
        out_path,
        boards=np.array(all_boards, dtype=np.int8),
        phases=np.array(all_phases, dtype=np.int8),
        targets=np.array(all_targets, dtype=np.float32),
    )
    print("Dataset extraction successfully completed!")

if __name__ == "__main__":
    tuner_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(tuner_dir, "data")

    parser = argparse.ArgumentParser(description="Batch extract Kaggle Parquet datasets.")
    parser.add_argument("--data-dir", type=str, default=default_data_dir, help="Directory containing .parquet files.")
    parser.add_argument("--max-per-file", type=int, default=250_000, help="Max positions per file.")
    args = parser.parse_args()

    run_pipeline(data_dir=args.data_dir, max_per_file=args.max_per_file)