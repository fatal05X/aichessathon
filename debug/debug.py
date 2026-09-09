import sys
import os
import chess.pgn

# 1. Dynamically add the parent directory to sys.path so we can import agent.py
# This prevents namespace shadowing issues and lets us keep the debug folder separate.
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import agent  # Now this successfully imports your root agent.py


def run_debug():
    # 2. Reference the exact PGN file sitting next to this script
    pgn_filename = "aichessathon-round-85-tristanize-chess-vs-1caramel.pgn"
    pgn_path = os.path.join(os.path.dirname(__file__), pgn_filename)

    if not os.path.exists(pgn_path):
        print(f"ERROR: Could not find {pgn_filename} in the debug directory.")
        return

    print(f"Loading {pgn_filename}...")

    # 3. Parse the PGN to reach the final position
    with open(pgn_path, "r", encoding="utf-8") as pgn_file:
        game = chess.pgn.read_game(pgn_file)

    if game is None:
        print("ERROR: PGN file was empty or unreadable.")
        return

    board = game.end().board()

    print(f"\nFEN after 73. Kxd1: {board.fen()}")
    print(f"Legal moves for Black: {[m.uci() for m in board.legal_moves]}\n")

    # 4. Test your agent's output
    try:
        # Simulating ~16.8 seconds remaining based on Black's last clock + 0.5s increment
        move = agent.get_move(board.fen(), 16828)
        print(f"Agent returned: '{move}'")

        if move not in [m.uci() for m in board.legal_moves]:
            print(f"FAILURE: '{move}' is an illegal move or empty string!")
        else:
            print(f"SUCCESS: '{move}' is a valid, legal move.")

    except Exception as e:
        print(f"CRASH: Agent threw an exception: {e}")


if __name__ == "__main__":
    run_debug()