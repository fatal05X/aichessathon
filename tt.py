import config

# ==========================================
# TT FLAGS
# ==========================================
# These flags indicate the type of score stored based on alpha-beta pruning bounds
TT_EXACT = 0  # The score is an exact evaluation (PV node)
TT_LOWERBOUND = 1  # The score is a lower bound (Beta cutoff / Fail-high)
TT_UPPERBOUND = 2  # The score is an upper bound (Alpha cutoff / Fail-low)

# ==========================================
# GLOBAL TABLE MEMORY
# ==========================================
# The container process is suspended between moves, meaning this dictionary
# will safely persist its data across your own turns without needing to write to disk.
_tt = {}


def probe(zobrist_key: int, depth: int, alpha: int, beta: int):
    """
    Checks the transposition table for a previously searched position.

    Returns:
        tuple: (score, best_move_uci) if a valid cutoff is found.
               (None, best_move_uci) if the depth is insufficient for a cutoff, but a move exists for ordering.
               (None, None) if the position is not in the table.
    """
    entry = _tt.get(zobrist_key)

    if entry is not None:
        # If the cached search was at least as deep as our current requirement
        if entry['depth'] >= depth:
            score = entry['score']
            flag = entry['flag']

            # Check if the cached score can cause a pruning cutoff
            if flag == TT_EXACT:
                return score, entry['best_move']
            elif flag == TT_LOWERBOUND and score >= beta:
                return score, entry['best_move']
            elif flag == TT_UPPERBOUND and score <= alpha:
                return score, entry['best_move']

        # Even if the depth is too shallow for a hard cutoff,
        # the previously found best move is extremely valuable for move ordering.
        return None, entry['best_move']

    return None, None


def store(zobrist_key: int, depth: int, flag: int, score: int, best_move: str):
    """
    Saves a position's evaluation and best move to the transposition table.
    Enforces the strict memory limits defined in config.py to prevent OOM crashes.
    """
    global _tt

    # 1. Memory Safety Cap: Prevent exceeding the 2 GB container limit.
    if len(_tt) >= config.TT_MAX_ENTRIES:
        # A simple wipe is the safest approach in standard Python dictionaries
        # to guarantee memory is freed immediately.
        _tt.clear()

    existing = _tt.get(zobrist_key)

    # 2. Depth Replacement Scheme: Only overwrite an existing entry if the new
    # search goes at least as deep, ensuring we don't overwrite high-quality
    # data with shallow search results.
    if existing is None or depth >= existing['depth']:
        _tt[zobrist_key] = {
            'depth': depth,
            'flag': flag,
            'score': score,
            'best_move': best_move
        }


def clear():
    """Manually flushes the transposition table (useful for testing or resetting)."""
    _tt.clear()