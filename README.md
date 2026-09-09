# AI Chessathon Custom Agent

A high-performance classical search chess agent built for the **AI Chessathon**. 

The agent runs strictly within the tournament's single-core CPU container limits, utilizing an iterative deepening Alpha-Beta (Negamax) search, tapered PeSTO piece-square evaluation JIT-compiled with Numba, an in-memory Transposition Table, and dynamic time management.

---

## Technical Architecture

### 1. Search Pipeline (`search.py`)
* **Core Algorithm:** Iterative Deepening Negamax search with Alpha-Beta pruning.
* **Tactical Extension:** Quiescence search evaluating captures and promotions to prevent the horizon effect.
* **Move Ordering:**
  * Transposition Table best move prioritized first.
  * MVV-LVA (Most Valuable Victim – Least Valuable Attacker) for captures.
  * Killer move heuristic (2 slots per ply) and history table for quiet moves.
* **Time Check:** Wall-clock polling every 1024 nodes via bitmask to exit searches cleanly without flagging.

### 2. Fast Evaluation (`eval.py`)
* **Tapered Heuristics:** Piece-square tables and material evaluations smoothly interpolated between midgame and endgame phases.
* **Numba JIT Acceleration:** Static evaluation compiled into machine code via `@numba.njit(cache=False)` to maximize nodes/sec on a single CPU core.
* **Contempt & Draw Awareness:** Penalizes threefold repetitions and near-cap states when holding a winning advantage.

### 3. Memory & Match State (`tt.py`, `state.py`, `config.py`)
* **Transposition Table:** In-memory Zobrist hash table capped at 2,000,000 entries (~150–200 MB) to stay safely inside the 2 GB memory ceiling.
* **Game History:** Persists across moves in container memory to track half-move counts and repeated positions from the initial curated FEN up to the 600-ply draw cap.
* **Dynamic Time Budgeting:** Allocates time per move based on `time_left_ms / max(20, expected_moves_remaining)`, adapting gracefully during low-clock scrambles.

---

## Repository & Submission Structure

The platform mandates a completely flat file layout inside `submission.zip` (max 50 MB unzipped):

```text
submission.zip
├── agent.py         # Required entrypoint exposing get_move(fen, time_left_ms)
├── config.py        # Constants (600 max plies, 2M TT entries, dynamic timing)
├── state.py         # Match history, repetition tracking, and game counters
├── tt.py            # Memory-safe Transposition Table
├── eval.py          # Numba-jitted PeSTO evaluation and warmup routines
├── search.py        # Iterative deepening, Negamax, Quiescence, and move ordering
└── book.bin         # Optional Polyglot opening book (loaded if present)