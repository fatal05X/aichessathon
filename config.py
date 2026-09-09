# ==========================================
# MEMORY & TRANSPOSITION TABLE
# ==========================================
# The tournament hardware provides a hard limit of 2 GB of RAM.
# We restrict the TT to 2,000,000 entries to stay well within limits (~150-200 MB).
TT_MAX_ENTRIES = 2_000_000

# ==========================================
# TIME MANAGEMENT
# ==========================================
# Time control is 120s per side with a 0.5s increment per move.
# Time budgeting formula: allocated_ms = time_left_ms / max(TIME_MIN_DIVISOR, expected_moves_left)
TIME_MIN_DIVISOR = 20

# The initial setup budget before the clock starts is 90 seconds.
# We use 85 seconds as a safe margin to finish Numba JIT compilation and loading.
INIT_TIME_BUDGET_MS = 85_000

# ==========================================
# SEARCH BOUNDS & LIMITS
# ==========================================
MAX_DEPTH = 64              # Max absolute depth for iterative deepening
MAX_PLY = 128               # Max ply limit for internal search arrays (e.g., killer moves)

# Evaluation bounds (Centipawns)
INFINITY = 32000
MATE_SCORE = 30000
MATE_BOUND = 29000
DRAW_SCORE = 0

# Penalty applied to candidate moves leading to repetitions if evaluation is positive.
CONTEMPT_FACTOR = -50

# ==========================================
# RULES & ADJUDICATION
# ==========================================
# Games are strictly drawn at 600 plies (including the starting FEN).
# There is NO 300-ply material adjudication.
MAX_GAME_PLIES = 600