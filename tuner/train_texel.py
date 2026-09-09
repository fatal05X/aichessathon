import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

MAX_PHASE = 24


def build_initial_pesto():
    """Builds initial baseline PeSTO weights for PyTorch initialization."""
    mg_val = [0, 82, 337, 365, 477, 1025, 0]
    eg_val = [0, 94, 281, 297, 512, 936, 0]

    center_bonus = np.array([
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10,   5,   5,   5,   5,   5, -10, -10,
        -10,   5,  10,  15,  15,  10,   5, -10,
        -10,   5,  15,  20,  20,  15,   5, -10,
        -10,   5,  15,  20,  20,  15,   5, -10,
        -10,   5,  10,  15,  15,  10,   5, -10,
        -10,   5,   5,   5,   5,   5, -10, -10,
        -20, -10, -10, -10, -10, -10, -10, -20
    ], dtype=np.float32)

    pawn_mg = np.array([0, 5, 10, 20, 30, 50, 70, 0], dtype=np.float32)
    pawn_eg = np.array([0, 10, 20, 30, 50, 70, 90, 0], dtype=np.float32)

    mg_init = np.zeros((7, 64), dtype=np.float32)
    eg_init = np.zeros((7, 64), dtype=np.float32)

    for sq in range(64):
        rank = sq // 8
        mg_init[1, sq] = mg_val[1] + pawn_mg[rank] + center_bonus[sq] / 2.0
        eg_init[1, sq] = eg_val[1] + pawn_eg[rank]
        for p in range(2, 6):
            mg_init[p, sq] = mg_val[p] + center_bonus[sq]
            eg_init[p, sq] = eg_val[p] + center_bonus[sq] / 2.0
        mg_init[6, sq] = -center_bonus[sq]
        eg_init[6, sq] = center_bonus[sq]

    return mg_init, eg_init


class TexelModel(nn.Module):
    """Linear evaluation model with tapered game phase evaluation."""
    def __init__(self, init_mg, init_eg):
        super().__init__()
        # 7 piece types (0=empty, 1..6 = P, N, B, R, Q, K) x 64 squares
        self.mg_weights = nn.Parameter(torch.tensor(init_mg, dtype=torch.float32))
        self.eg_weights = nn.Parameter(torch.tensor(init_eg, dtype=torch.float32))

        # Flip table mapping for Black piece symmetry (sq ^ 56)
        flip_indices = [sq ^ 56 for sq in range(64)]
        self.register_buffer("flip_indices", torch.tensor(flip_indices, dtype=torch.long))

    def forward(self, boards: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        """
        Calculates position evaluation in centipawns and converts to win probability.
        boards: (batch_size, 64) with values 0..12
        phases: (batch_size,) with values 0..24
        """
        batch_size = boards.size(0)

        # Separate masks for White (1..6) and Black (7..12)
        white_mask = (boards >= 1) & (boards <= 6)
        black_mask = (boards >= 7) & (boards <= 12)

        white_pieces = torch.where(white_mask, boards, 0)
        black_pieces = torch.where(black_mask, boards - 6, 0)

        squares = torch.arange(64, device=boards.device).unsqueeze(0).expand(batch_size, -1)
        flipped_squares = self.flip_indices[squares]

        # Look up Midgame values
        mg_white = self.mg_weights[white_pieces, squares] * white_mask.float()
        mg_black = self.mg_weights[black_pieces, flipped_squares] * black_mask.float()
        mg_score = (mg_white - mg_black).sum(dim=1)

        # Look up Endgame values
        eg_white = self.eg_weights[white_pieces, squares] * white_mask.float()
        eg_black = self.eg_weights[black_pieces, flipped_squares] * black_mask.float()
        eg_score = (eg_white - eg_black).sum(dim=1)

        # Taper evaluation based on phase
        tapered_eval = (mg_score * (MAX_PHASE - phases) + eg_score * phases) / float(MAX_PHASE)

        # Sigmoid win probability: P(Win) = 1 / (1 + 10^(-eval / 400))
        # Mathematically equivalent to torch.sigmoid(eval * (ln(10) / 400))
        scaling_factor = 0.00575646273  # ln(10) / 400
        win_prob = torch.sigmoid(tapered_eval * scaling_factor)
        return win_prob


def train():
    dataset_path = os.path.join(os.path.dirname(__file__), "dataset.npz")
    if not os.path.exists(dataset_path):
        print(f"Error: dataset.npz not found at {dataset_path}")
        print("Run `python tuner/extract_data.py <path_to_pgn>` first.")
        sys.exit(1)

    print(f"Loading {dataset_path}...")
    data = np.load(dataset_path)
    boards = torch.tensor(data["boards"], dtype=torch.long)
    phases = torch.tensor(data["phases"], dtype=torch.float32)
    targets = torch.tensor(data["targets"], dtype=torch.float32)

    dataset = TensorDataset(boards, phases, targets)
    loader = DataLoader(dataset, batch_size=4096, shuffle=True)

    init_mg, init_eg = build_initial_pesto()
    model = TexelModel(init_mg, init_eg)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    criterion = nn.MSELoss()

    print(f"Beginning training on {len(dataset):,} positions over 15 epochs...")

    for epoch in range(1, 16):
        total_loss = 0.0
        for b_boards, b_phases, b_targets in loader:
            optimizer.zero_grad()
            preds = model(b_boards, b_phases)
            loss = criterion(preds, b_targets)
            loss.backward()

            # Fix empty piece slot to strictly zero
            with torch.no_grad():
                model.mg_weights[0].zero_()
                model.eg_weights[0].zero_()

            optimizer.step()
            total_loss += loss.item() * len(b_targets)

        epoch_loss = total_loss / len(dataset)
        scheduler.step(epoch_loss)
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:02d} | MSE Loss: {epoch_loss:.6f} | LR: {lr:.4f}")

    # Export final weights matrix formatted as (2, 13, 64)
    print("Formatting and exporting trained weights...")
    final_mg = model.mg_weights.detach().cpu().numpy()
    final_eg = model.eg_weights.detach().cpu().numpy()

    weights_matrix = np.zeros((2, 13, 64), dtype=np.int32)

    for sq in range(64):
        flip_sq = sq ^ 56
        for p in range(1, 7):
            # White pieces (1..6)
            mg_val = int(round(final_mg[p, sq]))
            eg_val = int(round(final_eg[p, sq]))
            weights_matrix[0, p, sq] = mg_val
            weights_matrix[1, p, sq] = eg_val

            # Black pieces (7..12) - Mirrored and negated
            weights_matrix[0, p + 6, flip_sq] = -mg_val
            weights_matrix[1, p + 6, flip_sq] = -eg_val

    # Place directly into root submission directory
    out_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "weights.npy"))
    np.save(out_root_path, weights_matrix)
    print(f"Successfully exported tuned weights to: {out_root_path}")


if __name__ == "__main__":
    train()