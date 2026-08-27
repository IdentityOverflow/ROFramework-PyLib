"""
Weight Checkpoint Reproduction — Modular Addition & Grokking

Re-runs the exact Phase 8a training (08_knowledge_tracker.py: same seeds,
same architecture, same hyperparameters) but saves model state_dicts at
milestone epochs so the network's weights can be inspected before, during,
and after grokking.

Saved artifacts (in experiments/grokking/checkpoints/, gitignored):
    ckpt_init.pt        — untrained network (epoch 0, before any step)
    ckpt_e{N}.pt        — after N training steps, at milestone epochs
    history.json        — train/test accuracy at every eval interval

Requires: PyTorch
Runtime:  ~30s on GPU (p=97, 7500 epochs)
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CKPT_DIR = Path(__file__).parent / "checkpoints"

# Milestones: init, memorization onset (~250), features-strong (~500),
# pre-grok plateau, grok transition (~4000), post-grok cleanup, final.
MILESTONES = [250, 500, 1000, 2000, 3000, 3500, 4000, 4500, 5000, 6000, 7499]


class ModularAdditionMLP(nn.Module):
    """Identical to 08_knowledge_tracker.py."""

    def __init__(self, p: int, embed_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.p = p
        self.hidden_dim = hidden_dim
        self.embed_a = nn.Embedding(p, embed_dim)
        self.embed_b = nn.Embedding(p, embed_dim)
        self.fc1 = nn.Linear(2 * embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, p)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(torch.cat([self.embed_a(a), self.embed_b(b)], -1))))


def make_dataset(p: int, train_frac: float = 0.5, seed: int = 42):
    """Identical to 08_knowledge_tracker.py."""
    rng = np.random.default_rng(seed)
    all_pairs = [(a, b) for a in range(p) for b in range(p)]
    rng.shuffle(all_pairs)

    split = int(len(all_pairs) * train_frac)
    train = all_pairs[:split]
    test = all_pairs[split:]

    def to_tensors(pairs):
        a = torch.tensor([pair[0] for pair in pairs], dtype=torch.long, device=DEVICE)
        b = torch.tensor([pair[1] for pair in pairs], dtype=torch.long, device=DEVICE)
        y = (a + b) % p
        return a, b, y

    return to_tensors(train), to_tensors(test)


def main(
    p: int = 97,
    embed_dim: int = 128,
    hidden_dim: int = 128,
    train_frac: float = 0.5,
    lr: float = 1e-3,
    weight_decay: float = 1.0,
    num_epochs: int = 7500,
    eval_interval: int = 50,
) -> None:
    CKPT_DIR.mkdir(exist_ok=True)

    torch.manual_seed(42)
    model = ModularAdditionMLP(p, embed_dim, hidden_dim).to(DEVICE)
    (train_a, train_b, train_y), (test_a, test_b, test_y) = make_dataset(p, train_frac)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    torch.save(model.state_dict(), CKPT_DIR / "ckpt_init.pt")
    print(f"Saved init checkpoint. Device: {DEVICE}")

    history = []
    milestones = set(MILESTONES)

    for epoch in range(num_epochs):
        model.train()
        loss = criterion(model(train_a, train_b), train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % eval_interval == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                train_acc = (model(train_a, train_b).argmax(1) == train_y).float().mean().item()
                test_acc = (model(test_a, test_b).argmax(1) == test_y).float().mean().item()
            history.append({"epoch": epoch, "train_acc": train_acc,
                            "test_acc": test_acc, "loss": loss.item()})
            if epoch % 500 == 0 or epoch == num_epochs - 1:
                print(f"epoch {epoch:>5} | train {train_acc:5.0%} | test {test_acc:5.0%}")

        if epoch in milestones:
            torch.save(model.state_dict(), CKPT_DIR / f"ckpt_e{epoch}.pt")

    with open(CKPT_DIR / "history.json", "w") as f:
        json.dump(history, f)

    grok = next((h["epoch"] for h in history if h["test_acc"] > 0.95), None)
    print(f"\nGrokking (test > 95%) at epoch: {grok}")
    print(f"Checkpoints in {CKPT_DIR}")


if __name__ == "__main__":
    main()
