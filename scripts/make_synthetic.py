"""Write small synthetic files in C-MAPSS format for tests and smoke runs.
NOT real data - only used to check the pipeline runs end to end."""
import sys
from pathlib import Path

import numpy as np


def make(out_dir, subset="FD001", n_train=30, n_test=10, seed=0, n_regimes=1):
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    settings = rng.uniform(0, 1, size=(n_regimes, 3))

    def engine(uid, life, truncate=None):
        rows = []
        reg = rng.integers(n_regimes)
        for c in range(1, life + 1):
            deg = (c / life) ** 2
            sens = [10 + i + (i % 3 + 1) * deg * (1 if i % 2 else -1) + rng.normal(0, 0.1) for i in range(21)]
            sens[0] = 5.0  # constant sensor
            rows.append([uid, c, *settings[reg], *sens])
        return rows[:truncate] if truncate else rows

    train = [r for u in range(1, n_train + 1) for r in engine(u, int(rng.integers(120, 220)))]
    test, ruls = [], []
    for u in range(1, n_test + 1):
        life = int(rng.integers(120, 220)); cut = int(rng.integers(60, life - 5))
        test += engine(u, life, cut); ruls.append(life - cut)
    fmt = lambda rows: "\n".join(" ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) for r in rows)
    (out / f"train_{subset}.txt").write_text(fmt(train))
    (out / f"test_{subset}.txt").write_text(fmt(test))
    (out / f"RUL_{subset}.txt").write_text("\n".join(map(str, ruls)))


if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else "data/synthetic")
