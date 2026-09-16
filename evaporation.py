"""Terminal entry point for the independent evaporator safe-SAC experiment.

Run from a terminal with:

    python train_evaporation_safe_sac.py

The default is three seeds (including 42), 300 episodes and 2000 steps.

    python train_evaporation_safe_sac.py --episodes 300 --steps 2000
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from multiseed import main


if __name__ == "__main__":
    main()
