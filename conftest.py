"""Make `pulselab` importable when running tests from the repo root.

The package lives at the repo root (it has __init__.py); we add the parent of
this file to sys.path so `import pulselab.x.y` resolves.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
