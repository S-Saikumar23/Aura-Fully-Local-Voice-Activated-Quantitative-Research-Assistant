"""
AURA — Pytest Configuration.

Adds the project root to sys.path so that all modules
can be imported correctly during test execution.
"""

import sys
from pathlib import Path

# Add project root to sys.path for test imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
