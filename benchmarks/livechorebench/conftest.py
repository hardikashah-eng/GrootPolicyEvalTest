"""Make the benchmark package importable when running ``pytest benchmarks/livechorebench``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
