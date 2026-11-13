import sys
from pathlib import Path

# make tests/synth.py importable as `synth`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
