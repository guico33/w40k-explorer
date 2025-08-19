import sys
from pathlib import Path


def pytest_sessionstart(session):
    # Ensure the src/ directory is on sys.path so `import w40k` works
    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

