import sys
from pathlib import Path

# Ensure backend/ is on the import path
sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

from app import app

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
