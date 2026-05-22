#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found. Install Miniconda/Anaconda and create SolarChain-rl first." >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate SolarChain-rl

python -m pip install -r requirements.txt
python -m pip install -e .

python - <<'PY'
import sys
import gymnasium
import openai
import pydantic
import pvlib
import stable_baselines3

print("Python:", sys.executable)
print("Gymnasium:", gymnasium.__version__)
print("OpenAI:", openai.__version__)
print("Pydantic:", pydantic.__version__)
print("pvlib:", pvlib.__version__)
print("Stable-Baselines3:", stable_baselines3.__version__)
PY

echo "SolarChain-Eval Linux setup complete."

