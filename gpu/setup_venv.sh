#!/bin/bash
# Create /home/mors1337/gpu-venv (Python 3.12 via uv) and install the AMD PyTorch
# ROCm wheel from repo.radeon.com.
# Docs: https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/install/installrad/wsl/install-pytorch.html
set -euo pipefail
UV=/home/mors1337/.local/bin/uv
VENV=/home/mors1337/gpu-venv
WHL_URL='https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torch-2.9.1%2Brocm7.2.0.lw.git7e1940d4-cp312-cp312-linux_x86_64.whl'
WHEEL_DIR=/home/mors1337/sb26/gpu/wheels
mkdir -p "$WHEEL_DIR"

echo "### wheel: $(curl -sI --max-time 30 "$WHL_URL" | awk 'tolower($1)=="content-length:"{print $2}' | tr -d '\r') bytes"
echo "$WHL_URL" > /tmp/torch_uris.txt
LIST=/tmp/torch_uris.txt DEST="$WHEEL_DIR" python3 /home/mors1337/sb26/gpu/fetch5.py
ls -lh "$WHEEL_DIR"/*.whl

echo "### venv"
$UV venv --python 3.12 "$VENV"
$UV pip install --python "$VENV/bin/python" "numpy==1.26.4" filelock typing-extensions sympy networkx jinja2 fsspec
$UV pip install --python "$VENV/bin/python" --no-deps "$WHEEL_DIR"/torch-*.whl

echo "### drop bundled hsa runtime so WSL uses the system rocr4wsl one"
LIB=$("$VENV/bin/python" -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))')
rm -f "$LIB"/libhsa-runtime64.so*
echo "### versions"
"$VENV/bin/python" -c 'import torch,numpy;print("torch",torch.__version__,"hip",torch.version.hip,"numpy",numpy.__version__)'
