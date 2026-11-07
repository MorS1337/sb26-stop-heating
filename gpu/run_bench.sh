#!/bin/bash
# Verify + benchmark the GPU. Run as user mors1337 (not root).
set -u
cd /home/mors1337/sb26/gpu
PY=/home/mors1337/gpu-venv/bin/python
export PATH=/opt/rocm/bin:$PATH
echo "########## verify"
bash verify.sh 2>&1 | tee verify.log
echo "########## bench"
timeout 600 $PY bench_gpu.py --out bench_gpu.json 2>&1 | tee bench.log
echo "########## amd-smi"
amd-smi static 2>&1 | head -30 || true
