#!/bin/bash
# Verification of the torch+ROCm GPU in WSL2.
set -u
PY=/home/mors1337/gpu-venv/bin/python
echo "=== /dev/dxg"; ls -l /dev/dxg
echo "=== rocminfo"; /opt/rocm/bin/rocminfo 2>&1 | grep -E "^Agent|Name:|Marketing Name|Uuid|Compute Unit|Segment|gfx|Size:|Pool Info" | head -40
echo "=== amd-smi"; (/opt/rocm/bin/amd-smi static 2>&1 | head -40) || echo "amd-smi unavailable"
echo "=== torch"; $PY - <<'EOF'
import torch, sys
print("torch", torch.__version__, "hip", torch.version.hip, "py", sys.version.split()[0])
print("cuda.is_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("name:", torch.cuda.get_device_name(0))
    print("props:", p)
    print("mem GiB:", round(p.total_memory/2**30, 2), "cu:", p.multi_processor_count, "arch:", p.gcnArchName)
    a = torch.randn(2048, 2048, device="cuda")
    b = torch.randn(2048, 2048, device="cuda")
    c = a @ b
    torch.cuda.synchronize()
    print("matmul ok:", c.shape, float(c.sum()), "allclose:", bool(torch.allclose(c, a @ b, atol=1e-2)))
    x = torch.zeros(1, device="cuda"); x += 1; print("alloc ok", float(x))
EOF
