#!/bin/bash
# ROCm WSL userspace for Radeon RX 9060 XT (gfx1200) on Ubuntu 24.04 WSL2.
# Docs: https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/install/installrad/wsl/install-radeon.html
# NOTE: only the 7.2 (70200) amdgpu-install still ships the 'wsl' usecase;
#       7.2.4 (70204) dropped it ("Usecase implementation 'wsl' is not supported").
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
echo "### base packages"
apt-get update
apt-get install -y wget ca-certificates gnupg2 python3-pip
echo "### amdgpu-install 7.2 deb"
cd /tmp
wget -q -O amdgpu-install-72.deb https://repo.radeon.com/amdgpu-install/7.2/ubuntu/noble/amdgpu-install_7.2.70200-1_all.deb
apt-get install -y --allow-downgrades ./amdgpu-install-72.deb
echo "### usecases"
amdgpu-install --list-usecase | sed -n '1,40p' || true
echo "### install wsl,rocm (no dkms)"
amdgpu-install -y --usecase=wsl,rocm --no-dkms
echo "### installed"
dpkg -l | grep -E "rocm|hsa|hip" | awk '{print $2, $3}' | head -40
ls -d /opt/rocm*
echo "### rocminfo"
/opt/rocm/bin/rocminfo > /home/mors1337/sb26/gpu/rocminfo.txt 2>&1 || echo "rocminfo rc=$?"
grep -E "Name|Marketing|gfx|Uuid|Compute Unit|Total Global" /home/mors1337/sb26/gpu/rocminfo.txt | head -20
echo "### amd-smi"
/opt/rocm/bin/amd-smi static 2>&1 | head -30 || true
