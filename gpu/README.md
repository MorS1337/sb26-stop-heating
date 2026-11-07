# GPU (WSL2 ROCm) for fastsim on the RX 9060 XT

Status: **working**. PyTorch 2.9.1 + ROCm 7.2.0 (gfx1200) runs in WSL2 Ubuntu 24.04 on the
Radeon RX 9060 XT with the Windows driver that was already installed. No Windows-side change
was needed.

Host: a Windows 11 PC (Ryzen 5 5600X, Radeon RX 9060 XT), WSL2 `Ubuntu-24.04`, kernel 6.18.54.1-microsoft-standard-WSL2,
12 vCPU / 11.9 GiB RAM, `/dev/dxg` + `/usr/lib/wsl/lib/{libdxcore,libd3d12}.so` present.

## Versions actually installed

| Component | Version |
|---|---|
| Windows display driver | Adrenalin `32.0.32015.7029` (2026-10-18) — unchanged |
| ROCm userspace (`rocm-core`) | `7.2.0.70200-43~24.04` in `/opt/rocm-7.2.0`, `/opt/rocm` symlink (22 GB on disk) |
| WSL HSA runtime | `hsa-runtime-rocr4wsl-amdgpu 25.30.13-2281980.24.04` |
| hip-runtime-amd | `7.2.26015.70200-43~24.04` |
| rocblas / hipblaslt / miopen-hip / comgr | `5.2.0` / `1.2.1` / `3.5.1` / `3.0.0` (all `70200-43~24.04`) |
| PyTorch | `2.9.1+rocm7.2.0.lw.git7e1940d4` (AMD "lw" wheel, links the system `/opt/rocm`) |
| Python / venv | 3.12.3 in `/home/mors1337/gpu-venv` (created by `uv`), numpy 1.26.4 |

The simulator venv `~/sb26/starter-kit/.venv` was not touched.

## Driver check (the "do I need a new Adrenalin" question)

AMD's 7.2/7.2.1 WSL matrix requires **AMD Software: Adrenalin Edition 26.1.1 for WSL2**
(<https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/compatibility/compatibilityrad/wsl/wsl_compatibility.html>).
The machine already has a newer release (driver build `32.0.32015.7029`, dated 2026-10-18,
i.e. the 26.10 branch; Adrenalin 26.10.2 ships `32.0.32015.12012`). `rocminfo` enumerates the
GPU, so the installed driver is sufficient — **nothing to install on Windows**.

Docs / release notes read for this: Radeon-Ryzen "Use ROCm on Radeon and Ryzen" 7.2 (WSL
install + WSL PyTorch pages) and ROCm 7.14 unified install page. Note that Radeon-Ryzen docs
stop at 7.2.1 while `repo.radeon.com` already has 7.2.2/7.2.3/7.2.4 packages; 7.13/7.14 docs
have no `wsl` usecase at all (they want a hand-built ROCDXG), so 7.2 is the supported WSL path.

## Exact commands that worked (root in WSL2)

```bash
# 1. installer script for the 7.2 line (Ubuntu 24.04 = noble)
sudo apt-get update
sudo apt-get install -y wget ca-certificates gnupg2 python3-pip
cd /tmp
wget https://repo.radeon.com/amdgpu-install/7.2/ubuntu/noble/amdgpu-install_7.2.70200-1_all.deb
sudo apt-get install -y --allow-downgrades ./amdgpu-install_7.2.70200-1_all.deb

# 2. userspace only, no DKMS (WSL has no amdgpu kernel module)
sudo amdgpu-install -y --usecase=wsl,rocm --no-dkms
```

Two things bit us and are worth remembering:

1. **Only the 7.2 installer still has the `wsl` usecase.** `amdgpu-install` from
   7.2.1/7.2.2/7.2.3/7.2.4 (e.g. `amdgpu-install_7.2.4.70204-1_all.deb`) fails with
   `ERROR: Usecase implementation 'wsl' is not supported or invalid`. Use `7.2/…/amdgpu-install_7.2.70200-1_all.deb`.
   The `wsl` usecase is what pulls `hsa-runtime-rocr4wsl-amdgpu` (the ROCDXG-based HSA runtime).
2. **`repo.radeon.com` is unusable through plain `apt` from this ISP**: single-stream throughput
   is ~250 KB/s and connections hang/drop after ~90 MB, and multi-GB objects sometimes come back
   with the Range header ignored. `apt-get` / `curl -C -` then either crawl (~7 h for the 6.2 GB
   set) or grow the file past its expected size (corruption). Fix: download the debs in 16 MB
   chunks with 12 parallel connections (`gpu/fetch5.py`, ~8–12 MB/s, ~10 min for 6.2 GB), then let
   `amdgpu-install` install from `/var/cache/apt/archives`. Recipe:

```bash
sudo apt-get install --print-uris -y hsa-runtime-rocr4wsl-amdgpu rocm \
  | sed -n "s/^'\(http[^']*\)'.*/\1/p" > /tmp/miss.txt
sudo LIST=/tmp/miss.txt DEST=/var/cache/apt/archives python3 gpu/fetch5.py
sudo amdgpu-install -y --usecase=wsl,rocm --no-dkms    # now installs from cache
```

### How much actually has to be downloaded (and why it is slow)

Measured `.deb` archive sizes: `--usecase=wsl,rocm` = **6.09 GB** (22 GB installed), AMD torch
wheel = **1.66 GB** → **7.75 GB total**. Time breakdown on this ISP: ~25–30 min of download at
8–12 MB/s plus ~5 min of `dpkg`; the same bytes through plain `apt` would take ~7 h.

The full `rocm` metapackage is *not* avoidable in general, but it is oversized. `libtorch_hip.so`
needs system `librocblas.so.5`, `libhipblaslt.so.1`, `librocsolver`, `librocsparse`, `librocfft`,
`libhipblas`, `libamdhip64`, `libhsa-runtime64` — the "lw" wheel bundles **none** of them
(`torch/lib` contains zero `librocblas*`/`libhipblaslt*`/`libmiopen*`), so `--usecase=wsl` alone
(i.e. only `hsa-runtime-rocr4wsl-amdgpu`) is not enough. The minimum that works is a curated
runtime set of **3.05 GB**:

```bash
sudo apt-get install -y --no-install-recommends \
  hsa-runtime-rocr4wsl-amdgpu rocm-core hip-runtime-amd comgr rocblas hipblaslt rocsolver \
  rocsparse rocfft hiprand hipsolver hipsparse hipfft hipblas rocminfo rocm-smi-lib \
  rocprofiler-register rocm-device-libs hipcub rocprim
```
(hipblaslt 1.54 GB + rocsolver 0.79 GB + rocsparse 0.40 GB + rocblas 0.16 GB + rocfft 0.10 GB
dominate; the rest of the 6.09 GB is dev/compiler/multimedia: `rocm-llvm`,
`composablekernel-dev`, `migraphx`, `mivisionx`, `hipify-clang`, `rocm-gdb`, `rccl`.)
Saving ≈3 GB ≈ 3–4 min at the rates below — the CDN throttling, not the size, is the problem.

`download.pytorch.org/whl/rocm7.2` also exists (torch 2.13/2.14) but those wheels *bundle* the
ROCm libs, so the total download is the same or larger and the torch version differs. `aria2c`
would be equivalent to what `fetch5.py` (12 connections × 16 MB chunks) already achieves; it is
not installed and installing it costs another download.

### PyTorch (user, separate venv)

```bash
uv venv --python 3.12 /home/mors1337/gpu-venv
# 1.66 GB wheel, also fetched with the chunked downloader
echo 'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torch-2.9.1%2Brocm7.2.0.lw.git7e1940d4-cp312-cp312-linux_x86_64.whl' > /tmp/torch_uris.txt
LIST=/tmp/torch_uris.txt DEST=~/sb26/gpu/wheels python3 gpu/fetch5.py
uv pip install --python /home/mors1337/gpu-venv/bin/python numpy==1.26.4 filelock typing-extensions \
  sympy networkx jinja2 fsspec
uv pip install --python /home/mors1337/gpu-venv/bin/python --no-deps ~/sb26/gpu/wheels/torch-*.whl
# AMD WSL step: the wheel ships its own libhsa-runtime64.so — remove it so the
# rocr4wsl runtime from /opt/rocm is used
LIB=$(/home/mors1337/gpu-venv/bin/python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))')
rm -f "$LIB"/libhsa-runtime64.so*
```

No `LD_LIBRARY_PATH` or `HSA_OVERRIDE_GFX_VERSION` is needed (native gfx1200 target).

## Verification

```
$ ~/gpu-venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
True AMD Radeon RX 9060 XT
```
`torch.cuda.get_device_properties(0)` → `gcnArchName=gfx1200`, `total_memory=16253 MB` (15.87 GiB),
`multi_processor_count=16`, `L2_cache_size=4MB`, `pci_bus_id=8`.
`rocminfo` → `Agent 2: gfx1200, Marketing Name: AMD Radeon RX 9060 XT, Vendor: AMD,
Queue Max Size 131072, Cacheline 64, Compute Unit: 32` (torch reports 16 because gfx12 counts
workgroup processors). A 2048² fp32 matmul matches the CPU result.

Quirks (both harmless):
* `Warning: Resource leak detected by SharedSignalPool, 2 Signals leaked.` is printed at
  interpreter exit — ROCDXG teardown noise on WSL, not an error.
* `amd-smi` does not work (`amdgpu not found in modules`) because WSL2 has no kernel driver;
  use `rocminfo` / `/sys/class/drm` or `torch.cuda` for device info.

## Benchmark numbers (`gpu/bench_gpu.py`, gfx1200, 16 GB, fp32 unless noted)

Dense peak: 4096³ fp32 matmul **23.98 ms → 5.73 TFLOPS** (rocblas).

(a) ROM step `x⁺ = A x + B u`, shared `A (m×m)`, `u` dim 8, two fused `addmm`s:

| m | B=1 000 | B=10 000 | B=100 000 |
|---|---|---|---|
| 100 | 0.156 ms → **6.39 M env-steps/s** | 0.312 ms → 32.0 M env-steps/s | 2.38 ms → 42.0 M env-steps/s |
| 200 | 0.121 ms → 8.29 M env-steps/s | 0.656 ms → 15.2 M env-steps/s | 6.29 ms → 15.9 M env-steps/s |
| 400 | 0.240 ms → 4.16 M env-steps/s | 2.03 ms → 4.94 M env-steps/s | 15.3 ms → 6.52 M env-steps/s |

So a 1000-environment ROM batch costs ~0.12–0.24 ms per step: ~10⁴–10⁵× cheaper than the
40–130 ms/step of one real simulator episode step. At B=1000 the launch overhead dominates
(m=200 is faster than m=100 because of rocblas kernel selection), so prefer B ≥ 10 000 when
throughput matters.

(b) 2-D 5-point Jacobi sweep, 600×400 grid (= 240 k control volumes ≈ the real 123 k–263 k range),
batched over copies, float32:

| copies | per-sweep | env-sweeps/s | effective BW |
|---|---|---|---|
| 64 | 3.42 ms | 18 707 | 107.8 GB/s |
| 256 | 14.13 ms | 18 115 | 104.3 GB/s |

Layout sensitivity (`bench_align.py`, 128 copies): nx=600 → 108.1 GB/s, nx=608 → 107.0,
nx=640 → 104.8, transposed 400×600 → 107.4 GB/s: alignment/layout is irrelevant, the kernel is
limited by the strided 4-byte stencil traffic (≈1/3 of the 322 GB/s theoretical HBM).
**float16 doubles it**: 3.43 ms/sweep → 37 358 env-sweeps/s, 215.2 GB/s.
Practical: one 600×400 sweep of ONE env costs 53 µs; a CG solve of ~100 sweeps × 128 envs
= 0.68 s ⇒ **5.3 ms per env-step for 128 parallel envs** (~10–25× faster than the CPU simulator,
and only the zone means must be read back).

## Files

| file | purpose |
|---|---|
| `install_rocm_wsl.sh` | the ROCm 7.2 WSL install, as run (apt debs, `--usecase=wsl,rocm --no-dkms`) |
| `fetch5.py` | chunked 16 MB / 12-way resumable downloader for `repo.radeon.com` (works around CDN throttling) |
| `setup_venv.sh` | `uv` venv + AMD torch wheel + `libhsa-runtime64.so` removal |
| `verify.sh` | rocminfo + torch checks |
| `bench_gpu.py`, `bench_align.py` | the benchmarks above (`bench_gpu.json`, `bench_align.json`) |
| `run_bench.sh` | runs verify + bench with the venv python |

Not installed on purpose: nothing put in `PATH` for the simulator; use
`/home/mors1337/gpu-venv/bin/python` explicitly.
