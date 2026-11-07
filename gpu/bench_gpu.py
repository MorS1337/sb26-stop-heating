#!/usr/bin/env python3
"""Benchmark the GPU for the fastsim surrogate.

(a) ROM step: x+ = A x + B u, batched over environments:
    X (B,m) @ A^T (m,m) + U (B,p) @ B^T (p,m)   -> float32, shared A/B.
(b) 2-D 5-point Jacobi sweep on a 600x400 grid, batched over C independent copies
    (memory-bandwidth / strided-access check).
Timing uses cuda synchronisation + perf_counter (HIP event timing on RDNA4
sometimes returns negative intervals).
"""
import argparse, json, time
import torch


def timed(fn, warmup=5, min_time=0.5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    one = max(time.perf_counter() - t0, 1e-6)
    n = max(5, min(500, int(min_time / one)))
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def rom_bench(dev, batches, ms, p=8, verbose=True):
    out = []
    for m in ms:
        A = torch.randn(m, m, device=dev) / m ** 0.5
        AT = A.t().contiguous()
        Bm = torch.randn(p, m, device=dev) / p ** 0.5      # B^T: (p,m)
        for B in batches:
            X = torch.randn(B, m, device=dev)
            U = torch.randn(B, p, device=dev)
            Y = torch.empty_like(X)

            def step():
                torch.addmm(Y, X, AT, out=Y)    # Y  = X @ A^T
                torch.addmm(Y, U, Bm, out=Y)    # Y += U @ B^T

            t = timed(step)
            env_s = B / t
            gb = (B * m * 2 + m * m + B * p + p * m) * 4 / 1e9 / t
            row = dict(kind="rom_step", B=B, m=m, ms_per_step=round(t * 1e3, 4),
                       env_steps_per_s=int(env_s), kenv_steps_per_s=round(env_s / 1e3, 1),
                       gbps=round(gb, 1))
            out.append(row)
            if verbose:
                print(json.dumps(row), flush=True)
            del X, U, Y
            torch.cuda.empty_cache()
        del A, AT, Bm
        torch.cuda.empty_cache()
    return out


def stencil_bench(dev, copies, nx=600, ny=400, sweeps=30, verbose=True):
    out = []
    for C in copies:
        a = torch.randn(C, ny, nx, device=dev)
        b = torch.empty_like(a)
        rhs = torch.randn(C, ny, nx, device=dev)  # holds -rhs + source terms

        def sweep():
            b[:, 1:-1, 1:-1] = 0.2 * (a[:, :-2, 1:-1] + a[:, 2:, 1:-1]
                                      + a[:, 1:-1, :-2] + a[:, 1:-1, 2:] - rhs[:, 1:-1, 1:-1])

        t = timed(sweep, warmup=3, min_time=0.3)
        cells = C * nx * ny
        nbytes = cells * 4 * 6
        row = dict(kind="stencil5p", copies=C, grid=f"{nx}x{ny}", cells=cells,
                   ms_per_sweep=round(t * 1e3, 3), env_steps_per_s=int(C / t),
                   msweeps_per_s=round(C / t / 1e3, 1),
                   gcells_per_s=round(cells / t / 1e9, 2), gbps=round(nbytes / 1e9 / t, 1))
        out.append(row)
        if verbose:
            print(json.dumps(row), flush=True)
        del a, b, rhs
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench_gpu.json")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    print("torch", torch.__version__, "hip", torch.version.hip,
          "cuda_available", torch.cuda.is_available(), flush=True)
    assert torch.cuda.is_available(), "no GPU"
    dev = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)
    print(f"device: {props.name}  mem {props.total_memory/2**30:.2f} GiB  "
          f"cu {props.multi_processor_count}  arch {props.gcnArchName}", flush=True)

    batches = [1000, 10000] if args.quick else [1000, 10000, 100000]
    ms = [100, 200] if args.quick else [100, 200, 400]
    copies = [64, 128] if args.quick else [64, 256]

    a = torch.randn(4096, 4096, device=dev)
    b = torch.randn(4096, 4096, device=dev)
    t = timed(lambda: a @ b, warmup=3)
    tflops = 2 * 4096 ** 3 / t / 1e12
    print(f"fp32 4096^3 matmul: {t*1e3:.2f} ms -> {tflops:.2f} TFLOPS", flush=True)
    del a, b
    torch.cuda.empty_cache()

    res = dict(torch=torch.__version__, hip=torch.version.hip, device=props.name,
               total_mem_gib=round(props.total_memory / 2 ** 30, 2),
               cu=props.multi_processor_count, arch=props.gcnArchName,
               fp32_tflops=round(tflops, 2))
    res["rom"] = rom_bench(dev, batches, ms)
    res["stencil"] = stencil_bench(dev, copies)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
