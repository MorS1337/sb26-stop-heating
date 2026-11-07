#!/usr/bin/env python3
"""Stencil layout/alignment sensitivity on the 600x400 grid (fastsim design input)."""
import time, torch, json

dev = torch.device("cuda:0")


def timed(fn, warmup=5, min_time=0.4):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
    one = max(time.perf_counter() - t0, 1e-6)
    n = max(5, min(300, int(min_time / one)))
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def sweep_bench(C, nx, ny, dtype=torch.float32):
    a = torch.randn(C, ny, nx, device=dev, dtype=dtype)
    b = torch.empty_like(a)
    rhs = torch.randn(C, ny, nx, device=dev, dtype=dtype)

    def sweep():
        b[:, 1:-1, 1:-1] = 0.2 * (a[:, :-2, 1:-1] + a[:, 2:, 1:-1]
                                  + a[:, 1:-1, :-2] + a[:, 1:-1, 2:] - rhs[:, 1:-1, 1:-1])

    t = timed(sweep)
    cells = C * nx * ny
    gb = cells * 4 * 6 / t / 1e9
    return dict(C=C, nx=nx, ny=ny, dtype=str(dtype).split(".")[-1],
                ms=round(t * 1e3, 3), env_sweeps_per_s=int(C / t), gbps=round(gb, 1))


out = []
for shape in [(600, 400), (640, 400), (608, 400), (400, 600)]:
    out.append(sweep_bench(128, *shape))
    print(json.dumps(out[-1]), flush=True)
out.append(sweep_bench(128, 600, 400, torch.float16))
print(json.dumps(out[-1]), flush=True)
json.dump(out, open("bench_align.json", "w"), indent=1)
