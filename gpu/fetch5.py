#!/usr/bin/env python3
"""Chunked, retry-safe downloader for the ROCm debs (repo.radeon.com drops/hangs
connections after ~90 MB, and sometimes ignores Range for multi-GB objects).
Each 16 MB range is fetched into its own part file, so no append corruption is
possible and any failed chunk is simply retried."""
import concurrent.futures as futures
import os, sys, time, urllib.parse, urllib.request

DEST = os.environ.get("DEST", "/var/cache/apt/archives")
PARTS = os.path.join(DEST, ".parts")
CHUNK = 16 << 20
THREADS = 12
LIST = os.environ.get("LIST", "/tmp/miss.txt")
UA = {"User-Agent": "curl/8"}


def size_of(url):
    req = urllib.request.Request(url, headers=UA, method="HEAD")
    with urllib.request.urlopen(req, timeout=40) as r:
        return int(r.headers["Content-Length"])


def fetch_chunk(job):
    url, fname, start, end, tries = job
    part = os.path.join(PARTS, f"{fname}.{start}")
    want = end - start + 1
    for attempt in range(tries):
        if os.path.exists(part) and os.path.getsize(part) == want:
            return (fname, start, "ok")
        req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={start}-{end}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if r.status != 206:
                    time.sleep(1); continue
                data = r.read()
            if len(data) != want:
                time.sleep(1); continue
            with open(part, "wb") as f:
                f.write(data)
            return (fname, start, "ok")
        except Exception:
            time.sleep(1.5)
    return (fname, start, "fail")


def main():
    uris = [l.strip() for l in open(LIST) if l.strip().startswith("http")]
    os.makedirs(PARTS, exist_ok=True)
    jobs, meta = [], {}
    for u in uris:
        fn = urllib.parse.unquote(os.path.basename(u))
        dest = os.path.join(DEST, fn)
        try:
            sz = size_of(u)
        except Exception as e:
            print("HEAD fail", fn, e, flush=True); continue
        meta[fn] = (u, sz)
        have = os.path.getsize(dest) if os.path.exists(dest) else 0
        if have == sz:
            print("have", fn, flush=True); continue
        if os.path.exists(dest):
            os.remove(dest)          # never trust a partial file; chunks rebuild it
        n = (sz + CHUNK - 1) // CHUNK
        for i in range(n):
            s = i * CHUNK
            jobs.append((u, fn, s, min(s + CHUNK, sz) - 1, 40))
    print(f"files={len(meta)} chunks={len(jobs)}", flush=True)

    done = 0
    with futures.ThreadPoolExecutor(THREADS) as ex:
        for res in ex.map(fetch_chunk, jobs):
            done += 1
            if res[2] != "ok":
                print("chunk fail", res, flush=True)
            elif done % 25 == 0:
                print(f"{done}/{len(jobs)} chunks", flush=True)

    ok = 0
    for fn, (u, sz) in meta.items():
        if os.path.exists(os.path.join(DEST, fn)) and os.path.getsize(os.path.join(DEST, fn)) == sz:
            ok += 1; continue
        parts = sorted(int(p.rsplit(".", 1)[1]) for p in os.listdir(PARTS) if p.startswith(fn + "."))
        n = (sz + CHUNK - 1) // CHUNK
        if len(parts) != n:
            print("INCOMPLETE", fn, f"{len(parts)}/{n} chunks", flush=True); continue
        with open(os.path.join(DEST, fn), "wb") as out:
            for s in parts:
                with open(os.path.join(PARTS, f"{fn}.{s}"), "rb") as f:
                    while True:
                        b = f.read(1 << 22)
                        if not b:
                            break
                        out.write(b)
        got = os.path.getsize(os.path.join(DEST, fn))
        print(("assembled " if got == sz else "SIZE-MISMATCH ") + fn, got, sz, flush=True)
        if got == sz:
            ok += 1
            for s in parts:
                os.remove(os.path.join(PARTS, f"{fn}.{s}"))
    print(f"files complete: {ok}/{len(meta)}")


if __name__ == "__main__":
    main()
