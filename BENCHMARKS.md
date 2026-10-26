# Benchmarks

> **No results below are measured unless a row says so explicitly. Every
> unmeasured cell is TBD.** Numbers are only added from runs whose raw
> outputs are committed under `bench/results/<date>-<host>/` (git-ignored by
> default; force-add the summary).

## Methodology

### Budget enforcement (cgroup v2, no root)
The engine **and** the daemon run in one transient scope with a hard memory
wall:

```
systemd-run --user --scope -p MemoryMax=<C> -p MemorySwapMax=<S> -- <cmd>
```

(`bench/cgroup_run.sh` wraps this.)
- Page-cache pages are charged to the cgroup of the task that first reads
  them. Running the daemon outside the scope would let it smuggle memory
  past the budget (IDEA_REVIEW §1.4).
- A cgroup wall is used instead of a memory balloon. The balloon
  methodology inflates MGLRU traffic about 2× (arXiv 2608.12103), so
  results with a balloon are not comparable.
- `MemorySwapMax=0` for the page-cache configurations. For the default
  (repack) baseline, swap/zram is allowed and recorded, because that is how
  the default behaves on a real laptop.

### Cold start
Before each run, evict the model file from the page cache with
`moepager fault-io --drop <file>`. That calls `posix_fadvise(DONTNEED)` on
an unmapped fd, which works without root for any file (verified on btrfs,
kernel 6.19). There is a 30 s idle period before the run.

### Workload
- `llama-bench -p 0 -n 128 -r 3` for decode throughput.
- `llama-cli` with fixed prompts from 3 domains (chat, code, prose), 256
  generated tokens, `--temp 0`, a fixed seed, and `-t <physical cores>`.

### Metrics

| metric | source |
|---|---|
| tokens/s (decode) | llama.cpp timings line (`eval time … tokens per second`) or llama-bench `tg` |
| SSD bytes read / token | Δ field 3 (sectors read) of `/sys/block/<dev>/stat` × 512 / tokens |
| major faults / token | Δ `pgmajfault` in the scope's `memory.stat` (fallback: `/proc/vmstat`) / tokens |
| refaults | Δ `workingset_refault_file` in `memory.stat` |
| memory PSI | `/sys/fs/cgroup/<scope>/memory.pressure` and `/proc/pressure/memory` avg10, plus Δ total |
| co-tenant slowdown | `bench/cotenant_probe.py` p50/p95 latency of touching its own 512 MB working set every 250 ms, relative to the solo run |
| daemon overhead | `/proc/<pid>/stat` utime+stime / wall |

### Machines

| id | CPU | RAM | SSD | kernel |
|---|---|---|---|---|
| L1 (dev laptop) | i5-12450H (8C/12T, AVX2) | 15 GiB + 8 GiB zram | Solidigm P41 Plus 512 GB (PCIe4 QLC), btrfs | 6.19.14 Fedora 42 |

## Baselines

| id | configuration |
|---|---|
| B0 | default llama.cpp (mmap + repack). Swap/zram allowed |
| B1 | `--no-mmap` (anonymous memory). Swap/zram allowed |
| B2 | `-nr` (mmap, no repack). Kernel default (MGLRU on) |
| B3 | `-nr`, MGLRU off (`/sys/kernel/mm/lru_gen/enabled = n`, root) |
| B4 | `-nr --mlock`. Only where the model fits the budget |
| B5 | `-nr` + naive readahead: `moepagerd --policy willneed-all` (WILLNEED the next layer's whole expert tensors) |
| B6 | `-nr` + `moepagerd --policy lru-pin` (LRU over experts, pinned) |
| D1 | `-nr` + `moepagerd --policy v` (V(e)-residency) |
| D2 | `-nr` + `moepagerd --policy v --prefetch completion` |
| D3 | `-nr` + `moepagerd --policy v --prefetch completion,predict` |
| O | Belady* oracle (simulator only, on the ground-truth trace) |

Configurations actually run in Phase 7 (`bench/phase7.sh`, experiment D):
B0, **B0S** (default llama.cpp with swap off, so held to the budget), B2,
D2 (`-nr` + moepagerd completion readahead), **P** (`-nr` + the
`tools/preload` shim, read-around off) and **PD** (P + moepagerd). B1, B3–B6,
D1 and D3 were not run: B3 needs root; pinning variants need
`CAP_IPC_LOCK`.

## Results

### Microbenchmark: fault-driven vs bulk read bandwidth (Experiment C)

Smoke runs only. A full run (Experiment C, P7.5) is still to do. Raw
output: `bench/results/L1-faultio-smoke.txt`. Setup:
- `moepager fault-io` over a 1 GiB incompressible file on btrfs
  (`compress=zstd:1`, bdi `read_ahead_kb = 4096`);
- random expert units, each split into 3 slices in different thirds of the
  file;
- 6 touching threads, desktop session running;
- WILLNEED issued per slice, either in 128 KiB chunks (what moepager does)
  or as one call, which the kernel truncates to the device's readahead
  size (IDEA_REVIEW §1.4).

"Amp" is storage bytes read (`/proc/self/io read_bytes`) divided by bytes
touched.

| machine | unit size | units × runs | mmap-fault GB/s (amp) | WILLNEED 128 KiB chunks GB/s (amp) | WILLNEED one call/slice GB/s (amp) | pread 1 MiB GB/s (amp) |
|---|---|---|---|---|---|---|
| L1 | 2.86 MB (Qwen3-30B-A3B-like) | 48 × 6 | 0.44–0.49 (3.3–3.7×) | 2.37–2.46 (1.00×) | 2.00–2.10 (1.00×) | 0.79–0.92 (1.05–1.10×) |
| L1 | 13.25 MB (gpt-oss-20b-like) | 16 × 2 | 0.76–0.87 (2.3×) | 3.09 (1.00×) | 1.92 (1.45×; 4.4 MB slices exceed the 4 MiB WILLNEED cap) | 0.99–1.00 (2.2×) |

Caveats:
- The touch pattern (64 KiB chunks interleaved across threads)
  approximates, but is not, ggml's row split.
- Neighbouring data pulled in by read-around is counted as waste. In a
  real model it is other experts, which may be used later.
- The numbers are from one machine and one filesystem.

### Experiment C on the real model file (OLMoE, Phase 7)

`bench/results/phase7-olmoe/faultio.txt`: 32 cold 3.8 MB units × 3 seeds,
8 threads.

| method | GB/s | amplification |
|---|---|---|
| mmap faults | 0.15–0.24 | 4.8× |
| WILLNEED, one call/slice | 0.97–1.42 | 1.00× |
| WILLNEED, 128 KiB chunks | 1.49–1.70 | 1.00× |
| pread 1 MiB | 0.44–0.56 | 1.24× |

### Simulator vs kernel validation (Experiment B)

Real prose trace (255 tokens, 0.488 GB of experts touched per token),
replayed with mmap and 4 threads under a cgroup wall with swap off
(`bench/results/phase7-olmoe/B-*`).

| model | budget | variant | tok/s | read GB/token (cgroup, incl. daemon) | majflt/token |
|---|---|---|---|---|---|
| OLMoE-1B-7B Q4_K_M | 1.5 GB | kernel default | 0.79 | 1.80 | 956 |
| OLMoE-1B-7B Q4_K_M | 1.5 GB | read-around off | 0.75 | 0.16 | 38,802 |
| OLMoE-1B-7B Q4_K_M | 1.5 GB | moepagerd (read-around on) | 0.66 | 2.23 | 881 |
| OLMoE-1B-7B Q4_K_M | 1.5 GB | read-around off + moepagerd | 2.58 | 0.15 | 12,342 |
| OLMoE-1B-7B Q4_K_M | 2.5 GB | kernel default | 3.25 | 0.37 | 338 |
| OLMoE-1B-7B Q4_K_M | 2.5 GB | read-around off | 5.81 | 0.02 | 5,704 |
| OLMoE-1B-7B Q4_K_M | 2.5 GB | moepagerd (read-around on) | 2.82 | 0.49 | 321 |
| OLMoE-1B-7B Q4_K_M | 2.5 GB | read-around off + moepagerd | 2.48 | 0.15 | 12,641 |

| gpt-oss-20b MXFP4 | 4 GB | kernel default | 1.05 | 1.33 | 708 |
| gpt-oss-20b MXFP4 | 4 GB | read-around off | 0.49 | 0.30 | 73,595 |
| gpt-oss-20b MXFP4 | 4 GB | moepagerd (read-around on) | 1.11 | 1.12 | 515 |
| gpt-oss-20b MXFP4 | 4 GB | read-around off + moepagerd | 1.64 | 0.31 | 16,151 |
| gpt-oss-20b MXFP4 | 6 GB | kernel default | 2.68 | 0.36 | 310 |
| gpt-oss-20b MXFP4 | 6 GB | read-around off | 1.24 | 0.12 | 28,529 |
| gpt-oss-20b MXFP4 | 6 GB | moepagerd (read-around on) | 3.27 | 0.36 | 247 |
| gpt-oss-20b MXFP4 | 6 GB | read-around off + moepagerd | 5.47 | 0.11 | 5,952 |

(gpt-oss rows: prose trace of 127 tokens, 1.272 GB of experts touched per
token, `bench/results/phase7-gptoss/B-*`.)

Read GB/token is the page-cache inflow charged to the cgroup
(`pgsteal` × 4 KiB + `file` at exit, from `memory.stat`), so it includes
moepagerd's WILLNEED reads. The replayer's own `/proc/self/io` misses
those; it agrees with the cgroup figure on every daemon-free run but
understated the daemon rows in an earlier version of this table.

Simulator LRU at 1.5 GB (≈ 37 % of expert bytes) predicts ≈ 0.14 GB/token,
which matches the read-around-off kernel run (0.16). With read-around on,
the kernel reads ≈ 12× more: pollution the simulator does not model.

### End-to-end (Experiment D)

llama-bench `-p 0 -n 32 -r 3 -lm mmap -t 8`, cold start, swap off unless
stated. tok/s is mean ± sd. Per-token metrics are divided by 97 decoded
tokens (3 × 32 + 1 warmup). Raw outputs:
`bench/results/phase7-olmoe/D-*`, `report.md`, `journal-final-pass.txt`.
Co-tenant slowdown was not measured.

| model | budget | config | tok/s | SSD GB/token | majflt/token | memory PSI avg10 |
|---|---|---|---|---|---|---|
| OLMoE-1B-7B Q4_K_M | 1.5 GB | B0 (zram allowed; 2.5 GB swap peak) | 3.17 ± 0.26 | 0.31 | 38,206 | 16.2 |
| OLMoE-1B-7B Q4_K_M | 1.5 GB | B0S | OOM-killed | — | — | — |
| OLMoE-1B-7B Q4_K_M | 1.5 GB | B2 | 0.35 ± 0.01 | 2.87 | 2,112 | 12.7 |
| OLMoE-1B-7B Q4_K_M | 1.5 GB | D2 | 0.44 ± 0.02 | 2.90 | 871 | 4.8 |
| OLMoE-1B-7B Q4_K_M | 1.5 GB | P | 0.74 ± 0.07 | 0.25 | 50,418 | 18.1 |
| OLMoE-1B-7B Q4_K_M | 1.5 GB | **PD** | **1.40 ± 0.22** | 0.34 | 23,902 | 1.0 |
| OLMoE-1B-7B Q4_K_M | 2.5 GB | B0 (zram allowed; 2.3 GB swap peak) | 7.89 ± 1.37 | 0.13 | 13,854 | 12.0 |
| OLMoE-1B-7B Q4_K_M | 2.5 GB | B0S | OOM-killed | — | — | — |
| OLMoE-1B-7B Q4_K_M | 2.5 GB | B2 | 0.87 ± 0.02 | 1.19 | 1,402 | 14.8 |
| OLMoE-1B-7B Q4_K_M | 2.5 GB | D2 | 1.25 ± 0.04 | 1.01 | 515 | 12.3 |
| OLMoE-1B-7B Q4_K_M | 2.5 GB | P | 2.51 ± 0.51 | 0.10 | 14,497 | 4.5 |
| OLMoE-1B-7B Q4_K_M | 2.5 GB | **PD** | **3.21 ± 0.65** | 0.17 | 8,161 | 2.5 |
| Qwen3-30B-A3B Q4_K_M | — | — | TBD (not run: disk space) | TBD | TBD | TBD |
| gpt-oss-20b MXFP4 | 4 / 6 GB | all | not run: the default-llama.cpp configuration swapped ≈ 10 GB into zram and the run was stopped for low system memory | — | — | — |

PD at 2.5 GB across three independent sessions: 3.30 (1 rep), 5.42 ± 1.07,
3.21 ± 0.65.

Ablation: a 20 ms per-unit refractory window gave 1.09 ± 0.04 and
2.05 ± 0.05, so it is now off by default
(`bench/results/phase7-olmoe-ablation-refr*`).

### Simulator results on synthetic traces
Synthetic traces only exercise the tooling. They are **not** evidence about
real models. `make demo` prints such a table. It is deliberately not copied
here.
