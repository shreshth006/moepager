# 6. Results and Performance Evaluation

## 6.1 Setup

| Item | Value |
|---|---|
| Machine | Laptop, Intel i5-12450H (8 cores / 12 threads, AVX2), 15 GiB RAM + 8 GiB zram swap, desktop session running |
| Storage | Solidigm P41 Plus 512 GB NVMe (PCIe 4, QLC); `/home` is btrfs with `compress=zstd:1`; btrfs readahead window 4 MiB |
| OS | Fedora 42, Linux 6.19.14, MGLRU enabled |
| Engine | llama.cpp [@llamacpp] commit 970b51c (2026-10-22), CPU only, 8 threads |
| Model | OLMoE-1B-7B-0924 [@olmoe] Q4_K_M (4.21 GB; 16 layers × 64 experts, top-8; experts 3.90 GB, units ≈ 3.8 MB) |
| Memory budget | cgroup v2 `memory.max` = 1.5 GB or 2.5 GB, swap off unless stated; daemon and engine in the same cgroup |
| Cold start | model evicted from the page cache before every run |

All numbers below are measured. Raw outputs are in
`bench/results/phase7-olmoe/` (logs, CSVs, cgroup statistics, journal
excerpts).

## 6.2 Experiment A: real routing, and how much any cache policy could gain

`moe_trace` recorded 255 decode tokens for each of three prompts: 32,640
expert uses per prompt. Measured properties:

| Prompt | Same expert reused on the next token | Top-10 % experts' share of uses | Next-layer recall: transitions / popularity / previous token |
|---|---|---|---|
| chat | 41 % | 30 % | 0.60 / 0.36 / 0.42 |
| code | 48 % | 55 % | 0.78 / 0.66 / 0.48 |
| prose | 45 % | 38 % | 0.69 / 0.45 / 0.46 |

Routing is strongly local in time (41–48 % reuse) and predictable across
layers (60–78 % recall). The simulator was fed these traces with expert
caches of 15–70 % of the expert bytes. Miss ratios for the prose prompt
(other prompts in the CSVs):

| Cache size | LRU | LFU | V(e), miss-only | Belady\* (optimal) | Belady\* saves vs LRU |
|---|---|---|---|---|---|
| 15 % | 0.552 | 0.500 | 0.672 | 0.345 | 38 % |
| 25 % | 0.434 | 0.331 | 0.391 | 0.237 | 45 % |
| 35 % | 0.291 | 0.205 | 0.274 | 0.144 | 50 % |
| 50 % | 0.222 | 0.080 | 0.151 | 0.058 | 74 % |
| 70 % | 0.004 | 0.004 | 0.004 | 0.004 | 4 % |

![Simulated policies](fig-sim-prose.png)

**Reading:**
- An optimal policy would avoid 34–74 % of LRU's misses at 15–50 %
  capacity across all three prompts, so kill criterion **K1 does not
  fire**.
- Our V(e) residency policy captures only part of that gap. It beats LRU
  at 25–50 % but is *worse* at 15 %.
- Plain LFU beats V(e) in most cells.

**Why V(e) lagged (pin-fraction sweep).** The default lets V(e) pin only
60 % of the cache and leaves the rest to LRU. With full observation and a
pin fraction of 0.95, V(e) matches LFU almost exactly: prose at 50 %, 0.081
vs 0.080; code at 15 %, 0.278 vs 0.276. Its ranking is sound.

The remaining gap is the **cost of black-box observation**. With misses
only, V(e) at 0.95 has 10–40 % more misses than with full observation in
most cells, and up to 2× in a few code-prompt cells (code at 25 %: 0.162 vs
0.075). That is exactly the information hidden from an unprivileged daemon,
because page-cache hits are invisible.

Pinning needs `CAP_IPC_LOCK`, so it was not used in the end-to-end runs.
Data: `bench/results/phase7-olmoe/pinfrac-sweep/`.

## 6.3 Experiment B: what the real kernel does (and simulator validation)

The prose trace was replayed on the real model file through `mmap`, with 4
threads touching each expert's pages, under the memory wall. The trace
needs 0.49 GB of expert data per token.

"Data read" is the page-cache inflow charged to the run's cgroup:
reclaimed pages (`pgsteal`) plus file pages still charged at exit
(Section 4.9). It includes the reads moepagerd issues on the engine's
behalf. An earlier version of this table used the replayer's own
`/proc/self/io`, which leaves out the daemon's reads and so understated
traffic for the two daemon variants; on the four daemon-free runs the two
counters agree to within 0.001 GB/token.

| Budget | Variant | tokens/s | Data read / token (incl. daemon) | Major faults / token |
|---|---|---|---|---|
| 1.5 GB | kernel default | 0.79 | **1.80 GB** | 956 |
| 1.5 GB | read-around off (`MADV_RANDOM`) | 0.75 | **0.16 GB** | 38,802 |
| 1.5 GB | + moepagerd (read-around on) | 0.66 | 2.23 GB | 881 |
| 1.5 GB | **read-around off + moepagerd** | **2.58** | 0.15 GB | 12,342 |
| 2.5 GB | kernel default | 3.25 | 0.37 GB | 338 |
| 2.5 GB | read-around off | **5.81** | 0.02 GB | 5,704 |
| 2.5 GB | + moepagerd (read-around on) | 2.82 | 0.49 GB | 321 |
| 2.5 GB | read-around off + moepagerd | 2.48 | 0.15 GB | 12,641 |

![Replay](fig-replay.png)

**Reading:**
- **Read-around pollution dominates.** With the kernel's default 4 MiB
  read-around, the SSD read 1.80 GB per token at 1.5 GB to deliver
  0.49 GB, and nearly every expert access missed: neighbouring, unused
  experts filled the cache.
- **The simulator is validated once pollution is removed.** With
  read-around off, traffic fell 11× to 0.16 GB/token, which matches the
  simulator's LRU prediction for this capacity (0.14 GB/token, Section
  6.2).
- **Fewer bytes alone is not faster.** Read-around off produced 39,000
  small 4 KiB faults per token, so speed stayed at 0.75 tok/s.
- **moepagerd needs read-around off.** With read-around on, its sentinels
  fire on polluted pages and it made things slightly worse. With
  read-around off it turns the small faults into bulk reads: **3.3×
  faster** than the kernel default at 1.5 GB.
- At 2.5 GB, where most accesses already hit, read-around off alone was
  best in this replay; the daemon added overhead.

## 6.4 Experiment C: fault-driven vs bulk read bandwidth

32 cold expert units (3.8 MB each, three slices) of the real model file
were read in three ways, 3 seeds each:

| Method | GB/s delivered | SSD bytes read ÷ needed |
|---|---|---|
| mmap page faults (8 threads) | 0.15–0.24 | 4.8× |
| WILLNEED, one call per slice | 0.97–1.42 | 1.00× |
| **WILLNEED, 128 KiB chunks** | **1.49–1.70** | **1.00×** |
| pread, 1 MiB blocks | 0.44–0.56 | 1.24× |

![Bandwidth](fig-bandwidth.png)

Bulk chunked readahead delivers expert bytes **≈ 7× faster** than page
faults, with no amplification, so kill criterion **K2 does not fire**.
Chunking matters: single calls are truncated by the kernel to the device
readahead size.

## 6.5 Experiment D: end to end with llama.cpp

`llama-bench` decoded 32 tokens × 3 repetitions (after a 1-token warmup)
inside the cgroup, starting cold. Decode throughput is mean ± standard
deviation over the 3 repetitions. SSD traffic and major faults are
cgroup/device totals divided by all 97 decoded tokens. PSI is the cgroup's
memory-pressure "some" average over the last 10 s at exit.

| Budget | Configuration | tokens/s | SSD GB / token | Major faults / token | Memory PSI (some, avg10) |
|---|---|---|---|---|---|
| 1.5 GB | B0 default llama.cpp (repack; zram swap allowed) | 3.17 ± 0.26 | 0.31 | 38,206 | 16.2 % |
| 1.5 GB | B0S default llama.cpp, no swap | **OOM-killed** | — | — | — |
| 1.5 GB | B2 `--repack 0` (page cache, kernel only) | 0.35 ± 0.01 | 2.87 | 2,112 | 12.7 % |
| 1.5 GB | D2 B2 + moepagerd | 0.44 ± 0.02 | 2.90 | 871 | 4.8 % |
| 1.5 GB | P B2 + read-around off (shim) | 0.74 ± 0.07 | 0.25 | 50,418 | 18.1 % |
| 1.5 GB | **PD B2 + read-around off + moepagerd** | **1.40 ± 0.22** | 0.34 | 23,902 | **1.0 %** |
| 2.5 GB | B0 default llama.cpp (repack; zram swap allowed) | 7.89 ± 1.37 | 0.13 | 13,854 | 12.0 % |
| 2.5 GB | B0S default llama.cpp, no swap | **OOM-killed** | — | — | — |
| 2.5 GB | B2 `--repack 0` | 0.87 ± 0.02 | 1.19 | 1,402 | 14.8 % |
| 2.5 GB | D2 B2 + moepagerd | 1.25 ± 0.04 | 1.01 | 515 | 12.3 % |
| 2.5 GB | P B2 + read-around off | 2.51 ± 0.51 | 0.10 | 14,497 | 4.5 % |
| 2.5 GB | **PD B2 + read-around off + moepagerd** | **3.21 ± 0.65** | 0.17 | 8,161 | 2.5 % |

![End to end](fig-e2e.png)

**Against the same engine serving weights from the page cache (B2):**
- PD is **4.0× faster at 1.5 GB** and **3.7× faster at 2.5 GB**.
- PD reads **8.4× and 7.0× fewer bytes from the SSD** per token.
- PD cuts memory-pressure stall (PSI) from 12.7 % to 1.0 % at 1.5 GB.
  This matters for other applications on the machine.

**Splitting the gain:**
- Turning read-around off accounts for 2.1–2.9×.
- moepagerd's completion readahead adds a further 1.9× (1.5 GB) and 1.3×
  (2.5 GB) on top.
- moepagerd *without* read-around off (D2) helps only 1.3–1.4×.

**Variability:** PD at 2.5 GB was measured in three independent sessions:
3.30 (1 repetition), 5.42 ± 1.07 and 3.21 ± 0.65 tok/s. The table reports
the last.

**Against default llama.cpp (B0):**
- B0 is faster (3.17 and 7.89 tok/s), but it gets there by pushing **2.5 GB
  and 2.3 GB of repacked weights into zram**, compressed swap that lives in
  RAM and is not counted against `memory.max` (journal: "1.4G memory peak,
  2.5G memory swap peak"). It therefore used far more than the budget.
- With swap disabled, so it is held to the same budget (B0S), default
  llama.cpp was **OOM-killed at both budgets**: it cannot run at all.
- moepager's configuration runs inside the budget.

A fair speed comparison against default llama.cpp therefore needs a
machine whose swap is on disk, or a budget that includes zram. That is
future work.

**Ablation (refractory window):** suppressing repeated completion requests
for 20 ms looked like an obvious efficiency fix, but it doubled misses and
cut PD at 2.5 GB from 3.21–5.42 to 1.09–2.05 tok/s. The repeats re-fetch
pages reclaimed while the engine is still working through an expert. The
window is therefore off by default.

## 6.6 Second model: gpt-oss-20b (experiments A–C)

The same pipeline was run on gpt-oss-20b MXFP4: 12.1 GB file, 24 layers ×
32 experts, top-4, 13.25 MB per expert unit. Memory limits were 4 GB and
6 GB, and routing was recorded for 3 prompts × 128 tokens. Raw data:
`bench/results/phase7-gptoss/`.

**Experiment D did not run.** Its first configuration, default llama.cpp,
repacks 100 % of gpt-oss's expert bytes (≈ 10 GB) into anonymous memory and
swaps them to zram, which lives in RAM. The machine ran critically low on
memory and the run was stopped. This is itself evidence of the problem
moepager avoids, but there are no end-to-end gpt-oss numbers.

- **Routing (A):** 40–47 % token-to-token reuse, next-layer recall 0.54–0.62
  versus 0.39–0.46 for popularity. Belady\* saves 44–67 % of LRU's misses
  at 15–50 % capacity, similar to OLMoE.
- **Bandwidth (C), 13.25 MB units:**

  | Method | GB/s | Amplification |
  |---|---|---|
  | mmap faults | 0.14–0.44 | ≈ 2.5× |
  | WILLNEED, 128 KiB chunks | 1.44–1.88 | 1.00× |
  | WILLNEED, one call per slice | 0.93–1.10 | 1.44×, truncated |
  | pread | 0.31–0.65 | — |

- **Replay (B), prose trace, 1.27 GB of experts per token:**

| Budget | Variant | tokens/s | Data read / token (incl. daemon) |
|---|---|---|---|
| 4 GB | kernel default | 1.05 | 1.33 GB |
| 4 GB | read-around off | 0.49 | 0.30 GB |
| 4 GB | + moepagerd (read-around on) | 1.11 | 1.12 GB |
| 4 GB | **read-around off + moepagerd** | **1.64** | **0.31 GB** |
| 6 GB | kernel default | 2.68 | 0.36 GB |
| 6 GB | read-around off | 1.24 | 0.12 GB |
| 6 GB | + moepagerd (read-around on) | 3.27 | 0.36 GB |
| 6 GB | **read-around off + moepagerd** | **5.47** | **0.11 GB** |

![gpt-oss replay](fig-replay-gptoss.png)

**Reading:**
- The best configuration is the same as for OLMoE: read-around off plus
  completion readahead, **1.6× and 2.0× faster** than the kernel default,
  with **3–4× less data read** (0.31 vs 1.33 and 0.11 vs 0.36 GB/token).
- Two differences from OLMoE:
  - gpt-oss's expert slices (4.4 MB) are about as large as the read-around
    window, so read-around wastes less (≈ 1.05× at 4 GB).
  - Read-around off *alone* is slower than the default here, because 4 KiB
    faults replace useful large reads.
- moepagerd turns those small faults back into bulk reads, and also helps
  somewhat on its own with read-around on (1.06× and 1.22×).

## 6.7 Comparison with expected outcomes and with existing methods

| Expectation (IDEA_REVIEW, before measuring) | Outcome |
|---|---|
| Smarter eviction/pinning is the main lever (≥ 15 % fewer misses possible, K1) | The gap exists (Belady\* saves 34–74 % vs LRU), but our online V(e) policy did not capture it reliably, and pinning needs privileges. **Not the main lever.** |
| Bulk readahead beats fault-driven reads (K2) | Confirmed: ≈ 7× (Exp. C). |
| Default llama.cpp is beaten (K3) | Only at equal memory, where it OOMs. With zram it is faster. **Partially.** |
| (not anticipated) | Fault read-around pollution is the dominant cost: 3.7–4.8× read amplification. Turning it off is required for both correctness of black-box detection and speed. |

**Relation to existing methods:**
- Si et al. [@si2026] found that `fadvise` lookahead hints gain ≤ 5 % on a GH200
  with a `pread`-based engine. Our result differs because on an
  mmap-based engine the problem is *how* missing experts are read (small
  synchronous faults plus large wasted read-around), not *which* experts
  to keep. Completion readahead attacks exactly that.
- In-engine systems (MoE-Infinity [@moeinf], HOBBIT [@hobbit], llama.cpp
  PR #25294 [@llamaprs]) see router
  decisions directly and can avoid the issue entirely. moepager reaches a
  3.7–4.0× improvement over the stock page-cache path without modifying
  the engine's code or binary, at the cost of one `LD_PRELOAD` hint.

## 6.8 Go / no-go decision

**GO, with narrowed scope** (recorded in PHASES.md):
- The product is "read-around off + expert-completion readahead",
  validated end to end.
- V(e) pinning and predictive prefetch are demoted to research options.
- The next phase must validate on a second machine with disk swap and
  ext4, and on a larger model.
