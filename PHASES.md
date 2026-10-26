# PHASES — living plan and project memory

This file is the hand-off document. A fresh session should read it first,
then IDEA_REVIEW.md and ARCHITECTURE.md. It is updated in the same commit as
the work it describes. Task IDs (`P3.2`) are referenced in commit messages
where useful.

## Current status

**Phase 7 (go/no-go) is complete: GO, with narrowed scope.** Measured on
OLMoE-1B-7B on the dev laptop; full numbers in BENCHMARKS.md, raw data in
`bench/results/phase7-olmoe/`, write-up in `docs/report/REPORT.pdf`.

- **The product is now:** llama.cpp `--repack 0` + `tools/preload` shim
  (`MADV_RANDOM`, read-around off) + `moepagerd --policy v --budget-mb 0`
  (expert-completion readahead).
- **Result:** 3.7–4.0× faster than the stock page-cache path, with 7–8×
  less SSD traffic, at 1.5 and 2.5 GB budgets.
- **Comparison with default llama.cpp:** it is OOM-killed at those budgets
  without swap. With zram it is faster, but uses 2.3–2.5 GB of RAM-backed
  swap beyond the budget.
- **Demoted:** V(e) pinning (LFU did better in simulation; needs
  `CAP_IPC_LOCK`) and predictive prefetch.
- **Second model:** gpt-oss-20b experiments A–C completed
  (`bench/results/phase7-gptoss/`). The same configuration was best:
  replay 1.6–2.0× faster, 3–4× less data read. Experiment D was stopped
  because the default-llama.cpp baseline swapped ≈ 10 GB into zram and ran
  the machine low on memory. Rerun D without the B0 config
  (`DCFGS="B0S B2 D2 P PD"`).
- **Tests:** 87 Rust + 13 Python, all passing; clippy and ruff clean; CI green.
- **Team docs:** Ayush's docs branch was merged on GitHub (PR #1). Its
  factual errors were then corrected on `main`; the record is in
  `docs/report/REVIEW_AYUSH_BRANCH.md`.

## Next up

1. **P8.2**: evaluate pinning end to end with `CAP_IPC_LOCK` and a budget ≈ 90 % of expert capacity (P8.3, P8.5 and P8.7 are done).
2. **P10.1** validation on a second machine (ext4, disk swap), so that
   default llama.cpp can be compared at equal memory, and on Qwen3-30B-A3B
   (needs ~19 GB of free disk).
3. Upstream: propose a llama.cpp option for `MADV_RANDOM` on expert tensors
   (would remove the shim).

## Phases

### Phase 0 — Idea review ✅
- [x] P0.1 Verify GGUF expert layout on real headers (Qwen3-30B-A3B, gpt-oss-20b, OLMoE)
- [x] P0.2 Verify llama.cpp decode access path (mul_mat_id, repack) from source
- [x] P0.3 Verify tracepoint fields, fadvise/mlock/mincore/cachestat semantics (source + experiments here)
- [x] P0.4 Related-work search (llama.cpp PRs/discussions, papers, kernel work)
- [x] P0.5 Back-of-envelope, counter-arguments, go/no-go design with kill criteria
- [x] P0.6 Write IDEA_REVIEW.md, RELATED_WORK.md

### Phase 1 — Scaffold and docs ✅
- [x] P1.1 PRD, ARCHITECTURE, PHASES, BENCHMARKS, README, CONTRIBUTING, trace spec, ADRs, PRIVILEGES
- [x] P1.2 Cargo workspace, crate skeletons, rustfmt/clippy config
- [x] P1.3 Python package skeleton (pyproject, ruff, pytest)
- [x] P1.4 Makefile (`test`, `lint`, `demo`, `bench`)
- [x] P1.5 CI workflow (fmt, clippy, tests, ruff, pytest, demo smoke)

### Phase 2 — GGUF map ✅
- [x] P2.1 ggml type table (block size, type size) incl. K-quants, IQ, MXFP4
- [x] P2.2 GGUF v2/v3 header parser (metadata KV, tensor infos, alignment, data offset)
- [x] P2.3 Python independent fixture writer + generated fixtures
- [x] P2.4 ExpertMap: units, slices (3-D weights, 2-D bias rows), page sets, shared boundary pages
- [x] P2.5 Reverse lookup page → (unit, slice) / dense region; sentinel pages
- [x] P2.6 `moepager gguf-map` CLI → map.json + summary
- [x] P2.7 Repack-risk detection (types × host CPU flags) with warning

### Phase 3 — Traces, synthetic generator, analyzer ✅
- [x] P3.1 Trace format spec + reader/writer (expert and page records), CSV export
- [x] P3.2 Token-boundary inference for black-box traces
- [x] P3.3 Deterministic RNG (SplitMix64/xoshiro256**) + Zipf sampler
- [x] P3.4 Synthetic generator: layers, experts, top-k, skew, cross-layer affinity, reuse, drift, timing
- [x] P3.5 Analyzer: reuse distance (units, bytes), exact LRU miss-ratio curve
- [x] P3.6 Analyzer: transitions + top-m recall, token reuse, popularity, per-layer timing
- [x] P3.7 `moepager synth` / `moepager analyze` CLIs

### Phase 4 — Simulator and policies ✅
- [x] P4.1 mp-core OnlineStats (full and miss-only observation regimes)
- [x] P4.2 mp-core ResidencyEngine (V(e), budget, hysteresis, exploration)
- [x] P4.3 mp-core PrefetchPlanner (transition scores, deadline budget) + completion readahead flag
- [x] P4.4 Simulator engine: byte-capacity cache, FIFO I/O channel, stall accounting
- [x] P4.5 Policies: LRU, LFU, prefix-pin, static-freq oracle, V-residency, V+prefetch, Belady*
- [x] P4.6 Cross-check: simulated LRU == analyzer MRC
- [x] P4.7 `moepager sim` CLI: policy × budget sweep → CSV/markdown table

### Phase 5 — Recorder and daemon skeleton ✅
- [x] P5.1 mp-os traits + MockOps
- [x] P5.2 Linux probes: mincore, cachestat; Linux ops: fadvise, mlock, process_madvise, /proc/pid/maps lookup
- [x] P5.3 Full-scan mincore diff recorder (page trace)
- [x] P5.4 Sentinel recorder (expert trace, miss-only)
- [x] P5.5 page→expert conversion; bpftrace script + ingest
- [x] P5.6 moepagerd: config, event loop, dry-run on traces with MockOps
- [x] P5.7 moepagerd: live mode wiring (sentinel source + LinuxOps) — untested on real engine

### Phase 6 — Benchmark harness ✅
- [x] P6.1 `moepager replay` (mmap replay of an expert trace on a real file) + `moepager fault-io` microbench
- [x] P6.2 cgroup v2 runner (`systemd-run --user`, memory.max) and metric snapshots (vmstat, diskstats, PSI, memory.stat)
- [x] P6.3 llama.cpp baseline matrix script (default, -nr, -nr+mlock where fits, -nr+naive WILLNEED, -nr+daemon policies)
- [x] P6.4 Co-tenant probe app
- [x] P6.5 Python metric parsers + report table, with tests
- [x] P6.6 `make demo`, `make bench` entry points

### Phase 7 — Go/no-go on real hardware ✅ (GO, narrowed)
- [x] P7.1 Ground-truth expert trace capture tool (libllama eval callback on `ffn_moe_topk-*`) + `moepager import-csv`, experiment-only (see docs/RUNBOOK_GO_NO_GO.md)
- [x] P7.2 Build llama.cpp (970b51c); download OLMoE. Qwen3/gpt-oss not run (disk space): moved to P10.1
- [x] P7.3 Experiment A: simulator sweeps on real traces
- [x] P7.4 Experiment B: `moepager replay` under memory.max versus simulator LRU (validation)
- [x] P7.5 Experiment C: `moepager fault-io` full run (fault vs bulk bandwidth)
- [x] P7.6 Experiment D: end-to-end llama-bench, configs B0/B0S/B2/D2/P/PD (MGLRU-off B3 not run: needs root)
- [x] P7.7 Decide go/no-go against K1–K4, record in decision log

### Phase 8 — Live daemon hardening ← NEXT (re-scoped after Phase 7)
- [ ] P8.1 libbpf-rs CO-RE tracer for filemap tracepoints (inode filter in-kernel)
- [ ] P8.2 Live actuators hardened (mlock budget accounting, process_madvise demotion, fallback chain)
- [x] P8.3 Same-cgroup check: `moepagerd --pid` warns when daemon and engine are in different cgroups
- [ ] P8.4 Optional DAMON-assisted hit observation (root)
- [x] P8.5 `--prefetch-mode populate`: `MADV_POPULATE_READ` on a worker thread with a bounded queue (tested on a real file; not benchmarked end to end)
- [ ] P8.6 Compute-time (not wall-time) layer estimate for the prefetch deadline budget (low priority: prediction prefetch demoted)
- [x] P8.8 Warn when the engine maps the model with read-around on (`/proc/pid/smaps` VmFlags `rr`)
- [x] P8.9 Read-around shim (`tools/preload`, ADR 0011)
- [x] P8.10 Refractory-window ablation; kept, but off by default (doubled misses when on)
- [x] P8.7 Investigated V(e) vs LFU (pin-fraction sweep on real traces): with full observation and pin fraction 0.95, V(e) ≈ LFU. The gap was the 60 % default plus miss-only blindness (10–40 % more misses in most cells, up to 2×). Recommendation: pin budget ≈ 90 % of expert capacity. End-to-end pinning still needs `CAP_IPC_LOCK` (P8.2)

### Phase 9 — Co-tenant protection
- [ ] P9.1 PSI-driven budget controller (`/proc/pressure/memory`, cgroup `memory.pressure`)
- [ ] P9.2 Co-tenant benchmark

### Phase 10 — Full benchmark campaign and reporting
- [ ] P10.1 Full matrix on ≥ 2 machines, ≥ 3 models
- [ ] P10.2 Dashboard / report generation

### Future ideas (not scheduled)
- cache_ext backend implementing V(e) in-kernel (needs a patched kernel).
- Support other mmap engines (ollama is llama.cpp; check mistral.rs and candle mmap paths).
- Auto-tuning `read_ahead_kb` / `MADV_RANDOM` for the dense part.

## Known issues / untested on real hardware

**Untested on real hardware (no llama.cpp, model, root or extra
capabilities in the dev session):**
- `bench/run_llama.sh` (superseded by `bench/phase7.sh`, which *was* run);
  the live daemon is now measured with llama-bench, but only on OLMoE and
  on one machine;
- `bpf/moepager.bt` (needs root/bpftrace) and the bpftrace ingest on real
  output (only on sample text);
- pinning beyond the 8 MB `RLIMIT_MEMLOCK` (needs `CAP_IPC_LOCK`) and
  `process_madvise(MADV_COLD)` demotion (needs `CAP_SYS_NICE`);
- MGLRU-off baseline B3 (needs root);
- co-tenant probe under real memory pressure;
- larger models (Qwen3-30B-A3B, gpt-oss-20b) end to end; ext4; disk swap.

**Known issues:**

- Repack default: llama.cpp on AVX2 repacks Q4_0/Q4_K/IQ4_NL/MXFP4 into
  anonymous memory. moepager needs `--no-repack`. Measured: default
  llama.cpp's cgroup swapped 2.3–2.5 GB, and was OOM-killed without swap.
- **Read-around is the dominant cost** (Phase 7). moepagerd with
  read-around on helps only 1.3–1.4×. Its sentinels fire on polluted
  pages (331k events for 32k real accesses in replay). Always run the
  engine with the shim.
- **V(e) residency** underperforms LFU at the default 60 % pin fraction.
  At 0.95 with full observation it matches LFU. Miss-only observation
  costs 10–40 % more misses in most cells, up to 2× (P8.7).
- **Run-to-run variance** of PD at 2.5 GB is high: 3.2–5.4 tok/s across
  sessions.
- llama.cpp `--lazy-mode on` puts weights in anonymous memory: OOM-killed
  under the wall. `--numa` applies `MADV_RANDOM` only on multi-node
  machines.

- Prediction prefetch derives its deadline budget from the *observed* layer
  time. Under I/O stalls that estimate includes stall time, so the budget
  inflates and prefetch competes with demand misses. On synthetic traces this
  wastes ≈0.4 GB/token. The daemon only sees wall time, so a fix needs a
  compute-time estimate (e.g. the minimum gap over recent tokens). Open.

- btrfs device quirk: `stat()` reports the subvolume's anonymous device
  (0:45 here), while /proc/pid/maps and the filemap tracepoints use the
  superblock device (0:23). Mapping lookup therefore matches on inode plus
  device-or-path, and the eBPF filter must learn `s_dev` from our own
  /proc/self/maps (`kernel_dev_of_mapping`), not from `stat()`.
- `process_madvise(MADV_COLD)` demotion is implemented but untested: it
  needs `CAP_SYS_NICE` and a running engine.

- **btrfs readahead is 4 MiB** (`/sys/class/bdi/btrfs-1/read_ahead_kb =
  4096` on the dev laptop, versus 128 KiB for the NVMe block device).
  Sequential reads of one expert pull neighbouring experts' interiors into
  the cache. This causes false positives for miss-based expert inference.
  It may also mean mmap read-around on btrfs amplifies llama.cpp's expert
  reads far beyond the 128 KiB assumed in IDEA_REVIEW §1.2. Measure with
  `moepager fault-io` (bytes read vs touched) in Experiment C. The e2e
  recorder tests disable readahead (`POSIX_FADV_RANDOM`) to test recorder
  logic in isolation.
- btrfs `compress=zstd` reads whole 128 KiB compressed extents. Measured:
  reading a 544 KiB slice inserted 17 pages of the neighbouring expert.
  Page→expert conversion ignores pages within 128 KiB of slice edges.
- Sentinel and scan recorders are tested only against the replayer
  (pread, readahead off, cache dropped per token), not against llama.cpp.

- **Single `WILLNEED` calls are silently truncated** to
  `max(bdi->io_pages, ra->ra_pages)`. Source: `force_page_cache_ra()` in
  mm/readahead.c. Observed on the GitHub runner: exactly 64 pages (256 KiB)
  populated for a 1 MiB request. The first CI failure was misread as "no
  effect"; it was this truncation.
  - **Fix:** `LinuxOps::prefetch` and `fault-io` issue 128 KiB chunks
    (`mp_os::WILLNEED_CHUNK`).
  - The dev laptop hid the bug because btrfs's bdi readahead is 4 MiB.
  - Prior negative WILLNEED results (RELATED_WORK) may or may not have been
    affected; worth checking when comparing.
  - The tests still skip if WILLNEED populates nothing at all. A
    readahead-independent fallback (`MADV_POPULATE_READ` from a worker
    thread) remains future work.

## Decision log

| date | decision | why |
|---|---|---|
| 2026-09-24 | Keep the idea, modified (IDEA_REVIEW §7) | OS-level black-box niche is unoccupied. Assumptions corrected (repack, DONTNEED, hits invisible) |
| 2026-09-24 | Gate the daemon behind a go/no-go (K1–K4) | Prior evidence (arXiv 2608.12103) says prediction prefetch and fancy residency win less than intuition suggests |
| 2026-09-24 | Rust core + Python tooling, dual MIT/Apache-2.0 | Shared policy code between simulator and daemon. Python fixture writer is an independent check |
| 2026-09-29 | Prefetch reframed around expert-completion readahead; prediction prefetch kept behind a flag | fault-io smoke run: ≈4× bandwidth gap between mmap faults and bulk WILLNEED. Simulator: prediction wastes ≈0.4 GB/token on synthetic traces |
| 2026-09-29 | Simulator cost defaults = measured dev-laptop numbers (0.45 / 2.0 GB/s) | Replace guessed 1 / 3 GB/s with measurements; re-measure per machine |
| 2026-09-30 | Chunk WILLNEED into 128 KiB requests | The kernel truncates each WILLNEED to the device readahead size (found via CI) |
| 2026-10-23 | Phase 7 decision: **GO, narrowed** to read-around off + completion readahead | K1/K2 did not fire; read-around pollution dominant; PD 3.7–4.0× over B2 (BENCHMARKS.md) |
| 2026-10-23 | Ship an optional LD_PRELOAD read-around shim (ADR 0011) | No stock llama.cpp option disables read-around on single-node machines; `--lazy-mode` OOMs |
| 2026-10-23 | Refractory window off by default | Ablation: 20 ms window doubled misses, PD 2.5 GB 5.4 → 2.1 tok/s |
| 2026-10-23 | Demote V(e) pinning and prediction prefetch to research options | LFU beat V(e) in simulation; prediction wasted bandwidth; pinning needs privileges |
| 2026-10-01 | Bulk bandwidth default 2.4 GB/s (chunked WILLNEED smoke) | Re-measured after the chunking fix |
| 2026-10-08 | Stop at end of phase 6 (≈50 %) | Planned scope. Deferred to later phases: real-hardware runs (P7), live eBPF/daemon hardening (P8), PSI tuning (P9), dashboard (P10) |
