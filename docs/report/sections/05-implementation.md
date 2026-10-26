# 5. Implementation

## 5.1 Technology stack and structure

Table: Technology stack.

| Layer | Choice | Why |
|---|---|---|
| Core, simulator, daemon, CLI | **Rust** (stable), 10 crates, ≈ 8,900 lines | Predictable latency for a daemon; fast enough for simulation sweeps; one language, so the daemon and the simulator share the *same* policy code |
| OS interface | `libc` syscalls behind traits (`PageCacheOps`, `ResidencyProbe`) | Real Linux implementation + `MockOps` for deterministic tests |
| Ground-truth capture | C++ against `libllama` (`tools/moe_trace`, 169 lines) | Needs the engine's eval callback; experiment-only |
| Read-around shim | C, `LD_PRELOAD` (`tools/preload`, 64 lines) | Disables fault read-around on the model mapping without touching the engine's code or binary |
| Privileged tracing | bpftrace script (26 lines) | Optional, root only |
| Fixtures, metric parsing, probes | Python (stdlib + pytest), 674 lines | Independent GGUF writer so parser tests aren't self-confirming |
| Experiments | Bash + cgroup v2 (`systemd-run --user`), no root | Fixed memory budget per run |
| CI | GitHub Actions: fmt, clippy `-D warnings`, tests, ruff, pytest, demo smoke run | |

Table: Rust crates and their responsibilities.

| Crate | Responsibility |
|---|---|
| `mp-gguf` | GGUF v2/v3 header parser, ggml type table, `ExpertMap`, page index, sentinels, repack-risk detection |
| `mp-trace` | `.mpt` binary trace format (spec in `docs/TRACE_FORMAT.md`), CSV import/export, token inference |
| `mp-core` | Deterministic RNG, cost model, `OnlineStats` (miss-only), `ResidencyEngine` (V(e)), `PrefetchPlanner` |
| `mp-synth` | Synthetic routing generator (skew, reuse, cross-layer affinity, drift) |
| `mp-analyze` | Fenwick-tree reuse distance, exact LRU miss-ratio curve, routing predictability, timing |
| `mp-sim` | Trace-driven cache + I/O-channel simulator, policies, Belady\* |
| `mp-os` | `mincore`, `cachestat`, chunked `fadvise`, `mlock`, `process_madvise`, `/proc/pid/maps`/`smaps` parsing |
| `mp-recorder` | Sentinel and full-scan recorders, page→expert conversion, bpftrace ingest, mmap/pread trace replayer |
| `moepager` | CLI: `gguf-map`, `synth`, `analyze`, `sim`, `import-csv`, `trace-csv`, `record`, `ingest-bpftrace`, `page2expert`, `replay`, `fault-io` |
| `moepagerd` | Daemon: policies `none`, `willneed-all`, `lru-pin`, `v`; sources `sentinel` / `trace:`; ops `linux` / `mock` |

## 5.2 Functional completeness

Table: Functional completeness.

| Feature | Status |
|---|---|
| GGUF expert map, verified on 3 real models | done, tested |
| Ground-truth routing capture + import | done, used on OLMoE |
| Trace analysis, simulator, Belady bound | done, tested, used |
| Unprivileged sentinel observation | done, tested end to end on real files |
| Chunked completion readahead | done, measured (Section 6) |
| Read-around shim | done, measured (Section 6) |
| V(e) pinning | implemented and unit-tested; live use needs `CAP_IPC_LOCK` (default memlock limit is 8 MB), **not measured** |
| `MADV_COLD` demotion | implemented; needs `CAP_SYS_NICE`, **not measured** |
| eBPF observation | script + ingest written; needs root, **not run** |
| Start-up checks: read-around still on, daemon outside the engine's cgroup | done, tested live |
| `MADV_POPULATE_READ` prefetch fallback (`--prefetch-mode populate`) | done, tested on a real file; not benchmarked |
| PSI-driven budget control, dashboard | future work |

## 5.3 Correctness

- **Tests:** 87 Rust and 13 Python tests, run in CI. They include:
  - parser checks against GGUF files written by an *independent* Python
    writer;
  - an assertion that simulated LRU equals the analytically computed LRU
    curve exactly;
  - a check that Belady\* never loses to any other policy;
  - real-kernel tests of `mincore`/`cachestat`/`fadvise`/`mlock` semantics
    on a real file;
  - end-to-end recorder tests that recover a replayed trace with 100 %
    recall;
  - daemon dry runs.
- **Real-model checks:**
  - The map of OLMoE/Qwen3/gpt-oss matches tensor sizes in the headers.
  - Each captured (token, layer) has exactly top-k experts.
  - The no-read-around replay reproduced the simulator's LRU prediction
    (Section 6.3).
- **Bugs found by testing and fixed:**
  - WILLNEED truncation (found by CI);
  - the btrfs device-number mismatch between `stat()` and `/proc/pid/maps`;
  - read-around/compression spill causing false expert detections;
  - duplicate completion prefetches (refractory window).

## 5.4 Demonstration

`make demo` runs the full pipeline with no model or root. It parses the
real Qwen3 map, generates routing, analyzes it, simulates policies, does a
daemon dry run, and records a replayed trace black-box. With a model,
`bench/phase7.sh` runs experiments A–D. Terminal screenshots of the tools
on the real OLMoE model follow.

## 5.5 Screenshots (real output on OLMoE-1B-7B)

![gguf-map](screens/01-gguf-map.png)
*Figure: Expert map from the real GGUF header, with the repack warning for this AVX2 CPU.*

![moe_trace](screens/05-moe-trace.png)
*Figure: Ground-truth routing capture through llama.cpp's eval callback.*

![import + analyze](screens/02-import-analyze.png)
*Figure: Importing the routing trace and analyzing it: reuse, predictability, exact LRU curve.*

![sim](screens/03-sim.png)
*Figure: Policy simulation, including the Belady\* bound.*

![phase 7](screens/06-phase7-d.png)
*Figure: Collected end-to-end results (B0S rows are OOM-killed runs), daemon counters, and the shim at work.*

![tests](screens/04-tests.png)
*Figure: Test and lint status.*
