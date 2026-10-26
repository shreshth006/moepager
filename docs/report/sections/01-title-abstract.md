# moepager

## Expert-Aware Page-Cache Management for Mixture-of-Experts LLM Inference on Memory-Constrained Linux Machines

**Formal title:** System and Method for Non-Invasive, Expert-Aware
Page-Cache Management for Mixture-of-Experts Model Inference on
Memory-Constrained Linux Hosts

**Repository:** https://github.com/shreshth006/moepager · team fork:
https://github.com/AyushSinha2210/moepager (branch `ayush`)

# 1. Title, Abstract and Architecture

## 1.1 Abstract

Mixture-of-Experts (MoE) language models activate only a few "expert"
sub-networks per layer for each token: 8 of 64 in OLMoE-1B-7B, 4 of 32 in
gpt-oss-20b, 8 of 128 in Qwen3-30B-A3B. That makes them attractive for
consumer machines, which run them with llama.cpp in `mmap` mode when the
model is larger than free RAM, leaving the Linux page cache to decide
which weights stay in memory.

We show that the kernel's generic behaviour fits this workload poorly:
- **Scattered units.** One expert's weights are three slices in three
  tensors hundreds of megabytes apart.
- **Read-around pollution.** On a page fault, the kernel reads a window
  around the faulting page: 4 MiB on our test machine's btrfs. That pulls
  unused neighbouring experts into the cache and multiplies SSD traffic
  3.7–4.8×.
- **Repacking.** llama.cpp's default CPU weight repacking moves 77–100 % of
  expert bytes into anonymous memory, outside the page cache.

We present **moepager**, a Rust toolchain and an unprivileged user-space
daemon that manage the shared page cache at the granularity of
(layer, expert) units, without changing the inference engine's code or
binary:
1. It parses the GGUF header into an exact expert map.
2. It observes expert misses black-box by polling sentinel pages with
   `mincore()`.
3. It answers each miss with **expert-completion readahead**: bulk
   `posix_fadvise(WILLNEED)` of the expert's remaining slices in 128 KiB
   chunks. The chunking is necessary because the kernel silently truncates
   larger requests.
4. A 64-line `LD_PRELOAD` shim turns off fault read-around on the model
   mapping, which no stock llama.cpp option does on single-node machines.

The evaluation toolchain records ground-truth routing from llama.cpp,
computes exact LRU miss-ratio curves, simulates policies against a Belady
optimum, and runs llama.cpp under cgroup memory limits.

On a 15 GB laptop running OLMoE-1B-7B (4.2 GB) under memory limits of
1.5 GB and 2.5 GB, moepager's configuration decoded **1.40 and 3.21
tokens/s**, versus 0.35 and 0.87 tokens/s for the same engine served by
the default page cache. That is **3.7–4.0× faster**, with **7–8× less SSD
traffic** and memory-pressure stall reduced from 12.7 % to 1.0 %. Default
llama.cpp, by contrast, was OOM-killed at both limits unless allowed to
spill 2.3–2.5 GB into compressed RAM swap. On a second model, gpt-oss-20b
(12.1 GB), replaying real routing under 4 GB and 6 GB limits showed the
same pattern: 1.6–2.0× faster than the kernel default, with 3–4× less
data read.

We also report what did not work: a value-based pinning policy captured
little of the optimal policy's 34–74 % miss reduction, and completion
readahead without turning off read-around made performance worse.

## 1.2 System architecture

moepager is a **user-space control loop around the Linux page cache**. It
sits beside an unmodified inference engine, observes the engine only
through the page cache they share, and acts only through standard kernel
interfaces. Figure 1 shows the whole system; the subsections below define
its structure, components, interfaces, data model, runtime behaviour,
process and privilege model, and the design decisions behind them.

![moepager architecture](architecture-portrait.png)

*Figure 1. moepager architecture (source: `docs/report/architecture-portrait.dot`; a landscape version is `architecture.png`).*

### 1.2.1 Architectural style

- **Layered, with ports and adapters.** Policy code is pure: it consumes
  observations and emits `Action` values, and never calls the kernel. All
  operating-system effects sit behind two traits (`PageCacheOps`,
  `ResidencyProbe`), with a Linux implementation and a mock one. This lets
  the same policy run in the simulator, in tests and in the daemon.
- **One core, two hosts.** The policy crate `mp-core` is linked both into
  the trace-driven simulator (`mp-sim`) and into the live daemon
  (`moepagerd`), so a policy evaluated offline is exactly the policy
  deployed (ADR 0001).
- **Three planes.**
  - An **offline plane** runs once per model file: it parses the header
    and builds the expert map.
  - An **online plane** is the observe → decide → act loop of the daemon.
  - An **evaluation plane** records ground truth, simulates policies
    against an optimal bound, and benchmarks under memory limits.
- **Non-invasive.** Nothing in the engine's source or binary is changed.
  The only intervention in the engine's process is an optional
  `LD_PRELOAD` shim that adds one `madvise` call after `mmap`.

The crate layers, from Cargo dependencies (each layer uses only layers
below it):

```
 L4  Front ends     moepager (CLI: gguf-map, synth, analyze, sim, record, replay, fault-io, …)
                    moepagerd (daemon)
 L3  Evaluation     mp-sim (simulator)   mp-analyze (reuse, MRC)   mp-synth (synthetic traces)
 L2  Policy core    mp-core: OnlineStats, CostModel, ResidencyEngine, Prefetcher -> Action
 L1  Adapters       mp-gguf (file format)   mp-recorder (observation)   mp-os (kernel calls)
 L0  Data           mp-trace (.mpt trace format, ExpertEvent)
 ----------------------------------------------------------------------------------------
     Linux 6.x: page cache, mincore, posix_fadvise, madvise, mlock, process_madvise,
                cgroup v2, PSI
```

### 1.2.2 Components and responsibilities

| Component | Language / size | Responsibility | Needs privilege |
|---|---|---|---|
| `mp-gguf` | Rust | Parse the GGUF header; build the expert map (units, slices, sentinels, page→unit index); report which tensors llama.cpp would repack | no |
| `mp-trace` | Rust | `.mpt` binary trace format; CSV import; token inference from layer order | no |
| `mp-os` | Rust | Kernel interfaces: `mincore`, `cachestat`, chunked `WILLNEED`, `MADV_POPULATE_READ` worker, `mlock`, `process_madvise`; `/proc` parsing (maps, smaps VmFlags, cgroup) | only for pin / demote |
| `mp-recorder` | Rust | Observation: sentinel polling, full residency scans, bpftrace event conversion, trace replay on a real file | bpftrace only |
| `mp-core` | Rust | Online statistics (decayed miss rates, layer transitions), cost model, V(e) residency planner, completion and predictive prefetch → `Action`s | no |
| `mp-sim` | Rust | Byte-capacity cache simulator with an I/O channel; evictors LRU, LFU, Belady\*; controllers wrapping `mp-core` | no |
| `mp-analyze` | Rust | Reuse distances (Fenwick tree), exact LRU miss-ratio curve, locality statistics | no |
| `moepagerd` | Rust | The daemon: event source → `mp-core` → `PageCacheOps`; start-up checks (read-around, cgroup, memlock limit) | optional |
| `moepager` | Rust | CLI front end for all offline and evaluation tasks | no |
| `libmoepager_preload.so` | C, 64 lines | `LD_PRELOAD` shim: `MADV_RANDOM` on read-only shared `*.gguf` mappings | no |
| `moe_trace` | C++, 169 lines | *Experiment only*: records ground-truth routing from llama.cpp's eval callback | no |
| `bench/` | Bash | Experiments A–D under cgroup v2 memory walls; metric snapshots | no (user systemd scope) |
| `python/moepager_tools` | Python | Result collection, report figures | no |
| llama.cpp [@llamacpp] | external, unmodified | The inference engine; maps the model with `mmap` | no |

### 1.2.3 Interfaces

The boundaries between layers are four Rust traits and one value type:

```rust
/// mp-os: actuators on byte ranges of the model file (LinuxOps, MockOps).
pub trait PageCacheOps {
    fn prefetch(&mut self, offset: u64, len: u64) -> io::Result<()>; // async readahead
    fn pin(&mut self, offset: u64, len: u64) -> io::Result<()>;      // resident + unevictable
    fn unpin(&mut self, offset: u64, len: u64) -> io::Result<()>;
    fn demote(&mut self, offset: u64, len: u64) -> io::Result<()>;   // evict first
}

/// mp-os: residency queries (MappedFile via mincore, MockOps).
pub trait ResidencyProbe {
    fn resident(&self, first: u64, n: u64, out: &mut Vec<bool>) -> io::Result<()>;
    fn page_size(&self) -> u64;
}

/// moepagerd: where observations come from (SentinelSource, TraceSource).
pub trait EventSource {
    fn poll(&mut self) -> io::Result<Option<Vec<ExpertEvent>>>;
}

/// mp-core -> OS layer or simulated cache; units are dense ids l*E + e.
pub enum Action { Pin(u32), Unpin(u32), Prefetch(u32), Demote(u32) }

/// mp-sim: policy hooks driven by the simulator (StaticPin, VControl, ...).
pub trait Controller {
    fn on_access(&mut self, layer: u16, expert: u16, unit: u32, hit: bool, t_ns: u64);
    fn after_layer(&mut self, layer: u16, view: &View) -> Vec<Action>;
    fn end_token(&mut self, view: &View) -> Vec<Action>;
}
```

- The daemon is generic over `PageCacheOps` (`Daemon<O: PageCacheOps>`);
  `--ops mock` only logs actions, `--ops linux` executes them.
- An `Action` names a *unit*; the daemon expands it into the unit's slice
  byte ranges using the expert map, so policy code never sees offsets.
- In the simulator, controllers that model black-box observation must
  drop the `hit = true` calls themselves, which is how the miss-only
  regime is evaluated.
- External interfaces: the GGUF file format [@gguf]; the kernel system
  calls and `/proc` files listed under `mp-os` [@manpages]; cgroup v2
  control files [@cgroup]; the `.mpt` trace format
  (`docs/TRACE_FORMAT.md`); the expert map as JSON (`map.json`).

### 1.2.4 Data model

| Entity | Fields | Notes |
|---|---|---|
| `ExpertMap` | arch, page size, data offset, file size, `n_layers` L, `n_experts` E, top-k, `units`, `dense`, warnings | built once per file; serialisable as JSON |
| `Unit` | layer l, expert e, bytes s_u, `slices` | dense id u = l·E + e; 1,024 units for OLMoE, 6,144 for Qwen3-30B-A3B |
| `Slice` | kind (`up`, `gate`, `down`, `gate_up`, `*_bias`), tensor name, absolute offset, length | 32-byte aligned, so a page may belong to two units |
| `DenseRegion` | name, layer, offset, length | attention, norms, embeddings: everything that is not an expert |
| Sentinel | page index, (l, e), reliable flag | middle page of a weight slice, ≥ 128 KiB from its edges |
| `PageIndex` | sorted intervals + prefix-max of ends | page → `Owner::Expert{unit, slice}` or `Owner::Dense{region}` |
| `ExpertEvent` | `t_ns` u64, `token` u32, `layer` u16, `expert` u16 | 16-byte `.mpt` record; "used" (full) or "missed" (miss-only) |
| `.mpt` file | magic `MPTRACE\0`, version, record kind, JSON header (model, observation regime, units), fixed-size records | expert records (16 B) or page records (24 B) |
| `OnlineStats` | decayed counts and exposures per unit, layer-transition rows, layer time | the daemon's only state besides the pinned set |

### 1.2.5 Runtime data flow

For each decoded token, with the recommended configuration (engine with
`-nr` and the shim; daemon with `--policy v --budget-mb 0`):

1. The engine computes layer l and reads the k experts the router chose.
   Pages of the chosen units that are not resident cause page faults.
   Because of `MADV_RANDOM`, each fault reads only the faulting page.
2. The faulted page is inserted into the shared page cache. If it is a
   sentinel, the unit's sentinel flips from non-resident to resident.
3. Within one poll period (300 µs in the experiments), `SentinelSource`
   reports the flip as an `ExpertEvent` for (l, e).
4. `moepagerd` updates `OnlineStats` and infers token boundaries from the
   layer order.
5. The prefetcher returns `Action::Prefetch(u)`. `LinuxOps` issues
   `posix_fadvise(WILLNEED)` for every slice of u in 128 KiB chunks, and
   the kernel starts large asynchronous reads.
6. The engine's remaining faults on u are satisfied by pages that are
   already resident or already under I/O. They wait on the in-flight read
   instead of issuing their own small reads.
7. If a pin budget is set, every 4 tokens the residency planner may emit `Pin` / `Unpin` / `Demote` actions,
   which become `mlock` / `munlock` / `process_madvise(MADV_COLD)`.
8. Under the cgroup's `memory.max`, the kernel's reclaim (MGLRU [@mglru])
   evicts unpinned pages as usual; the daemon never evicts by itself
   unless demotion is enabled.

Offline, `mp-gguf` produces the map used in steps 2–5. For evaluation,
`moe_trace` replaces steps 2–3 with exact router output, and `mp-sim`
replaces steps 1, 6 and 8 with a model of the cache and the I/O device.

### 1.2.6 Process, privilege and memory view

| Process | Runs | Maps the model | Privilege |
|---|---|---|---|
| Engine (`llama-cli`, `llama-bench`, …) | unmodified binary, optionally `LD_PRELOAD` shim | `MAP_SHARED`, `PROT_READ`, whole file | none |
| `moepagerd` | separate process, same user | `MAP_SHARED`, `PROT_READ`, whole file (for `mincore` and `mlock`) | none for observation and prefetch; `CAP_IPC_LOCK` for pinning; `CAP_SYS_NICE` for demotion of the engine's pages |

- **No copies.** Both processes map the same file, so they share the same
  page-cache folios. A page the daemon prefetches or pins is the page the
  engine reads.
- **One memory budget.** Both run in one cgroup v2 scope with a
  `memory.max` limit [@cgroup]. Page-cache pages are charged to the
  cgroup of the task that brought them in, so co-location makes the
  daemon's prefetches count against the same budget as the engine's
  faults (ADR 0008). The daemon warns when it is not in the engine's
  cgroup.
- **Start-up checks** (`--pid`): the engine's mapping must carry the `rr`
  (random-read) VmFlag in `/proc/<pid>/smaps`, otherwise read-around is
  still on; the memlock limit is compared with the pin budget.
- **Failure isolation.** If the daemon stops, the engine continues with
  plain kernel behaviour; every action is a hint, and pins are released
  when the daemon's mapping is closed.

### 1.2.7 Key design decisions

| ADR | Decision | Reason |
|---|---|---|
| 0001 | Policy core in Rust, shared by simulator and daemon | what is simulated is what runs |
| 0002 | Treat black-box observation as miss-only | hits are served by the MMU and are invisible to user space |
| 0003 | Residency via `mlock` pin + `MADV_COLD` demote, not `fadvise(DONTNEED)` | `DONTNEED` does not drop pages another process maps |
| 0004 | Measure (go/no-go, kill criteria) before building the live daemon | avoid optimising a cost that does not exist |
| 0005 | 128 KiB guard distance for sentinel pages | a neighbour's read-around must not trigger a sentinel |
| 0006 | Chunk `WILLNEED` into 128 KiB requests | the kernel silently truncates larger requests |
| 0007 | Require `--no-repack` / `-nr` for llama.cpp | repacked weights live in anonymous memory, outside the page cache |
| 0008 | Run daemon and engine in one cgroup | fair accounting of prefetched pages |
| 0009 | Two-tier observation: unprivileged sentinels, optional bpftrace | works without root; exact events when root is available |
| 0010 | Token boundaries from layer regression with a slack | the engine exposes no token signal |
| 0011 | Optional `LD_PRELOAD` shim to turn off read-around | no stock llama.cpp option does it on single-node machines |
