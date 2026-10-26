# 4. Methodology and Algorithms

This section defines the problem formally (4.1), then describes each
algorithm in the order data flows through the system: the expert map
(4.2), black-box observation (4.3), read-around suppression (4.4),
expert-completion readahead (4.5), the optional residency policy (4.6) and
the offline evaluation algorithms (4.7). Sections 4.8–4.10 give the
experimental protocol, the metric definitions and the threats to validity.
Every algorithm below corresponds to code in the repository; the crate is
named in each heading.

## 4.1 Problem model and notation

Table: Notation.

| Symbol | Meaning |
|---|---|
| F, P | the GGUF model file; the page size (4 KiB on the test machine) |
| L, E, k | number of MoE layers, experts per layer, experts chosen per token per layer (top-k) |
| u = (l, e) | an expert **unit**: expert e of layer l; U is the set of all L·E units |
| S(u) | the slices of u: one byte range `[a, b)` of F per expert tensor (`up`, `gate`, `down`, optional bias rows) |
| s_u | size of unit u in bytes, Σ over S(u) of (b − a) |
| R_{t,l} | the k experts the router selects at decode step (token) t, layer l |
| σ | an access trace: the sequence of units R_{1,1}, R_{1,2}, …, R_{T,L} |
| C | the memory budget: cgroup v2 `memory.max` [@cgroup] |
| B_d, B_b | delivered bandwidth of fault-driven reads and of bulk readahead |
| t_f | fixed cost of one demand miss (fault entry plus first I/O latency) |

Decoding token t touches, layer by layer, every unit in R_{t,l} plus the
dense (non-expert) tensors of layer l. The engine maps F read-only and
shared, so every touched byte is served from the page cache, which holds
at most C bytes (minus the engine's anonymous memory). A unit access is a
**hit** if all its pages are resident and a **miss** otherwise.

The quantity to minimise is decode time per token. Under memory pressure
it is dominated by I/O stall, which a simple cost model captures:

    stall(miss of u)  =  t_f + s_u / B_d               (kernel default)
    stall(miss of u)  =  t_f + t_detect + s_u / B_b     (with completion readahead)

Two levers follow directly. **Residency** (which units stay cached)
reduces the number of misses. **Read path** (how a missing unit is read)
reduces the cost of each one. The original hypothesis was that residency
is the main lever; Section 6 shows that on an `mmap` engine the read path
dominates, because B_d is small and the kernel's read-around inflates the
bytes read per miss.

## 4.2 Expert map from the GGUF header (`mp-gguf`)

A GGUF file [@gguf] starts with a header listing every tensor's name,
shape, ggml type and data offset. `mp-gguf` parses only this header, so a
truncated prefix of the file is enough. For each tensor named
`blk.<l>.ffn_{up,gate,down,gate_up}_exps.weight` with shape
`[ne0, ne1, E]`, the expert index is the outermost (slowest-varying)
dimension, so each expert's weights are contiguous inside the tensor:

```
Algorithm 1  BuildExpertMap(header)
  for each tensor T matching blk.<l>.ffn_*_exps.*:
      bytes  := type_size(T.type) * (ne0 * ne1 * E) / block_size(T.type)
      stride := bytes / E                       # warn if E does not divide bytes
      for e in 0 .. E-1:
          add slice [T.offset + e*stride, T.offset + (e+1)*stride) to unit (l, e)
  for each 2-D per-expert tensor ([ne0, E], e.g. *_exps.bias):
      add one row per expert to unit (l, e)
  sort all slices by offset; build page->unit interval index (below)
  for each slice: choose a sentinel page (Section 4.3)
```

- **Page → unit index.** Slices are 32-byte aligned rather than page
  aligned, so a boundary page can belong to two units. The index stores
  slices as intervals sorted by start, plus a prefix-maximum array of
  interval ends, so the owners of any page are found by binary search in
  O(log n + r), where r ≤ 2 is the number of owners.
- **Repacking check.** `mp-gguf` also reports which expert tensors
  llama.cpp's CPU backend would repack into anonymous memory, based on
  the quantisation type and the repack rules in the engine source
  [@llamacpp]. On AVX2 this covers 77 % (Q4_K_M) to 100 % (MXFP4) of the
  expert bytes, so the engine must run with `--repack 0` / `-nr` for the
  weights to stay in the page cache.
- **Cost.** O(number of tensors) to parse, O(n log n) to sort the n slices
  (n = 3·L·E; 3,072 for OLMoE, 18,432 for Qwen3-30B-A3B).
- **Verification.** Fixture GGUF files with known layouts are checked
  slice by slice (kind, offset, length of every unit), the page index is
  checked against the slices, and the real headers of OLMoE-1B-7B
  [@olmoe], gpt-oss-20b [@gptoss] and Qwen3-30B-A3B [@qwen3] are checked
  for layer, expert and top-k counts and unit sizes.

## 4.3 Black-box observation by sentinel pages (`mp-recorder`)

The engine is a black box: the daemon cannot see router decisions. It
infers them from the page cache, which is shared.

**Sentinel choice.** For each slice, one page in its middle is chosen as
the **sentinel**, at least 128 KiB (`SENTINEL_GUARD`) from either edge of
the slice. A fault in a *neighbouring* slice brings in a read-around window
around the faulting page; with the default 128 KiB window, that window
cannot reach this sentinel. Slices shorter than 2 × 128 KiB + 2P are marked
unreliable, and bias rows (a few KiB, inside other tensors' pages) get no
sentinel. On btrfs the window is 4 MiB, larger than a whole OLMoE slice,
which is why read-around must be turned off (Section 4.4) for observation
to be correct as well as fast.

**Polling.** The daemon maps F read-only (`MAP_SHARED`, `PROT_READ`) and
calls `mincore(2)` [@manpages] on the sentinel pages. This needs no
privilege and works for files the daemon does not own. (`cachestat(2)`
would need write access to the file.)

```
Algorithm 2  SentinelTick(sentinels, prev, now_ns)       # every Δ = 300 µs
  events := []
  for i, s in enumerate(sentinels):
      r := mincore(s.page)
      if r and not prev[i] and (s.layer, s.expert) not in events:
          events.append((now_ns, s.layer, s.expert))      # a miss of unit (l, e)
      prev[i] := r
  return events
```

- **Semantics.** A sentinel changing from non-resident to resident means
  "unit (l, e) has just been faulted in", which is a **miss**. Hits are
  invisible: pages already mapped are read through the CPU's page tables
  without entering the kernel, so no user-space interface reports them.
  All online statistics are therefore *miss-only* (ADR 0002).
- **Cost.** One `mincore` call per sentinel per sweep: O(3·L·E) system
  calls every Δ. The poll period Δ bounds detection latency to at most Δ
  plus the remaining sweep time.
- **False positives.** A sentinel can also turn resident because of the
  daemon's own prefetch of the same unit, or because read-around from an
  unrelated fault reached it. The first case is harmless (it re-issues a
  prefetch for a unit being read anyway). The second is eliminated by
  turning read-around off.
- **Privileged alternative.** With root, the same events can be taken
  exactly from the `filemap:mm_filemap_add_to_page_cache` tracepoint; a
  bpftrace script is included (ADR 0009). It was not used for the results.

**Token boundaries** are not visible either. Within one decode step layers
are visited in increasing order, so a new token is inferred when an event's
layer falls more than a slack (default 2 layers) below the highest layer
seen in the current step:

```
Algorithm 3  InferToken(event)                             # ADR 0010
  if max_layer is set and event.layer + slack < max_layer:
      end_token()                       # closes statistics, may trigger re-plan
  max_layer := max(max_layer, event.layer)
```

The slack absorbs reordering when several layers are discovered in the
same sweep. On ground-truth traces this recovers the true token count; on
miss-only traces a token whose misses are all in high layers can merge
with the previous one, which only affects the V(e) rate estimates.

## 4.4 Read-around suppression (`tools/preload`)

On a page fault in a file mapping, Linux reads a window of pages around
the faulting address ("read-around"), sized from the backing device's
`read_ahead_kb` (4 MiB for this machine's btrfs). For a 1–1.3 MB expert
slice, most of that window is neighbouring experts that will not be used.
The waste has two effects: up to 3.7–4.8× more SSD traffic than needed,
and a cache filled with unused pages, which raises the miss ratio.

Setting `MADV_RANDOM` on the mapping (`VM_RAND_READ` in the kernel)
disables read-around for it, so a fault reads exactly the faulting page.
Stock llama.cpp applies this only when `--numa` is set on a multi-node
machine. A 64-line `LD_PRELOAD` shim wraps `mmap()` and, for read-only
shared mappings of `*.gguf` files, calls `madvise(addr, len, MADV_RANDOM)`
after the original `mmap` returns (ADR 0011). The engine's code and binary
are unchanged. The daemon verifies the effect by checking for the `rr`
flag in the engine's `/proc/<pid>/smaps` VmFlags and warns if it is
missing.

Turning read-around off alone trades one problem for another: each miss
becomes hundreds of synchronous 4 KiB faults (39,000 per token in
Experiment B). Completion readahead (next section) turns them back into
bulk reads, but only for the unit that was actually chosen.

## 4.5 Expert-completion readahead (`moepagerd`, `mp-os`)

When a miss of unit u is observed, the daemon immediately asks the kernel
to read **all** of u asynchronously. The engine is typically still inside
the first slice (`up`); the `gate` and `down` slices, hundreds of
megabytes away in the file, arrive as large reads instead of thousands of
small faults.

```
Algorithm 4  OnMiss(u)                                     # mp-core::prefetch
  for each slice [a, b) in S(u):
      for c in a, a+128K, a+256K, ... < b:                  # chunked (ADR 0006)
          posix_fadvise(fd, c, min(128K, b - c), POSIX_FADV_WILLNEED)
  stats.observe(u, MISS)
```

- **Chunking.** A single `WILLNEED` is silently truncated by
  `force_page_cache_ra()` in `mm/readahead.c` [@readahead] to
  max(`io_pages`, `ra_pages`) pages, so a 1 MiB request populated only
  256 KiB in our tests. Issuing 128 KiB chunks makes every byte requested
  actually read.
- **Asynchrony.** `WILLNEED` queues reads and returns, so the daemon never
  blocks on I/O. The alternative `MADV_POPULATE_READ`
  (`--prefetch-mode populate`) is synchronous and runs on a worker thread;
  it is for devices where readahead is disabled and `WILLNEED` has no
  effect.
- **Prediction-free.** The action is taken only for units the router has
  already chosen, so it never wastes bandwidth on wrong guesses. A
  cross-layer predictive variant (`--predict`) exists but is not
  recommended (Section 6).
- **Repeated misses.** Each unit has three sentinels, so one real miss can
  trigger up to three prefetches of the unit. A refractory window that
  suppressed repeats within 20 ms was evaluated and *doubled* misses
  (Phase 7 ablation: decode speed fell from 5.4 to 2.1 tok/s), because
  the repeats re-fetch pages reclaimed while the engine is still working
  through the unit. The window is therefore off by default.
- **Cost.** O(s_u / 128 KiB) system calls per miss, about 30 for an OLMoE
  unit; no data is copied, and the pages are shared with the engine.
- **Memory accounting.** Pages read on the daemon's behalf are charged to
  the daemon's cgroup. Daemon and engine are therefore run in the same
  cgroup (ADR 0008), so prefetching cannot exceed the budget C.

## 4.6 Residency policy V(e) (`mp-core`; optional, needs privilege)

With `CAP_IPC_LOCK` (or a raised `RLIMIT_MEMLOCK`), the daemon can pin
units with `mlock` through its own shared mapping, which pins the same
page-cache folios the engine reads. The policy chooses a pinned set
within a pin budget C_pin ≤ C by **value per byte**:

    V(u) = rate(u) · (t_f + s_u / B_d) / s_u

that is, the expected stall avoided per token per byte held. This is a
fractional-knapsack relaxation; the greedy order is optimal for the
relaxation and near-optimal when units are small compared with C_pin.

**Rate estimator under miss-only observation.** A pinned unit produces no
misses, so naive miss counts would make every pinned unit look cold.
rate(u) is instead defined as the **miss rate per token while unpinned**:

    rate(u) = (Σ_t w_t · miss_t(u) + p·k) / (Σ_{t: u unpinned at t} w_t + k)

- w_t = g^t with g = 2^{1/h} is an exponential decay weight with half-life
  h = 64 tokens, implemented by growing the weight of new events instead
  of shrinking old ones (O(1) per event, renormalised when w exceeds
  10^100).
- p is a prior rate and k = 0.05·w_t a light shrinkage, so a unit exposed
  for one token is not estimated at exactly 0 or 1.
- While a unit is pinned its estimate is frozen. At each re-plan a random
  2 % of pinned units are released ("exploration") so frozen estimates
  are refreshed.

```
Algorithm 5  Replan(stats, cost)                  # every 4 tokens (replan_every)
  excluded := random 2 % of pinned units          # exploration, miss-only only
  cand := [(V(u) * (1.25 if pinned[u] else 1), u) for u not excluded, V(u) > 0]
  sort cand by value descending                   # hysteresis factor 1.25
  want := {}; used := 0
  for (v, u) in cand:
      if used + s_u <= C_pin: want += u; used += s_u
  emit Unpin(u) [+ Demote(u)] for u pinned and not in want     # unpins first,
  emit Pin(u) for u in want and not pinned                     # so C_pin holds
```

- **Cost.** O(n log n) per re-plan for n = L·E units (1,024 for OLMoE),
  amortised over 4 tokens.
- **Hysteresis.** A pinned unit's value is multiplied by 1.25 when
  ranking, so a newcomer must beat it by 25 % to displace it. This bounds
  pin/unpin churn.
- **Demotion.** Units that leave the pinned set can be marked cold with
  `process_madvise(pidfd, MADV_COLD)` on the engine's mapping (needs
  `CAP_SYS_NICE`). `posix_fadvise(DONTNEED)` was rejected because it does
  not drop pages that another process has mapped (verified on Linux 6.19;
  ADR 0003).

The policy code is one Rust implementation used by both the simulator and
the daemon (ADR 0001), so simulated and live behaviour cannot drift apart.

## 4.7 Offline evaluation algorithms (`mp-sim`, `mp-analyze`)

**Trace-driven simulation.** `mp-sim` replays a trace σ against a cache of
C bytes:
- a hit costs nothing;
- a miss is queued on a single FIFO I/O channel and costs
  `t_f + s/B_d`, or `t_f + t_detect + s/B_b` with completion readahead;
- compute time per layer is taken from the trace timestamps;
- the reported quantities are miss ratio, bytes read, and modelled stall.

Policies compared: LRU (a stand-in for the kernel's MGLRU [@mglru]), LFU,
prefix pinning, an oracle static-frequency pin, V(e) with full or
miss-only observation, predictive prefetch, and **Belady\***.

**Belady\*** evicts the cached unit whose next use is furthest in the
future [@belady]. For equal sizes this is optimal (MIN). With variable
sizes exact optimality is NP-hard, so the same rule is used as the
standard heuristic and labelled with an asterisk. Implementation: for
each unit, the sorted list of its positions in σ; the next use after
position i is found by binary search; cached units are kept in an ordered
set keyed by next use, so each access costs O(log N).

**Exact LRU miss-ratio curve.** `mp-analyze` computes, independently of
the simulator, the miss ratio of a byte-capacity LRU cache at *every*
capacity in one pass, using stack-distance analysis [@mattson] with a
Fenwick tree [@fenwick]:

```
Algorithm 6  ByteReuseDistances(σ)
  fenwick := zeros(N)                      # one slot per trace position
  for i, u in enumerate(σ):
      if last[u] exists:
          D_i := fenwick.sum(last[u]+1 .. i-1)   # bytes of distinct units since last use
          fenwick.add(last[u], -s_u)
      else:
          D_i := infinity                        # cold miss
      fenwick.add(i, +s_u); last[u] := i
  # access i hits in LRU of capacity C  <=>  D_i + s_u <= C
  MRC(C) := |{ i : D_i + s_u > C }| / N         # sort thresholds once, O(N log N)
```

Each position holds the size of the unit whose *latest* access is there,
so a range sum counts each distinct unit once. The whole curve costs
O(N log N). The simulator's LRU agrees with this curve exactly at every
tested capacity; this cross-check between two independent
implementations is a regression test.

## 4.8 Experimental protocol

**Hypotheses and kill criteria.** The project was run as a go/no-go
experiment (ADR 0004) with criteria fixed before measuring (IDEA_REVIEW
§6):
- **K1:** Belady\* cannot save at least 15 % of misses relative to LRU at
  any realistic capacity → residency management is pointless.
- **K2:** fault-driven bandwidth is at least 0.75× bulk bandwidth → the
  read path is not worth changing.
- **K3:** default llama.cpp beats every page-cache configuration at equal
  memory → the approach has no user.

**Experiments.**

Table: Experiments A–D.

| | Question | Independent variables | Measured |
|---|---|---|---|
| **A** | How local and predictable is real routing; how much could any policy gain? | prompt domain (chat, code, prose); cache size 15–70 % of expert bytes; policy | reuse, concentration, next-layer recall; miss ratio per policy |
| **B** | What does the real kernel do with this access pattern? | memory budget (1.5, 2.5 GB); read-around on/off; daemon on/off | tokens/s, SSD bytes/token, major faults/token |
| **C** | How fast are fault-driven vs bulk reads of expert units? | read method (fault, WILLNEED one call, WILLNEED 128 KiB chunks, pread 1 MiB) | GB/s delivered, read amplification |
| **D** | Does it help a real engine end to end? | budget; configuration B0, B0S, B2, D2, P, PD | decode tokens/s, SSD bytes/token, faults/token, memory PSI |

- **A: ground truth.** `moe_trace`, a 170-line experiment-only program,
  records the router's choices through llama.cpp's scheduler eval
  callback by reading the `ffn_moe_topk-<l>` tensors row by row. Three
  prompts × 255 decode tokens give 32,640 unit uses per prompt for OLMoE
  (255 × 16 layers × top-8). The traces are then analysed
  (4.7) and simulated at seven capacities.
- **B: real-kernel replay.** `moepager replay` touches, for each token of
  the prose trace, every page of every chosen unit through `mmap` with 4
  threads (64 KiB chunks interleaved). It isolates the kernel's paging
  behaviour from the engine's compute.
- **C: bandwidth.** 32 cold, randomly chosen expert-sized ranges of the
  real file per method, 3 seeds, 8 threads.
- **D: end to end.** `llama-bench -p 0 -n 32 -r 3 -t 8 -lm mmap`
  [@llamacpp]. Configurations: B0 default (repacking, zram swap allowed);
  B0S default without swap; B2 `--repack 0`; D2 B2 + moepagerd; P B2 +
  read-around off (shim); PD P + moepagerd.

**Controls applied to every run:**
- **Cold start:** the model's pages are dropped with
  `posix_fadvise(DONTNEED)` while no process maps it, followed by `sync`
  and a 5 s settle.
- **Memory wall:** each run executes in a transient cgroup v2 scope
  (`systemd-run --user --scope -p MemoryMax=C -p MemorySwapMax=0`)
  [@cgroup]. Swap is allowed only for B0 (`MemorySwapMax=infinity`).
  Daemon and engine share the scope.
- **Same binary, same file:** one llama.cpp build (commit 970b51c), one
  GGUF file per model, CPU only, 8 threads (`nproc` − 4).
- **Environment capture:** kernel version, CPU, free memory, filesystem
  type, every bdi's `read_ahead_kb`, MGLRU state and the llama.cpp commit
  are saved with each result set (`env.txt`).
- **Isolation of runs:** no test suite or build ran concurrently with
  measurements; one set of D runs that overlapped a test run was
  discarded and repeated.

## 4.9 Metric definitions

Table: Metric definitions.

| Metric | Definition | Source |
|---|---|---|
| tokens/s (D) | llama-bench `tg` mean ± standard deviation over repetitions | llama-bench output |
| tokens/s (B) | replayed tokens ÷ wall time | replay tool |
| SSD bytes / token (D) | (sectors read on the model's block device after − before) × 512 ÷ tokens; tokens = n_gen × reps + 1 (warm-up) | `/proc/diskstats` |
| Data read / token (B) | (`pgsteal` × 4 KiB + `file` at exit) ÷ tokens: page-cache inflow charged to the run's cgroup, including the daemon's reads | cgroup `memory.stat` |
| Major faults / token | D: `pgmajfault` of the cgroup ÷ tokens; B: the replayer's `ru_majflt` ÷ tokens | cgroup `memory.stat`, `getrusage` |
| Memory PSI | `some avg10` of the cgroup's `memory.pressure` at exit: share of the last 10 s in which at least one task stalled on memory [@psi] | cgroup |
| Read amplification | bytes read from storage ÷ expert bytes needed | C: `/proc/self/io` (single process, no daemon) |
| Miss ratio | misses ÷ unit accesses | simulator, MRC |
| Belady\* saving | 1 − misses(Belady\*) ÷ misses(LRU) at equal capacity | simulator |
| Next-layer recall | share of R_{t,l+1} contained in a size-k prediction made after layer l | `mp-analyze` |

Only measured numbers are reported. Where a configuration did not finish
(OOM kill, timeout), it is reported as such, never as zero or
extrapolated.

## 4.10 Threats to validity

- **Internal.**
  - A desktop session ran during the measurements; background activity
    adds noise, visible as run-to-run spread in some D configurations.
  - B uses one replay per variant; D uses 3 repetitions of 32 tokens.
    Differences below about 10 % should not be read as significant.
  - `/home` is btrfs with zstd compression, so device bytes can be lower
    than logical bytes. Quantised weights compress poorly, and the
    comparisons are between configurations on the same file.
  - Per-process I/O counters (`/proc/self/io`) do not include reads
    issued by another process. B's traffic is therefore taken from the
    cgroup (4.9); an earlier draft used the replayer's own counter and
    understated the daemon variants by 0.1–0.6 GB/token.
  - B0's swap device is zram, which is RAM. Its "fast" result uses memory
    outside the 1.5 / 2.5 GB budget, so it is reported separately.
- **Construct.**
  - The replay tool (B) touches pages but does not compute; it measures
    paging, not inference. D measures the real engine.
  - LRU in the simulator is an approximation of MGLRU [@mglru]; B shows
    the kernel's behaviour directly.
  - Belady\* with variable sizes is a heuristic bound, not a proven
    optimum.
- **External.**
  - One machine, one SSD, one filesystem. Read-around on ext4 defaults to
    128 KiB, so the pollution effect, and the benefit of turning it off,
    will be smaller there.
  - End-to-end results exist for OLMoE only. gpt-oss-20b has A–C
    results; its D run was stopped because the B0 configuration exhausted
    system memory. Qwen3-30B-A3B was mapped from its header only.
  - Only llama.cpp's CPU backend was studied.
- **Conclusion.** The go/no-go decision depends on K1–K3 as defined
  before measuring; the unanticipated read-around finding is reported as
  such rather than folded into the original hypothesis.
