# 7. Conclusion

## 7.1 Summary

We built moepager, a toolchain and daemon that manages the Linux page cache
for Mixture-of-Experts inference using the model file's structure, without
changing the inference engine's code.

The project was run as an experiment with kill criteria fixed in advance.
On a real model (OLMoE-1B-7B) on a 15 GB laptop, the outcome was:

- **Go on the mechanism, no-go on the original emphasis.** We expected
  smarter *eviction* to be the main lever. Measurement showed the dominant
  cost under memory pressure is the kernel's fault **read-around**. On this
  machine's btrfs it read 3.7× more data than needed and filled the cache
  with unused neighbouring experts.
- **What works:**
  - turning read-around off for the model mapping, with a 64-line
    `LD_PRELOAD` shim, since no stock llama.cpp option does this on
    single-node machines;
  - plus moepagerd's expert-completion readahead in 128 KiB chunks.

  Together they turn tens of thousands of 4 KiB faults per token into bulk
  reads. See Section 6.5 for the end-to-end numbers.
- **What doesn't (yet):**
  - Completion readahead *without* turning read-around off made things
    worse.
  - Our V(e) pinning policy only matched LFU when allowed to pin most of
    the cache. With black-box (miss-only) observation it still loses
    10–40 % in most cells, up to 2×, and it needs extra privileges.
  - Default llama.cpp is faster than every page-cache configuration
    *if* it may spill 2.3–2.5 GB of repacked weights into zram (compressed
    RAM). Without swap it is OOM-killed at both budgets.
- **Smaller findings with independent value:**
  - The kernel silently truncates `WILLNEED`.
  - llama.cpp's default repacking moves 77–100 % of expert bytes out of
    the page cache.
  - llama.cpp's `--lazy-mode` keeps weights in anonymous memory.
  - btrfs reports different device numbers in `stat()` and
    `/proc/pid/maps`.

## 7.2 Limitations

- One machine, one filesystem (btrfs with a 4 MiB readahead window).
  End-to-end llama.cpp numbers exist only for OLMoE. gpt-oss-20b was
  evaluated with real routing, replay and the bandwidth benchmark, but its
  llama-bench run was stopped for low system memory. Qwen3-30B was only
  mapped from its header.
- Short runs (32 decode tokens × 3 repetitions in llama-bench); some
  configurations show large run-to-run spread.
- A desktop session was running during measurements.
- Pinning (`mlock`), demotion (`MADV_COLD`) and eBPF observation are
  implemented but untested, because they need privileges.
- The shim is still an intervention in the engine's process, although it
  doesn't change its code or binary.

## 7.3 Future work

1. Upstream: propose a llama.cpp option to apply `MADV_RANDOM` to expert
   tensors on single-node machines, which would remove the need for the
   shim.
2. Repeat on ext4 (128 KiB readahead) and on the larger models, with
   longer runs.
3. Evaluate pinning end to end with `CAP_IPC_LOCK` and a pin budget near
   90 % of the expert cache. Narrow the miss-only gap with privileged hit
   observation (DAMON [@damon]) where available.
4. Add a fallback when WILLNEED is ineffective (`MADV_POPULATE_READ`), and
   PSI-driven [@psi] control of the memory budget to protect other
   applications.

# 8. References

References are numbered in order of first citation. arXiv identifiers
are given for every preprint; where a peer-reviewed version is known, its
venue is listed first.

<!-- REFERENCES -->
