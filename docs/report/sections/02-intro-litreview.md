# 2. Introduction and Problem Statement

## 2.1 Background

Mixture-of-Experts (MoE) language models [@shazeer; @fedus] replace each
dense feed-forward block with many smaller "expert" networks plus a
router. For every token,
the router picks a few experts per layer (top-k), so only a small fraction
of the parameters is used per token. In the three models studied here,
OLMoE [@olmoe], gpt-oss [@gptoss] and Qwen3 [@qwen3] (values read from
the real model file headers [@hfmodels]):

Table: The three MoE models studied (values from the GGUF headers).

| Model | File (quantisation) | Size | Layers × experts | top-k | Expert share of file |
|---|---|---|---|---|---|
| OLMoE-1B-7B-0924 | Q4_K_M | 4.21 GB | 16 × 64 | 8 | 92.6 % |
| gpt-oss-20b | MXFP4 | 12.11 GB | 24 × 32 | 4 | 84.1 % |
| Qwen3-30B-A3B | Q4_K_M | 18.56 GB | 48 × 128 | 8 | 94.6 % |

Because only 4–8 experts per layer are touched per token, MoE models are an
attractive fit for consumer machines. A common way to run a model larger
than free RAM is llama.cpp [@llamacpp] in `mmap` mode: the model file is mapped into
memory, and the Linux page cache loads weights on demand and evicts them
under memory pressure.

## 2.2 Problem statement

The kernel's page cache is generic. Its reclaim policy (MGLRU [@mglru])
and its fault read-around [@readahead] know nothing about how a GGUF file
[@gguf] is laid out or how MoE
routing behaves. Our measurements on a 15 GB laptop (Section 6) show three
concrete problems:

1. **The unit of use is scattered.** One expert at one layer is stored as
   three slices in three different tensors (`ffn_up_exps`, `ffn_gate_exps`,
   `ffn_down_exps`), hundreds of megabytes apart. The kernel cannot know
   these three ranges belong together.
2. **Read-around pollutes the cache.** On a page fault, Linux reads a window
   around the faulting page: 4 MiB on this machine's btrfs filesystem. For
   expert slices of about 1–1.3 MB, most of that window belongs to
   neighbouring experts that will not be used. Replaying real OLMoE routing
   under a 1.5 GB budget, the SSD read **1.80 GB per token to deliver
   0.49 GB of needed expert weights** (3.7× amplification), and almost
   every expert access missed.
3. **Fault-driven reads are slow.** Faulting cold experts in through mmap
   delivered only 0.15–0.24 GB/s on the real model file. Reading the same
   bytes with large prefetch requests delivered 1.5–1.7 GB/s.

A further, practical problem: llama.cpp on x86 CPUs with AVX2 by default
*repacks* the most common quantised expert weights into ordinary
(anonymous) memory. That is 77 % of the expert bytes for OLMoE and Qwen3
Q4_K_M, and 100 % for gpt-oss MXFP4. Under memory pressure those pages go
to swap instead of the page cache.

**Goal:** make MoE inference faster on memory-constrained Linux machines,
using knowledge of the model file's structure and of routing behaviour,
**without modifying the inference engine** and without copying the
weights.

# 3. Literature Review and Objectives

## 3.1 Related work

**MoE offloading inside the inference engine.** Most systems that keep
only part of an MoE model in fast memory are built into the engine, where
the router's decisions are directly visible:
- mixtral-offloading (Eliseev and Mazur [@eliseev]): LRU expert cache
  plus speculative prefetch.
- MoE-Infinity (Xue et al. [@moeinf]): activation-aware expert cache.
- HOBBIT (Tang et al. [@hobbit]): mixed-precision experts, implemented as
  about 8,000 lines added to llama.cpp.
- Pre-gated MoE [@pregated], DuoServe-MoE [@duoserve] and similar
  systems: predict next-layer experts and prefetch them. Liu et al.
  [@survey] survey this design space.

Inside llama.cpp itself there are several open proposals [@llamaprs]:
PR #25294 (O_DIRECT expert streaming that bypasses the page cache),
PR #26824 (expert heat map and mmap pinning), PR #29887, issue #20757 and
discussion #27149. All of these modify the engine.

**Using the OS page cache as the expert tier.** Si et al. [@si2026] is
the closest study. On a datacenter GH200 node they
show that kernel recency (LRU) serves about as much demand as an oracle
static-frequency pin, that Belady's bound [@belady] leaves about one third of LRU's
misses addressable, and that `fadvise(WILLNEED)` prefetch hints gained only
0.3–5 %. Their engine reads with `pread`, so mmap fault behaviour and file
structure play no role.

**Operating-system mechanisms.**
- cache_ext / cachebpf (Zussman et al. [@cacheext]) lets eBPF programs
  replace page-cache eviction, but needs a patched kernel.
- MGLRU [@mglru] and DAMON [@damon] are generic and know nothing of model
  structure.
- vmtouch [@vmtouch] pins or evicts file ranges given by the user.
- "LLM in a flash" (Alizadeh et al. [@llminflash]) bundles co-used
  weights so that flash reads are large. That is the same idea as our
  expert-completion readahead, but it is implemented inside the engine and
  requires reformatting the model. PowerInfer [@powerinfer] similarly
  exploits activation locality inside the engine, keeping frequently
  activated ("hot") neurons in fast memory.

**Gap.** We found no system that manages the page cache for an
*unmodified* engine using the *model file's structure*, and none that
measures the effect of mmap fault read-around on MoE experts.

## 3.2 Objectives

- **O1.** Build an exact map from GGUF files to (layer, expert) units and
  their byte and page ranges, verified on real models.
- **O2.** Observe expert usage from outside the engine, without root.
- **O3.** Build an evaluation toolchain: ground-truth routing traces, trace
  analysis, a policy simulator with an optimal (Belady) bound, real-kernel
  replay, and benchmarks under a fixed memory budget.
- **O4.** Implement page-cache actuators: chunked readahead, pinning and
  demotion. Combine them in a daemon whose policy code is identical to the
  simulated one.
- **O5.** Decide go/no-go on real hardware with predefined kill criteria,
  and report only measured numbers.
