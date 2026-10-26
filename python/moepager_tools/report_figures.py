#!/usr/bin/env python3
"""Render the report figures from a bench/phase7.sh results directory.

Usage: report_figures.py RESULTS_DIR OUT_DIR

Reads only measured outputs (faultio.txt, B-*/*/replay.txt, sim-*.csv,
D-*/*/llama.out) and writes PNGs. Palette: the validated reference
categorical slots 1-4 (light mode); every bar carries a visible value label
because slots 3-4 are below 3:1 contrast on white.
"""

from __future__ import annotations

import csv
import re
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(ax, ylabel: str) -> None:
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)


def label_bars(ax, bars, fmt: str) -> None:
    for b in bars:
        h = b.get_height()
        ax.annotate(
            fmt.format(h),
            (b.get_x() + b.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=INK,
        )


def bandwidth(res: Path, out: Path) -> None:
    """Experiment C: GB/s and amplification per read method (mean over seeds)."""
    text = (res / "faultio.txt").read_text()
    blocks = re.split(r"^-- seed \d+ willneed-chunk-kb (\d+)$", text, flags=re.M)
    vals: dict[str, list[tuple[float, float]]] = {}
    for chunk, body in zip(blocks[1::2], blocks[2::2], strict=False):
        for m in re.finditer(
            r"^\| (fault|willneed|pread) \| ([\d.]+) \| [\d.]+ \| ([\d.]+) \|", body, re.M
        ):
            mode = m.group(1)
            if mode == "willneed":
                mode = (
                    "WILLNEED,\n128 KiB chunks" if chunk == "128" else "WILLNEED,\none call/slice"
                )
            elif chunk != "128":
                continue  # fault/pread are identical across chunk settings; use one set
            name = {"fault": "mmap page\nfaults", "pread": "pread\n1 MiB"}.get(mode, mode)
            vals.setdefault(name, []).append((float(m.group(2)), float(m.group(3))))
    order = [
        "mmap page\nfaults",
        "WILLNEED,\none call/slice",
        "WILLNEED,\n128 KiB chunks",
        "pread\n1 MiB",
    ]
    order = [o for o in order if o in vals]
    gbs = [statistics.mean(v[0] for v in vals[o]) for o in order]
    amp = [statistics.mean(v[1] for v in vals[o]) for o in order]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    b = axes[0].bar(order, gbs, color=SERIES[0], width=0.6)
    label_bars(axes[0], b, "{:.2f}")
    style(axes[0], "GB/s of expert bytes delivered")
    axes[0].set_title("Throughput (higher is better)", color=INK, fontsize=10)
    b = axes[1].bar(order, amp, color=SERIES[0], width=0.6)
    label_bars(axes[1], b, "{:.2f}×")
    style(axes[1], "SSD bytes read ÷ bytes needed")
    axes[1].set_title("Read amplification (lower is better)", color=INK, fontsize=10)
    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle(
        "Cold OLMoE expert units read from the real model file (mean of 3 seeds)",
        color=INK,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out / "fig-bandwidth.png", dpi=200)
    plt.close(fig)


def cgroup_read_gb(run: Path, tokens: int) -> float | None:
    """Page-cache inflow per token charged to the run's cgroup, in GB.

    Reclaimed pages (pgsteal) plus file pages still charged at exit. Unlike
    the replayer's own /proc/self/io, this includes reads issued by
    moepagerd's WILLNEED; on daemon-free runs the two agree.
    """
    f = run / "memory.stat"
    if not f.exists():
        return None
    st = dict(line.split() for line in f.read_text().splitlines())
    return (int(st["pgsteal"]) * 4096 + int(st["file"])) / tokens / 1e9


def replay(
    res: Path, out: Path, model: str = "OLMoE", need: str = "0.49", name: str = "fig-replay.png"
) -> None:
    """Experiment B: replay of the real prose trace under a memory wall."""
    variants = [
        ("plain", "kernel default"),
        ("random", "read-around off"),
        ("daemon", "+ moepagerd\n(read-around on)"),
        ("randaemon", "read-around off\n+ moepagerd"),
    ]
    budgets = sorted(p.name[2:] for p in res.glob("B-*") if p.is_dir())
    data = {}
    for bud in budgets:
        for v, _ in variants:
            f = res / f"B-{bud}" / v / "replay.txt"
            if f.exists():
                t = f.read_text()
                tps = float(re.search(r"tok/s=([\d.]+)", t).group(1))
                tokens = int(re.search(r"tokens=(\d+)", t).group(1))
                ssd = cgroup_read_gb(f.parent, tokens)
                if ssd is None:
                    ssd = float(re.search(r"storage_read/token=([\d.]+)GB", t).group(1))
                data[(bud, v)] = (tps, ssd)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    w = 0.8 / len(variants)
    for i, (v, label) in enumerate(variants):
        xs = [j + (i - (len(variants) - 1) / 2) * w for j in range(len(budgets))]
        for k, (ax, fmt) in enumerate(((axes[0], "{:.2f}"), (axes[1], "{:.2f}"))):
            ys = [data.get((b, v), (0, 0))[k] for b in budgets]
            bars = ax.bar(xs, ys, w * 0.92, color=SERIES[i], label=label)
            label_bars(ax, bars, fmt)
    for ax, t, yl in (
        (axes[0], "Replay speed (higher is better)", "tokens/s"),
        (axes[1], "Data read, incl. daemon (lower is better)", "GB read per token"),
    ):
        ax.set_xticks(range(len(budgets)), [f"memory.max = {b}" for b in budgets])
        style(ax, yl)
        ax.set_title(t, color=INK, fontsize=10)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(
        f"Experiment B: real {model} routing replayed on the real file "
        f"({need} GB of experts needed per token)",
        color=INK,
        fontsize=10.5,
    )
    fig.tight_layout()
    fig.savefig(out / name, dpi=200)
    plt.close(fig)


def simulation(res: Path, out: Path, domain: str = "prose") -> None:
    """Experiment A: miss ratio vs cache size for the main policies."""
    rows = list(csv.DictReader((res / f"sim-{domain}.csv").open()))
    caps = sorted({int(r["capacity_bytes"]) for r in rows})
    total = 3.901e9
    pol = [
        ("lru", "LRU (kernel-like)"),
        ("lfu", "LFU"),
        ("v-miss", "V(e), miss-only"),
        ("belady*", "Belady* (optimal)"),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, (p, name) in enumerate(pol):
        ys = []
        for c in caps:
            r = next(
                r
                for r in rows
                if r["policy"] == p and int(r["capacity_bytes"]) == c and r["completion"] == "false"
            )
            ys.append(int(r["misses"]) / max(int(r["accesses"]), 1))
        xs = [100 * c / total for c in caps]
        ax.plot(xs, ys, color=SERIES[i], linewidth=2, marker="o", markersize=5, label=name)
    style(ax, "miss ratio")
    ax.set_xlabel("cache size (% of expert bytes)", color=INK2)
    ax.set_xlim(10, 75)
    ax.set_title(
        f"Experiment A: simulated policies on real OLMoE routing ({domain} prompt)",
        color=INK,
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"fig-sim-{domain}.png", dpi=200)
    plt.close(fig)


def end_to_end(res: Path, out: Path) -> dict:
    """Experiment D: llama-bench decode tokens/s per configuration and budget."""
    cfgs = [
        ("B0", "default\n(repack, zram swap)"),
        ("B0S", "default\n(no swap)"),
        ("B2", "--repack 0"),
        ("D2", "--repack 0\n+ moepagerd"),
        ("P", "--repack 0\n+ read-around off"),
        ("PD", "read-around off\n+ moepagerd"),
    ]
    budgets = sorted(p.name[2:] for p in res.glob("D-*") if p.is_dir())
    table = {}
    for b in budgets:
        for c, _ in cfgs:
            f = res / f"D-{b}" / c / "llama.out"
            m = re.search(r"tg\d+ \|\s+([\d.]+) ± ([\d.]+)", f.read_text()) if f.exists() else None
            table[(b, c)] = (float(m.group(1)), float(m.group(2))) if m else None
    fig, ax = plt.subplots(figsize=(10, 3.8))
    w = 0.38
    for i, b in enumerate(budgets):
        xs = [j + (i - 0.5) * w for j in range(len(cfgs))]
        ys = [table[(b, c)][0] if table[(b, c)] else 0 for c, _ in cfgs]
        es = [table[(b, c)][1] if table[(b, c)] else 0 for c, _ in cfgs]
        ax.bar(
            xs,
            ys,
            w * 0.92,
            color=SERIES[i],
            label=f"memory.max = {b}",
            yerr=es,
            error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 2},
        )
        for x, y, e, (c, _) in zip(xs, ys, es, cfgs, strict=False):
            txt = "OOM" if table[(b, c)] is None else f"{y:.2f}"
            ax.annotate(
                txt,
                (x, y + e),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=INK,
            )
    ax.set_xticks(range(len(cfgs)), [n for _, n in cfgs], fontsize=8)
    style(ax, "decode tokens/s")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(
        "Experiment D: llama.cpp (llama-bench, OLMoE-1B-7B Q4_K_M) under a cgroup memory wall",
        color=INK,
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out / "fig-e2e.png", dpi=200)
    plt.close(fig)
    return table


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[2] == "--replay-only-gptoss":
        replay(Path(argv[0]), Path(argv[1]), "gpt-oss-20b", "1.27", "fig-replay-gptoss.png")
        return 0
    if len(argv) != 2:
        print(__doc__)
        return 2
    res, out = Path(argv[0]), Path(argv[1])
    out.mkdir(parents=True, exist_ok=True)
    bandwidth(res, out)
    replay(res, out)
    for d in ("prose", "chat", "code"):
        simulation(res, out, d)
    t = end_to_end(res, out)
    for k, v in sorted(t.items()):
        print(k, v)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
