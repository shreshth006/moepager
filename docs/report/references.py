"""Bibliography for the report, IEEE style.

Sections cite with [@key] or [@a; @b]; build_report.py numbers the keys in
order of first citation and writes the list at <!-- REFERENCES -->.
"""

REFS = {
    # MoE foundations and the models studied
    "shazeer": "N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, and J. Dean, "
    "\"Outrageously large neural networks: The sparsely-gated mixture-of-experts layer,\" "
    "in *Proc. Int. Conf. Learning Representations (ICLR)*, 2017, arXiv:1701.06538.",
    "fedus": "W. Fedus, B. Zoph, and N. Shazeer, \"Switch Transformers: Scaling to trillion "
    "parameter models with simple and efficient sparsity,\" *J. Mach. Learn. Res.*, vol. 23, "
    "no. 120, pp. 1–39, 2022, arXiv:2101.03961.",
    "olmoe": "N. Muennighoff *et al.*, \"OLMoE: Open mixture-of-experts language models,\" "
    "arXiv:2409.02060, 2024.",
    "gptoss": "OpenAI, \"gpt-oss-120b & gpt-oss-20b model card,\" arXiv:2508.10925, 2025.",
    "qwen3": "A. Yang *et al.*, \"Qwen3 technical report,\" arXiv:2505.09388, 2025.",
    "hfmodels": "Model files on Hugging Face: allenai/OLMoE-1B-7B-0924-GGUF, "
    "ggml-org/gpt-oss-20b-GGUF, unsloth/Qwen3-30B-A3B-GGUF. [Online]. Available: "
    "https://huggingface.co",
    # Engine and file format
    "llamacpp": "ggml-org, \"llama.cpp: LLM inference in C/C++,\" source code, commit 970b51c "
    "(Oct. 2026); files `src/llama-mmap.cpp`, `src/llama-model-loader.cpp`, "
    "`ggml/src/ggml-cpu/repack.cpp`. [Online]. Available: https://github.com/ggml-org/llama.cpp",
    "llamaprs": "ggml-org/llama.cpp, pull requests #25294, #26824, #29887 and #29250, issue "
    "#20757, discussion #27149. [Online]. Available: https://github.com/ggml-org/llama.cpp",
    "gguf": "ggml-org, \"GGUF,\" file format specification, `docs/gguf.md` in the ggml "
    "repository. [Online]. Available: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md",
    # Linux
    "mglru": "The Linux Kernel Documentation, \"Multi-Gen LRU.\" [Online]. Available: "
    "https://docs.kernel.org/admin-guide/mm/multigen_lru.html",
    "readahead": "Linux kernel source, v6.19: `mm/readahead.c` (`force_page_cache_ra`, "
    "`page_cache_sync_ra`), `mm/filemap.c` (fault read-around), "
    "`include/trace/events/filemap.h`. [Online]. Available: https://git.kernel.org",
    "manpages": "The Linux man-pages project, `mincore(2)`, `posix_fadvise(2)`, `madvise(2)`, "
    "`process_madvise(2)`, `mlock(2)`, `cachestat(2)`. [Online]. Available: "
    "https://man7.org/linux/man-pages/",
    "cgroup": "The Linux Kernel Documentation, \"Control Group v2.\" [Online]. Available: "
    "https://docs.kernel.org/admin-guide/cgroup-v2.html",
    "psi": "The Linux Kernel Documentation, \"PSI — Pressure Stall Information.\" [Online]. "
    "Available: https://docs.kernel.org/accounting/psi.html",
    "damon": "The Linux Kernel Documentation, \"DAMON: Data Access MONitor.\" [Online]. "
    "Available: https://docs.kernel.org/admin-guide/mm/damon/index.html",
    "vmtouch": "D. Hoyte, \"vmtouch: the Virtual Memory Toucher,\" source code. [Online]. "
    "Available: https://github.com/hoytech/vmtouch",
    # MoE offloading and expert caching
    "eliseev": "A. Eliseev and D. Mazur, \"Fast inference of mixture-of-experts language models "
    "with offloading,\" arXiv:2312.17238, 2023.",
    "moeinf": "L. Xue, Y. Fu, Z. Lu, L. Mai, and M. Marina, \"MoE-Infinity: Efficient MoE "
    "inference on personal machines with sparsity-aware expert cache,\" arXiv:2401.14361, 2024.",
    "hobbit": "P. Tang, J. Liu, X. Hou, Y. Pu, J. Wang, P.-A. Heng, C. Li, and M. Guo, \"HOBBIT: "
    "A mixed precision expert offloading system for fast MoE inference,\" arXiv:2411.01433, 2024.",
    "pregated": "R. Hwang, J. Wei, S. Cao, C. Hwang, X. Tang, T. Cao, and M. Yang, \"Pre-gated "
    "MoE: An algorithm-system co-design for fast and scalable mixture-of-expert inference,\" "
    "arXiv:2308.12066, 2023.",
    "duoserve": "Y. Zhang, G. Pinkert, N. Yang, Y. Li, and D. Yuan, \"DuoServe-MoE: Dual-phase "
    "expert prefetch and caching for LLM inference QoS assurance,\" arXiv:2509.07379, 2025.",
    "survey": "J. Liu, P. Tang, W. Wang, Y. Ren, X. Hou, P.-A. Heng, M. Guo, and C. Li, \"A "
    "survey on inference optimization techniques for mixture of experts models,\" *ACM Comput. "
    "Surv.*, 2026, doi:10.1145/3794845, arXiv:2412.14219.",
    "si2026": "Y. Si, Y. Lin, D. Li, and J. Zhang, \"Who should own the expert cache? "
    "Kernel-managed tiering for trillion-parameter MoE inference,\" arXiv:2608.12103, 2026.",
    # On-device weight management and page-cache policy
    "llminflash": "K. Alizadeh, I. Mirzadeh, D. Belenko, K. Khatamifard, M. Cho, C. C. Del "
    "Mundo, M. Rastegari, and M. Farajtabar, \"LLM in a flash: Efficient large language model "
    "inference with limited memory,\" in *Proc. 62nd Annu. Meeting Assoc. Comput. Linguistics "
    "(ACL)*, 2024, arXiv:2312.11514.",
    "powerinfer": "Y. Song, Z. Mi, H. Xie, and H. Chen, \"PowerInfer: Fast large language model "
    "serving with a consumer-grade GPU,\" in *Proc. ACM SIGOPS 30th Symp. Operating Systems "
    "Principles (SOSP)*, 2024, arXiv:2312.12456.",
    "cacheext": "T. Zussman, I. Zarkadas, J. Carin, A. Cheng, H. Franke, J. Pfefferle, and A. "
    "Cidon, \"Cache is king: Smart page eviction with eBPF,\" arXiv:2502.02750, 2025.",
    # Classical algorithms
    "belady": "L. A. Belady, \"A study of replacement algorithms for a virtual-storage "
    "computer,\" *IBM Syst. J.*, vol. 5, no. 2, pp. 78–101, 1966.",
    "mattson": "R. L. Mattson, J. Gecsei, D. R. Slutz, and I. L. Traiger, \"Evaluation "
    "techniques for storage hierarchies,\" *IBM Syst. J.*, vol. 9, no. 2, pp. 78–117, 1970.",
    "fenwick": "P. M. Fenwick, \"A new data structure for cumulative frequency tables,\" "
    "*Softw.: Pract. Exper.*, vol. 24, no. 3, pp. 327–336, 1994.",
}
