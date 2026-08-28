# CNE custom CPU kernels

Caedral Notre Engine hooks inside ggml-cpu. All CNE-owned kernel code lives here;
`repack.cpp` keeps a thin include seam only.

## Toggle

`CNE_KERNELS=1` (default) — fast path on. `CNE_KERNELS=0` — stock llama.cpp (A/B).

## Layout

| File | Role |
|---|---|
| `config.h` | `ggml_cpu_cne_kernels_enabled()` |
| `q8_act_cache.inl` | Reuse q8 activations when MoE gate/up share `src1` |
| `moe_fast_path.inl` | Top-2/4 `mul_mat_id` decode dispatch → fused GEMV |
| `arch/x86/gemv_q4_moe.inl` | AVX2 q4_K fused `2vx`/`4vx` MoE GEMV |
| `arch/x86/gemv_q6_moe.inl` | AVX2 q6_K fused `2vx`/`4vx` MoE GEMV |

## Requirements

- x86_64 + AVX2, `GGML_CPU_REPACK=ON`
- Single-token decode (`ne11==1`, `ne12==1`), MoE top-K ∈ {2, 4}
- Repacked q4_K / q6_K weights

Validated: LFM2-24B-A2B Q4_K_M. See Caedral `docs/FEATURES.md` § Custom kernels.
