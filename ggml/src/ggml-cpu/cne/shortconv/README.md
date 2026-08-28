# Shortconv kernels (LFM2 / lfm2moe)

Shortconv layers use `shortconv.in_proj` / `out_proj` q4_K `MUL_MAT` plus a tiny
`SSM_CONV` (l_cache=3). On prepared LFM2 decode, shortconv matmul is ~4% wall
(after MoE kernels shipped); `SSM_CONV` is negligible.

## Shipped (under `CNE_KERNELS`)

| Hook | File | What |
|---|---|---|
| Decode `MUL_MAT` | `../mul_mat_decode.inl` | Skip chunk pool for `ne11==1`; direct repack GEMV |

Benefits shortconv in_proj/out_proj, attention, dense FFN, and any other
single-token q4 `MUL_MAT` on the repack path.

## Backlog

Fused per-layer kernel (in_proj → B·x → conv → C·y → out_proj) as a custom
ggml op in `lfm2.cpp` — needs graph hook + identity gate.
