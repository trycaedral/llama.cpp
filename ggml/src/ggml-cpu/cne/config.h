#pragma once

#include <cstdlib>
#include <cstring>

// CNE custom ggml-cpu kernels (lossless; default on). One knob for all hooks:
//   CNE_KERNELS=1 — q8 activation cache, MoE mul_mat_id dispatch, fused q4/q6 GEMV
//   CNE_KERNELS=0 — stock llama.cpp mul_mat_id path (A/B benchmarking)
static inline bool ggml_cpu_cne_kernels_enabled() {
    static int cached = -1;
    if (cached >= 0) {
        return cached != 0;
    }

    const char * v = getenv("CNE_KERNELS");
    if (!v || !*v) {
        cached = 1;
        return true;
    }
    if (v[0] == '0' && v[1] == '\0') {
        cached = 0;
        return false;
    }
    if (strcmp(v, "false") == 0 || strcmp(v, "off") == 0 || strcmp(v, "no") == 0) {
        cached = 0;
        return false;
    }
    cached = 1;
    return true;
}
