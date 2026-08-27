// CNE K4b: AVX2 q6_Kx8 GEMV + fused 2vx/4vx (port of arch/arm/repack.cpp).

#if defined(__AVX2__)

namespace {

static inline __m128i q6k_load_q8_dup64(const int8_t * p) {
    return _mm_shuffle_epi32(_mm_loadl_epi64((const __m128i *) p), 0x44);
}

static inline void q6k_reconstruct(const __m128i ql, const __m128i qh, __m128i & lo, __m128i & hi) {
    const __m128i m4b = _mm_set1_epi8(0x0F);
    lo = _mm_or_si128(_mm_and_si128(ql, m4b), _mm_slli_epi16(_mm_and_si128(qh, _mm_set1_epi8(0x03)), 4));
    hi = _mm_or_si128(_mm_srli_epi16(ql, 4), _mm_and_si128(qh, _mm_set1_epi8(0x30)));
}

static inline void q6k_sb_accum(const __m128i q6_a, const __m128i q6_b,
        const __m128i q8_a, const __m128i q8_b, int32_t & o0, int32_t & o1) {
    __m128i sb = _mm_setzero_si128();
    sb = _mm_add_epi32(sb, _mm_madd_epi16(_mm_maddubs_epi16(q6_a, q8_a), _mm_set1_epi16(1)));
    sb = _mm_add_epi32(sb, _mm_madd_epi16(_mm_maddubs_epi16(q6_b, q8_b), _mm_set1_epi16(1)));
    o0 = _mm_extract_epi32(sb, 0) + _mm_extract_epi32(sb, 1);
    o1 = _mm_extract_epi32(sb, 2) + _mm_extract_epi32(sb, 3);
}

template <int NE>
static void q6k_x8_block_nvx(const block_q8_K & q8b, const block_q6_Kx8 * q6b[NE], float acc[NE][8]) {
    const float q8d = q8b.d;
    int16_t q6_scales[NE][16 * 8];
    int32_t bias[NE][8];
    float   sb_scale[NE][8];

    for (int e = 0; e < NE; ++e) {
        for (int j = 0; j < 8; ++j) {
            bias[e][j]     = 0;
            sb_scale[e][j] = GGML_CPU_FP16_TO_FP32(q6b[e]->d[j]) * q8d;
        }
        for (int i = 0; i < 16; ++i) {
            __m128i sc = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i *) (q6b[e]->scales + i * 8)));
            _mm_storeu_si128((__m128i *) (q6_scales[e] + i * 8), sc);
        }
        for (int i = 0; i < 16; ++i) {
            const int16_t bs = q8b.bsums[i];
            for (int j = 0; j < 8; ++j) {
                bias[e][j] += (int32_t) q6_scales[e][i * 8 + j] * bs;
            }
        }
        for (int j = 0; j < 8; ++j) {
            bias[e][j] <<= 5;
        }
    }

    int32_t iacc[NE][8] = {};

    for (int half = 0; half < 2; ++half) {
        const int8_t * q8_base_l = q8b.qs + half * 128;
        const int8_t * q8_base_h = q8_base_l + 64;

        for (int sb = 0; sb < QK_K / 64; ++sb) {
            const __m128i q8_l[2] = {
                q6k_load_q8_dup64(q8_base_l + sb * 16),
                q6k_load_q8_dup64(q8_base_l + sb * 16 + 8),
            };
            const __m128i q8_h[2] = {
                q6k_load_q8_dup64(q8_base_h + sb * 16),
                q6k_load_q8_dup64(q8_base_h + sb * 16 + 8),
            };

            for (int e = 0; e < NE; ++e) {
                const uint8_t * ql_base = q6b[e]->ql + half * 512;
                const uint8_t * qh_base = q6b[e]->qh + half * 256;
                const int ql_off = sb * (QK_K / 2);
                const int qh_off = ql_off & 255;

                for (int cp = 0; cp < 4; ++cp) {
                    __m128i ql0 = _mm_loadu_si128((const __m128i *) (ql_base + ql_off + cp * 16));
                    __m128i ql1 = _mm_loadu_si128((const __m128i *) (ql_base + ql_off + 64 + cp * 16));
                    __m128i qh0 = _mm_loadu_si128((const __m128i *) (qh_base + qh_off + cp * 16));
                    __m128i qh1 = _mm_loadu_si128((const __m128i *) (qh_base + qh_off + 64 + cp * 16));
                    if (sb > 1) {
                        qh0 = _mm_srli_epi16(qh0, 2);
                        qh1 = _mm_srli_epi16(qh1, 2);
                    }

                    __m128i q6_l0, q6_h0, q6_l1, q6_h1;
                    q6k_reconstruct(ql0, qh0, q6_l0, q6_h0);
                    q6k_reconstruct(ql1, qh1, q6_l1, q6_h1);

                    int32_t sum_l0, sum_l1, sum_h0, sum_h1;
                    q6k_sb_accum(q6_l0, q6_l1, q8_l[0], q8_l[1], sum_l0, sum_l1);
                    q6k_sb_accum(q6_h0, q6_h1, q8_h[0], q8_h[1], sum_h0, sum_h1);

                    const int scale_idx_l = half * 8 + sb;
                    const int scale_idx_h = half * 8 + sb + 4;
                    const int c0 = cp * 2;
                    const int c1 = cp * 2 + 1;

                    iacc[e][c0] += sum_l0 * q6_scales[e][scale_idx_l * 8 + c0] + sum_h0 * q6_scales[e][scale_idx_h * 8 + c0];
                    iacc[e][c1] += sum_l1 * q6_scales[e][scale_idx_l * 8 + c1] + sum_h1 * q6_scales[e][scale_idx_h * 8 + c1];
                }
            }
        }
    }

    for (int e = 0; e < NE; ++e) {
        for (int j = 0; j < 8; ++j) {
            acc[e][j] += (float) (iacc[e][j] - bias[e][j]) * sb_scale[e][j];
        }
    }
}

template <int NE>
static void ggml_gemv_q6_K_8x8_q8_K_nvx(int n, float * GGML_RESTRICT s_out[NE],
        const void * GGML_RESTRICT vx_in[NE], const void * GGML_RESTRICT vy, int nr, int nc) {
    const int nb = n / QK_K;
    const block_q8_K * q8_start = (const block_q8_K *) vy;

    for (int64_t y = 0; y < nr; ++y) {
        const block_q8_K * q8_row = q8_start + y * nb;
        for (int64_t x = 0; x < nc / 8; ++x) {
            float acc[NE][8] = {};
            for (int b = 0; b < nb; ++b) {
                const block_q6_Kx8 * q6p[NE];
                for (int e = 0; e < NE; ++e) {
                    q6p[e] = (const block_q6_Kx8 *) vx_in[e] + x * nb + b;
                }
                q6k_x8_block_nvx<NE>(q8_row[b], q6p, acc);
            }
            for (int e = 0; e < NE; ++e) {
                for (int j = 0; j < 8; ++j) {
                    s_out[e][y * nc + x * 8 + j] = acc[e][j];
                }
            }
        }
    }
}

} // namespace

void ggml_gemv_q6_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    if (nr != 1 || nc <= 0) {
        ggml_gemv_q6_K_8x8_q8_K_generic(n, s, bs, vx, vy, nr, nc);
        return;
    }
    const void * vx_in[1] = { vx };
    float * s_out[1] = { s };
    ggml_gemv_q6_K_8x8_q8_K_nvx<1>(n, s_out, vx_in, vy, nr, nc);
}

void ggml_gemv_q6_K_8x8_q8_K_2vx(int n, float * GGML_RESTRICT s0, float * GGML_RESTRICT s1, size_t bs,
        const void * GGML_RESTRICT vx0, const void * GGML_RESTRICT vx1, const void * GGML_RESTRICT vy,
        int nr, int nc) {
    if (nr != 1 || nc <= 0) {
        ggml_gemv_q6_K_8x8_q8_K_2vx_generic(n, s0, s1, bs, vx0, vx1, vy, nr, nc);
        return;
    }
    const void * vx_in[2] = { vx0, vx1 };
    float * s_out[2] = { s0, s1 };
    ggml_gemv_q6_K_8x8_q8_K_nvx<2>(n, s_out, vx_in, vy, nr, nc);
}

void ggml_gemv_q6_K_8x8_q8_K_4vx(int n, float * GGML_RESTRICT s0, float * GGML_RESTRICT s1,
        float * GGML_RESTRICT s2, float * GGML_RESTRICT s3, size_t bs,
        const void * GGML_RESTRICT vx0, const void * GGML_RESTRICT vx1,
        const void * GGML_RESTRICT vx2, const void * GGML_RESTRICT vx3,
        const void * GGML_RESTRICT vy, int nr, int nc) {
    if (nr != 1 || nc <= 0) {
        ggml_gemv_q6_K_8x8_q8_K_4vx_generic(n, s0, s1, s2, s3, bs, vx0, vx1, vx2, vx3, vy, nr, nc);
        return;
    }
    const void * vx_in[4] = { vx0, vx1, vx2, vx3 };
    float * s_out[4] = { s0, s1, s2, s3 };
    ggml_gemv_q6_K_8x8_q8_K_nvx<4>(n, s_out, vx_in, vy, nr, nc);
}

#else

void ggml_gemv_q6_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemv_q6_K_8x8_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemv_q6_K_8x8_q8_K_2vx(int n, float * GGML_RESTRICT s0, float * GGML_RESTRICT s1, size_t bs,
        const void * GGML_RESTRICT vx0, const void * GGML_RESTRICT vx1, const void * GGML_RESTRICT vy,
        int nr, int nc) {
    ggml_gemv_q6_K_8x8_q8_K_2vx_generic(n, s0, s1, bs, vx0, vx1, vy, nr, nc);
}

void ggml_gemv_q6_K_8x8_q8_K_4vx(int n, float * GGML_RESTRICT s0, float * GGML_RESTRICT s1,
        float * GGML_RESTRICT s2, float * GGML_RESTRICT s3, size_t bs,
        const void * GGML_RESTRICT vx0, const void * GGML_RESTRICT vx1,
        const void * GGML_RESTRICT vx2, const void * GGML_RESTRICT vx3,
        const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemv_q6_K_8x8_q8_K_4vx_generic(n, s0, s1, s2, s3, bs, vx0, vx1, vx2, vx3, vy, nr, nc);
}

#endif
