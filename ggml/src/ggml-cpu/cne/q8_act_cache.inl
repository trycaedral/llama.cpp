        // src1: float32 => param type. MoE gate/up share the same src1 on decode — reuse q8.
        static thread_local struct {
            const void * src_key = nullptr;
            int64_t ne10_k = 0;
            int64_t ne11_k = 0;
            int64_t ne12_k = 0;
            size_t buf_n = 0;
            char buf[32768];
        } q8_act_cache;

        const bool q8_cacheable = ggml_cpu_cne_kernels_enabled() &&
            (ne12 == 1 && ne11 == 1 && nbw3 <= sizeof(q8_act_cache.buf));
        const bool q8_cache_hit = q8_cacheable &&
            q8_act_cache.src_key == src1->data &&
            q8_act_cache.ne10_k == ne10 &&
            q8_act_cache.ne11_k == ne11 &&
            q8_act_cache.ne12_k == ne12 &&
            q8_act_cache.buf_n == nbw3;

        if (q8_cache_hit) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                    memcpy(wdata + i12 * nbw2 + i11 * nbw1,
                           q8_act_cache.buf + i12 * nbw2 + i11 * nbw1,
                           nbw1);
                }
            }
        } else {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                    from_float((float *)((char *) src1->data + i12 * nb12 + i11 * nb11),
                               (void *)               (wdata + i12 * nbw2 + i11 * nbw1),
                               ne10);
                }
            }
            if (ith == 0 && q8_cacheable) {
                memcpy(q8_act_cache.buf, wdata, nbw3);
                q8_act_cache.src_key = src1->data;
                q8_act_cache.ne10_k = ne10;
                q8_act_cache.ne11_k = ne11;
                q8_act_cache.ne12_k = ne12;
                q8_act_cache.buf_n = nbw3;
            }
        }
