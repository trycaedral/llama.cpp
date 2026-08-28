        // Single-token decode: skip chunk pool; partition output rows across threads.
        if (ggml_cpu_cne_kernels_enabled() && ne11 == 1 && ne12 == 1) {
            const int64_t nr0 = ggml_nrows(op->src[0]);
            int64_t ir0 = (ith * nr0) / nth;
            int64_t ir1 = ((ith + 1) * nr0) / nth;

            ir0 = (ir0 % NB_COLS) ? ir0 + NB_COLS - (ir0 % NB_COLS) : ir0;
            ir1 = (ir1 % NB_COLS) ? ir1 + NB_COLS - (ir1 % NB_COLS) : ir1;
            ir1 = MIN(ir1, nr0);

            if (ir0 < ir1) {
                const char * src0_ptr = (const char *) src0->data;
                const char * src1_ptr = (const char *) wdata;
                char *       dst_ptr  = (char *) dst->data;
                const int64_t ncols   = ir1 - ir0;

                gemv<BLOC_TYPE, INTER_SIZE, NB_COLS, PARAM_TYPE>(
                    ne00, (float *) (dst_ptr + ir0 * nb1), ne01,
                    src0_ptr + ir0 * nb01, src1_ptr, 1, ncols);
            }
            return;
        }
