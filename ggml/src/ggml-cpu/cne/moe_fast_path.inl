        // MoE decode fast path: top-K experts, single token.
        if constexpr ((std::is_same_v<BLOC_TYPE, block_q4_K> || std::is_same_v<BLOC_TYPE, block_q6_K>) &&
                      NB_COLS == 8 && INTER_SIZE == 8) {
            if (ggml_cpu_cne_kernels_enabled() && (n_ids == 2 || n_ids == 4) && ne12 == 1 && ne11 == 1) {
                int active[4] = { -1, -1, -1, -1 };
                int n_active = 0;
                for (int cur_a = 0; cur_a < n_as && n_active < n_ids; ++cur_a) {
                    if (matrix_row_counts[cur_a] > 0) {
                        active[n_active++] = cur_a;
                    }
                }
                if (n_active == n_ids) {
                    bool ok = true;
                    for (int i = 0; i < n_ids; ++i) {
                        if (matrix_row_counts[active[i]] != 1) {
                            ok = false;
                            break;
                        }
                    }
                    if (ok) {
                        const auto * src1_col = (const char *) wdata;
                        if (n_ids == 4) {
                            const struct mmid_row_mapping row[4] = {
                                MMID_MATRIX_ROW(active[0], 0),
                                MMID_MATRIX_ROW(active[1], 0),
                                MMID_MATRIX_ROW(active[2], 0),
                                MMID_MATRIX_ROW(active[3], 0),
                            };
                            const void * src0_ptr[4];
                            float * dst_ptr[4];
                            for (int i = 0; i < 4; ++i) {
                                src0_ptr[i] = (const char *) src0->data + active[i] * nb02 + src0_cur_start * nb01;
                                dst_ptr[i] = (float *) ((char *) dst->data +
                                        (row[i].i1 * nb1 + row[i].i2 * nb2)) + src0_cur_start;
                            }
                            if constexpr (std::is_same_v<BLOC_TYPE, block_q4_K>) {
                                ggml_gemv_q4_K_8x8_q8_K_4vx(
                                        ne00, dst_ptr[0], dst_ptr[1], dst_ptr[2], dst_ptr[3], ne01,
                                        src0_ptr[0], src0_ptr[1], src0_ptr[2], src0_ptr[3],
                                        src1_col, 1, gemv_nc);
                            } else {
                                ggml_gemv_q6_K_8x8_q8_K_4vx(
                                        ne00, dst_ptr[0], dst_ptr[1], dst_ptr[2], dst_ptr[3], ne01,
                                        src0_ptr[0], src0_ptr[1], src0_ptr[2], src0_ptr[3],
                                        src1_col, 1, gemv_nc);
                            }
                            return;
                        }
                        if (n_ids == 2) {
                            const int ea = active[0];
                            const int eb = active[1];
                            const struct mmid_row_mapping row_a = MMID_MATRIX_ROW(ea, 0);
                            const struct mmid_row_mapping row_b = MMID_MATRIX_ROW(eb, 0);
                            const int64_t i1a = row_a.i1;
                            const int64_t i2a = row_a.i2;
                            const int64_t i1b = row_b.i1;
                            const int64_t i2b = row_b.i2;
                            const auto * src1_col2 = (const char *) wdata + (i1a % ne11) * nbw1 + i2a * nbw2;
                            const auto * src0_a = (const char *) src0->data + ea * nb02;
                            const auto * src0_b = (const char *) src0->data + eb * nb02;
                            float * dst_a = (float *) ((char *) dst->data + (i1a * nb1 + i2a * nb2)) + src0_cur_start;
                            float * dst_b = (float *) ((char *) dst->data + (i1b * nb1 + i2b * nb2)) + src0_cur_start;
                            if constexpr (std::is_same_v<BLOC_TYPE, block_q4_K>) {
                                ggml_gemv_q4_K_8x8_q8_K_2vx(
                                        ne00, dst_a, dst_b, ne01,
                                        src0_a + src0_cur_start * nb01,
                                        src0_b + src0_cur_start * nb01,
                                        src1_col2, 1, gemv_nc);
                            } else {
                                ggml_gemv_q6_K_8x8_q8_K_2vx(
                                        ne00, dst_a, dst_b, ne01,
                                        src0_a + src0_cur_start * nb01,
                                        src0_b + src0_cur_start * nb01,
                                        src1_col2, 1, gemv_nc);
                            }
                            return;
                        }
                    }
                }
            }
        }
