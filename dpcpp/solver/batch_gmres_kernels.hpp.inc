/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

namespace {


//          using tile_t = int[AUX_MAX_SEGMENT];
//        tile_t &swork =
//        *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(wrk_grp);

template <const bool sg_kernel_all, typename BatchMatrixType_entry,
          typename PrecType, typename ValueType>
__dpct_inline__ void initialize(
    const int restart, const BatchMatrixType_entry& a_global_entry,
    const ValueType* const b_global_entry,
    const ValueType* const x_global_entry, ValueType* const x_shared_entry,
    ValueType* const r_shared_entry, const PrecType& prec_shared,
    ValueType* const z_shared_entry, ValueType* const V_shared_entry,
    ValueType* const H_shared_entry, ValueType* const cs_shared_entry,
    ValueType* const sn_shared_entry,
    typename gko::remove_complex<ValueType>& rhs_norms,
    typename gko::remove_complex<ValueType>& res_norms,
    sycl::nd_item<3> item_ct1)
{
    auto sg = item_ct1.get_sub_group();
    const auto sg_id = sg.get_group_id();
    const auto tid = item_ct1.get_local_linear_id();
    const auto group_size = item_ct1.get_local_range().size();

    for (int i = tid; i < a_global_entry.num_rows; i += group_size) {
        x_shared_entry[i] = x_global_entry[i];
        r_shared_entry[i] = b_global_entry[i];
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // r = b - A*x
    single_advanced_matvec_kernel(static_cast<ValueType>(-1.0), a_global_entry,
                                  x_shared_entry, static_cast<ValueType>(1.0),
                                  r_shared_entry, item_ct1);
    item_ct1.barrier(sycl::access::fence_space::local_space);

    if constexpr (sg_kernel_all) {
        if (sg_id == 0) {
            compute_norm2_sg_kernel(a_global_entry.num_rows, b_global_entry,
                                    rhs_norms, item_ct1);
        } else if (sg_id == 1) {
            compute_norm2_sg_kernel(a_global_entry.num_rows, r_shared_entry,
                                    res_norms, item_ct1);
        }
    } else {
        compute_norm2_kernel(a_global_entry.num_rows, b_global_entry, rhs_norms,
                             item_ct1);
        compute_norm2_kernel(a_global_entry.num_rows, r_shared_entry, res_norms,
                             item_ct1);
    }

    // z = precond * r
    prec_shared.apply(a_global_entry.num_rows, r_shared_entry, z_shared_entry,
                      item_ct1);

    for (int li = tid; li < (restart + 1) * restart; li += group_size) {
        H_shared_entry[li] = zero<ValueType>();
    }

    for (int i = tid; i < a_global_entry.num_rows * (restart + 1);
         i += group_size) {
        V_shared_entry[i] = zero<ValueType>();
    }

    for (int li = tid; li < restart; li += group_size) {
        cs_shared_entry[li] = zero<ValueType>();
        sn_shared_entry[li] = zero<ValueType>();
    }
}


template <const bool sg_kernel_all, typename ValueType>
__dpct_inline__ void update_v_naught_and_s(
    const int num_rows, const int restart,
    const ValueType* const z_shared_entry, ValueType* const V_shared_entry,
    ValueType* const s_shared_entry,
    typename gko::remove_complex<ValueType>& tmp_norm,
    sycl::nd_item<3> item_ct1)
{
    using real_type = typename gko::remove_complex<ValueType>;

    auto sg = item_ct1.get_sub_group();
    const auto sg_id = sg.get_group_id();
    const auto tid = item_ct1.get_local_linear_id();
    const auto group_size = item_ct1.get_local_range().size();

    if constexpr (sg_kernel_all) {
        if (sg_id == 0) {
            compute_norm2_sg_kernel(num_rows, z_shared_entry, tmp_norm,
                                    item_ct1);
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
    } else
        compute_norm2_kernel(num_rows, z_shared_entry, tmp_norm, item_ct1);

    for (int li = tid; li < num_rows; li += group_size) {
        V_shared_entry[li] = z_shared_entry[li] / tmp_norm;
    }

    for (int li = tid; li < restart + 1; li += group_size) {
        s_shared_entry[li] = zero<ValueType>();
    }
    if (tid == 0) {
        s_shared_entry[0] = static_cast<ValueType>(tmp_norm);
    }
}


template <const bool sg_kernel_all, typename BatchMatrixType_entry,
          typename ValueType, typename PrecType>
__dpct_inline__ void arnoldi(const BatchMatrixType_entry& a_global_entry,
                             const int inner_iter,
                             ValueType* const __restrict__ V_shared_entry,
                             ValueType* const __restrict__ H_shared_entry,
                             const size_type h_stride,
                             ValueType* const __restrict__ w_shared_entry,
                             ValueType* const __restrict__ helper_shared_entry,
                             const PrecType& prec_shared,
                             typename gko::remove_complex<ValueType>& tmp_norm,
                             sycl::nd_item<3> item_ct1)
{
    using real_type = typename gko::remove_complex<ValueType>;
    const int num_rows = a_global_entry.num_rows;

    const ValueType* const v_i_sh = V_shared_entry + inner_iter * num_rows;

    auto sg = item_ct1.get_sub_group();
    const auto sg_id = sg.get_group_id();
    const auto tid = item_ct1.get_local_linear_id();
    const auto group_size = item_ct1.get_local_range().size();
    const auto sg_size = sg.get_local_range().size();

    single_matvec_kernel(a_global_entry, v_i_sh, helper_shared_entry, item_ct1);
    item_ct1.barrier(sycl::access::fence_space::local_space);

    prec_shared.apply(num_rows, helper_shared_entry, w_shared_entry, item_ct1);
    item_ct1.barrier(sycl::access::fence_space::local_space);

    if constexpr (sg_kernel_all) {
        for (int k = 0; k <= inner_iter; k++) {
            const ValueType* const v_k_sh = V_shared_entry + k * num_rows;
            ValueType* const h_k_i_sh =
                H_shared_entry + k * h_stride + inner_iter;

            if (sg_id == 0) {
                compute_dot_product_sg_kernel(num_rows, w_shared_entry, v_k_sh,
                                              h_k_i_sh[0], item_ct1);
            }
            item_ct1.barrier(sycl::access::fence_space::local_space);

            for (int li = tid; li < num_rows; li += group_size) {
                w_shared_entry[li] -= h_k_i_sh[0] * v_k_sh[li];
            }
            item_ct1.barrier(sycl::access::fence_space::local_space);
        }
        if (sg_id == 0) {
            compute_norm2_sg_kernel(num_rows, w_shared_entry, tmp_norm,
                                    item_ct1);
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
    } else {
        for (int k = 0; k <= inner_iter; k++) {
            const ValueType* const v_k_sh = V_shared_entry + k * num_rows;
            ValueType* const h_k_i_sh =
                H_shared_entry + k * h_stride + inner_iter;

            compute_dot_product_kernel(num_rows, w_shared_entry, v_k_sh,
                                       h_k_i_sh[0], item_ct1);

            for (int li = tid; li < num_rows; li += group_size) {
                w_shared_entry[li] -= h_k_i_sh[0] * v_k_sh[li];
            }
            item_ct1.barrier(sycl::access::fence_space::local_space);
        }
        if (sg_id == 0) {
            compute_norm2_sg_kernel(num_rows, w_shared_entry, tmp_norm,
                                    item_ct1);
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    if (tid == sg_size) {
        H_shared_entry[(inner_iter + 1) * h_stride + inner_iter] = tmp_norm;
    }

    ValueType* const v_iplus1_sh = V_shared_entry + (inner_iter + 1) * num_rows;

    for (int li = tid; li < num_rows; li += group_size) {
        v_iplus1_sh[li] = w_shared_entry[li] / tmp_norm;
    }
}


template <typename ValueType>
__dpct_inline__ void apply_plane_rotation(
    const ValueType* const __restrict__ cs,
    const ValueType* const __restrict__ sn, ValueType* const __restrict__ x,
    ValueType* const __restrict__ y, sycl::nd_item<3> item_ct1)
{
    if (item_ct1.get_local_linear_id() == 0) {
        const ValueType temp = cs[0] * x[0] + sn[0] * y[0];
        y[0] = -one<ValueType>() * conj(sn[0]) * x[0] + conj(cs[0]) * y[0];
        x[0] = temp;
    }
}


template <typename ValueType>
__dpct_inline__ void generate_and_apply_plane_rot_fused(
    ValueType* const __restrict__ cs_i, ValueType* const __restrict__ sn_i,
    ValueType* const __restrict__ H_i_i,
    ValueType* const __restrict__ H_i_plus_1_i,
    ValueType* const __restrict__ s_i, ValueType* const __restrict__ s_i_plus_1,
    sycl::nd_item<3> item_ct1)
{
    auto sg = item_ct1.get_sub_group();
    const auto sg_id = sg.get_group_id();
    const auto tid = item_ct1.get_local_linear_id();
    const auto group_size = item_ct1.get_local_range().size();

    if (tid == 0) {
        if (H_i_plus_1_i[0] == zero<ValueType>()) {
            cs_i[0] = one<ValueType>();
            sn_i[0] = zero<ValueType>();
        } else if (abs(H_i_plus_1_i[0]) > abs(H_i_i[0])) {
            const ValueType temp = H_i_i[0] / H_i_plus_1_i[0];
            sn_i[0] = (one<ValueType>() * conj(H_i_plus_1_i[0])) /
                      (sqrt(one<ValueType>() + temp * temp) * H_i_plus_1_i[0]);
            cs_i[0] = (conj(H_i_i[0]) / conj(H_i_plus_1_i[0])) * sn_i[0];

        } else {
            const ValueType temp = H_i_plus_1_i[0] / H_i_i[0];
            cs_i[0] = (one<ValueType>() * conj(H_i_i[0])) /
                      (sqrt(one<ValueType>() + temp * temp) * H_i_i[0]);
            sn_i[0] = (conj(H_i_plus_1_i[0]) / conj(H_i_i[0])) * cs_i[0];
        }

        const ValueType temp = cs_i[0] * s_i[0];
        s_i_plus_1[0] = -one<ValueType>() * conj(sn_i[0]) * s_i[0];
        s_i[0] = temp;
        H_i_i[0] = cs_i[0] * H_i_i[0] + sn_i[0] * H_i_plus_1_i[0];
        H_i_plus_1_i[0] = zero<ValueType>();
    }
}


template <typename ValueType>
__dpct_inline__ void update_x(
    const int num_rows, const int m,
    const ValueType* const __restrict__ H_shared_entry,
    const size_type h_stride,
    const ValueType* const __restrict__ s_shared_entry,
    const ValueType* const __restrict__ V_shared_entry,
    ValueType* const __restrict__ x_shared_entry,
    ValueType* const __restrict__ y_shared_entry, sycl::nd_item<3> item_ct1)
{
    auto sg = item_ct1.get_sub_group();
    const auto sg_id = sg.get_group_id();
    const auto tid = item_ct1.get_local_linear_id();
    const auto group_size = item_ct1.get_local_range().size();

    // triangular solve
    if (tid == 0) {
        for (int r = m; r >= 0; r--) {
            ValueType temp_sum = zero<ValueType>();
            for (int t = m; t > r; t--) {
                temp_sum +=
                    H_shared_entry[r * h_stride + t] * y_shared_entry[t];
            }

            y_shared_entry[r] = (s_shared_entry[r] - temp_sum) /
                                H_shared_entry[r * h_stride + r];
        }
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // dense mat * vec multiplication
    for (int li = tid; li < num_rows; li += group_size) {
        for (int a = 0; a < m + 1; a++) {
            x_shared_entry[li] +=
                V_shared_entry[a * num_rows + li] * y_shared_entry[a];
        }
    }
}


}  // unnamed namespace


template <typename StopType, const bool vecs_shared_all,
          const bool sg_kernel_all, typename PrecType, typename LogType,
          typename BatchMatrixType, typename ValueType>
__dpct_inline__ void apply_kernel(
    const gko::kernels::batch_gmres::StorageConfig sconf, const int max_iter,
    const gko::remove_complex<ValueType> tol, const int restart, LogType logger,
    PrecType prec_shared, const BatchMatrixType a_global_entry,
    const ValueType* const __restrict__ b_global_entry,
    ValueType* const __restrict__ x_global_entry, const int nrows,
    const int nnz, ValueType* local_mem_sh, sycl::nd_item<3> item_ct1,
    ValueType* const workspace = nullptr)
{
    using real_type = typename gko::remove_complex<ValueType>;

    sycl::group group = item_ct1.get_group();
    auto sg = item_ct1.get_sub_group();
    const auto sg_id = sg.get_group_id();
    const auto tid = item_ct1.get_local_linear_id();
    const auto group_size = item_ct1.get_local_range().size();
    const auto ibatch = item_ct1.get_group_linear_id();

    const int gmem_offset =
        ibatch * sconf.gmem_stride_bytes / sizeof(ValueType);

    ValueType* prec_work_sh;
    ValueType* r_sh;
    ValueType* z_sh;
    ValueType* w_sh;
    ValueType* x_sh;
    ValueType* helper_sh;
    ValueType* cs_sh;
    ValueType* sn_sh;
    ValueType* y_sh;
    ValueType* s_sh;
    ValueType* H_sh;
    ValueType* V_sh;

    if constexpr (vecs_shared_all) {
        prec_work_sh = local_mem_sh;
        r_sh = prec_work_sh + PrecType::dynamic_work_size(nrows, nnz);
        z_sh = r_sh + sconf.padded_vec_len;
        w_sh = z_sh + sconf.padded_vec_len;
        x_sh = w_sh + sconf.padded_vec_len;
        helper_sh = x_sh + sconf.padded_vec_len;
        cs_sh = helper_sh + sconf.padded_vec_len;
        sn_sh = cs_sh + restart;
        y_sh = sn_sh + restart;
        s_sh = y_sh + restart;
        V_sh = s_sh + (restart + 1);
        H_sh = V_sh + nrows * (restart + 1);
    } else {
        if (sconf.prec_shared) {
            prec_work_sh = local_mem_sh;
        } else {
            prec_work_sh = workspace + gmem_offset;
        }
        if (sconf.n_shared >= 1 && sconf.prec_shared) {
            r_sh = prec_work_sh + PrecType::dynamic_work_size(nrows, nnz);
        } else if (sconf.n_shared == 0 && sconf.prec_shared) {
            r_sh = workspace + gmem_offset;
        } else {
            r_sh = prec_work_sh + PrecType::dynamic_work_size(nrows, nnz);
        }
        if (sconf.n_shared == 1) {
            z_sh = workspace + gmem_offset;
        } else {
            z_sh = r_sh + sconf.padded_vec_len;
        }
        if (sconf.n_shared == 2) {
            w_sh = workspace + gmem_offset;
        } else {
            w_sh = z_sh + sconf.padded_vec_len;
        }
        if (sconf.n_shared == 3) {
            x_sh = workspace + gmem_offset;
        } else {
            x_sh = w_sh + sconf.padded_vec_len;
        }
        if (sconf.n_shared == 4) {
            helper_sh = workspace + gmem_offset;
        } else {
            helper_sh = x_sh + sconf.padded_vec_len;
        }
        if (sconf.n_shared == 5 && sconf.rot_shared) {
            cs_sh = helper_sh + sconf.padded_vec_len;
            sn_sh = cs_sh + restart;
            y_sh = sn_sh + restart;
            s_sh = y_sh + restart;
        } else {
            cs_sh = workspace + gmem_offset;
            sn_sh = cs_sh + restart;
            y_sh = sn_sh + restart;
            s_sh = y_sh + restart;
        }
        if (sconf.n_shared == 5 && sconf.subspace_shared) {
            V_sh = s_sh + (restart + 1);
        } else {
            V_sh = workspace + gmem_offset;
        }
        if (sconf.n_shared == 5 && sconf.hess_shared) {
            H_sh = V_sh + nrows * (restart + 1);
        } else {
            H_sh = workspace + gmem_offset;
        }
    }

    real_type* norms_rhs_sh;
    real_type* norms_res_sh;
    real_type* norms_tmp_sh;
    if constexpr (sg_kernel_all) {
        using tile_real_t = real_type[3];
        tile_real_t& norms_sh =
            *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_real_t>(
                group);
        norms_rhs_sh = &norms_sh[0];
        norms_res_sh = &norms_sh[1];
        norms_tmp_sh = &norms_sh[2];
    } else {
        real_type norms_sh[3];
        norms_rhs_sh = &norms_sh[0];
        norms_res_sh = &norms_sh[1];
        norms_tmp_sh = &norms_sh[2];
    }

    // Stride of Hessenberg matrix
    const auto h_stride = static_cast<size_type>(restart);

    prec_shared.generate(ibatch, a_global_entry, prec_work_sh, item_ct1);

    // initialization
    // compute b norms
    // r = b - A*x
    // z = precond*r
    // compute residual norms
    // initialize V, H, cs, sn with zeroes
    initialize<sg_kernel_all>(restart, a_global_entry, b_global_entry,
                              x_global_entry, x_sh, r_sh, prec_shared, z_sh,
                              V_sh, H_sh, cs_sh, sn_sh, norms_rhs_sh[0],
                              norms_res_sh[0], item_ct1);
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // stopping criterion object
    StopType stop(tol, norms_rhs_sh);

    int total_iter = 0;

    // Note: restart - inner iterations and the outer iteration:  are
    // counted as (restart + 1) number of iterations instead of one.
    while (1) {
        if (stop.check_converged(norms_res_sh) || total_iter >= max_iter) {
            break;
        }

        // KrylovBasis_0 = z/norm(z)
        // s -> fill with zeroes
        // s(0) = norm(z)
        update_v_naught_and_s<sg_kernel_all>(nrows, restart, z_sh, V_sh, s_sh,
                                             norms_tmp_sh[0], item_ct1);
        item_ct1.barrier(sycl::access::fence_space::local_space);

        bool inner_loop_break_flag = false;

        for (int inner_iter = 0; inner_iter < restart; inner_iter++) {
            // w_temp = A * v_i
            // w = precond * w_temp
            // i = inner_iter
            // for k = 0 to inner_iter
            //     Hessenburg(k,i) =  w' * v_k
            //     w = w - Hessenburg(k,i) * v_k
            // end
            // Hessenburg(i+1, i) = norm(w)
            // KrylovBasis_i+1 = w / Hessenburg(i+1,i)
            arnoldi<sg_kernel_all>(a_global_entry, inner_iter, V_sh, H_sh,
                                   h_stride, w_sh, helper_sh, prec_shared,
                                   norms_tmp_sh[0], item_ct1);
            item_ct1.barrier(sycl::access::fence_space::local_space);

            for (int k = 0; k < inner_iter; k++) {
                // temp = cs(k) * Hessenberg( k, inner_iter )  +   sn(k) *
                // Hessenberg(k + 1, inner_iter)
                // Hessenberg(k + 1, inner_iter) = -1 * conj(sn(k)) *
                // Hessenberg( k , inner_iter) + conj(cs(k)) * Hessenberg(k
                // + 1 , inner_iter) Hessenberg(k,inner_iter) = temp
                apply_plane_rotation(
                    &cs_sh[k], &sn_sh[k], &H_sh[k * h_stride + inner_iter],
                    &H_sh[(k + 1) * h_stride + inner_iter], item_ct1);
                item_ct1.barrier(sycl::access::fence_space::local_space);
            }

            // compute sine and cos
            // temp = cs(inner_iter) * s(inner_iter)
            // s(inner_iter + 1) = -1 * conj(sn(inner_iter)) * s(inner_iter)
            // s(inner_iter) = temp
            // Hessenberg(inner_iter , inner_iter) = cs(inner_iter) *
            // Hessenberg(inner_iter , inner_iter) + sn(inner_iter) *
            // Hessenberg(inner_iter + 1, inner_iter)
            // Hessenberg(inner_iter + 1, inner_iter) = 0
            generate_and_apply_plane_rot_fused(
                &cs_sh[inner_iter], &sn_sh[inner_iter],
                &H_sh[inner_iter * h_stride + inner_iter],
                &H_sh[(inner_iter + 1) * h_stride + inner_iter],
                &s_sh[inner_iter], &s_sh[(inner_iter + 1)], item_ct1);
            item_ct1.barrier(sycl::access::fence_space::local_space);

            // estimate of residual norms
            if constexpr (sg_kernel_all) {
                if (tid == 0) {
                    norms_res_sh[0] = abs(s_sh[(inner_iter + 1)]);
                }
                item_ct1.barrier(sycl::access::fence_space::local_space);
            } else
                norms_res_sh[0] = abs(s_sh[(inner_iter + 1)]);

            total_iter++;

            if (stop.check_converged(norms_res_sh) || total_iter >= max_iter) {
                update_x(nrows, inner_iter, H_sh, h_stride, s_sh, V_sh, x_sh,
                         y_sh, item_ct1);
                inner_loop_break_flag = true;
                break;
            }
        }

        if (inner_loop_break_flag == true) {
            break;
        }

        // y = Hessenburg(0:restart - 1,0:restart - 1) \ s(0:restart-1)
        // x = x + KrylovBasis(:,0 : restart - 1) * y
        update_x(nrows, restart - 1, H_sh, h_stride, s_sh, V_sh, x_sh, y_sh,
                 item_ct1);
        item_ct1.barrier(sycl::access::fence_space::local_space);

        // r = b
        copy_kernel(nrows, b_global_entry, r_sh, item_ct1);
        item_ct1.barrier(sycl::access::fence_space::local_space);
        // r = r - A*x
        single_advanced_matvec_kernel(
            static_cast<ValueType>(-1.0), a_global_entry, x_sh,
            static_cast<ValueType>(1.0), r_sh, item_ct1);
        item_ct1.barrier(sycl::access::fence_space::local_space);

        if constexpr (sg_kernel_all) {
            if (sg_id == 0) {
                compute_norm2_sg_kernel(nrows, r_sh, norms_res_sh[0], item_ct1);
            }
        } else
            compute_norm2_kernel(nrows, r_sh, norms_res_sh[0], item_ct1);

        prec_shared.apply(nrows, r_sh, z_sh, item_ct1);
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    // copy x back to global memory
    copy_kernel(nrows, x_sh, x_global_entry, item_ct1);

    logger.log_iteration(ibatch, total_iter, norms_res_sh[0]);
}

/*
template <typename StopType, typename PrecType, typename LogType,
          typename BatchMatrixType, typename ValueType>
__dpct_inline__ void small_apply_kernel(
    const gko::kernels::batch_gmres::StorageConfig sconf, const int max_iter,
    const gko::remove_complex<ValueType> tol, const int restart, LogType logger,
    PrecType prec_shared, const BatchMatrixType a_global_entry,
    const ValueType* const __restrict__ b_global_entry,
    ValueType* const __restrict__ x_global_entry, const int nrows,
    const int nnz, ValueType* local_mem_sh, sycl::nd_item<3> item_ct1,
    ValueType* const workspace = nullptr)
{
    using real_type = typename gko::remove_complex<ValueType>;

    auto group = item_ct1.get_group();
    auto sg = item_ct1.get_sub_group();
    const auto sg_id = sg.get_group_id();
    const auto tid = item_ct1.get_local_linear_id();
    const auto group_size = item_ct1.get_local_range().size();
    const auto ibatch = item_ct1.get_group_linear_id();

    ValueType* const prec_work_sh = local_mem_sh;

    ValueType* const r_sh =
        prec_work_sh + PrecType::dynamic_work_size(nrows, nnz);
    ValueType* const z_sh = r_sh + sconf.padded_vec_len;
    ValueType* const w_sh = z_sh + sconf.padded_vec_len;
    ValueType* const x_sh = w_sh + sconf.padded_vec_len;
    ValueType* const helper_sh = x_sh + sconf.padded_vec_len;

    ValueType* const cs_sh = helper_sh + sconf.padded_vec_len;
    ValueType* const sn_sh = cs_sh + restart;
    ValueType* const y_sh = sn_sh + restart;
    ValueType* const s_sh = y_sh + restart;

    // Krylov subspace basis vectors
    // storage:row-major order , subspace basis vectors corr. to each rhs
    // are stored in a single col. one after the other-(on paper). This
    // allows for efficient memory access And to get vi : that is ith basis
    // vector for each rhs: vi_entry{  &V[i* V_entry.stride * nrows],
    // V_entry.stride , nrows, nrhs}; So if nrhs=1, effectively the cols.
    // are stored contiguously in memory one after the other.
    // const gko::batch_dense::BatchEntry<ValueType> V_shared_entry{
    //     V_sh, static_cast<size_type>(nrhs), nrows * (restart + 1), nrhs};
    ValueType* const V_sh = s_sh + (restart + 1);

    // Hessenberg matrix
    // const gko::batch_dense::BatchEntry<ValueType> H_shared_entry{
    //     H_sh, static_cast<size_type>(nrhs * restart), restart + 1,
    //     nrhs * restart};
    // storage:row-major ,  entry (i,j) for different RHSs are placed after
    // the other in a row - when drawn on paper, (and the same is true for
    // actual storage as the storage order is row-major) to get entry (i,j)
    // for rhs: rhs_k , H_entry.stride*i + j*nrhs  + rhs_k
    ValueType* const H_sh = V_sh + nrows * (restart + 1);


    // real_type *const norms_rhs_sh = reinterpret_cast<real_type *>(
    //     prec_work_sh + PrecType::dynamic_work_size(nrows, nnz));
    // real_type *const norms_res_sh = norms_rhs_sh + nrhs;
    // real_type *const norms_tmp_sh = norms_res_sh + nrhs;
    using tile_t_small = real_type[2];
    tile_t_small& norms_sh =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t_small>(
            group);
    real_type* norms_rhs_sh = &norms_sh[0];
    real_type* norms_res_sh = &norms_sh[1];

    // Stride of Hessenberg matrix
    const auto h_stride = static_cast<size_type>(restart);

    prec_shared.generate(ibatch, a_global_entry, prec_work_sh, item_ct1);

    // initialization
    // compute b norms
    // r = b - A*x
    // z = precond*r
    // compute residual norms
    // initialize V, H, cs, sn with zeroes
    initialize(restart, a_global_entry, b_global_entry, x_global_entry, x_sh,
               r_sh, prec_shared, z_sh, V_sh, H_sh, cs_sh, sn_sh,
               norms_rhs_sh[0], norms_res_sh[0], item_ct1);
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // stopping criterion object
    StopType stop(tol, norms_rhs_sh);

    int total_iter = 0;

    // Note: restart - inner iterations and the outer iteration:  are
    // counted as (restart + 1) number of iterations instead of one.
    while (1) {
        if (stop.check_converged(norms_res_sh) || total_iter >= max_iter) {
            break;
        }

        // KrylovBasis_0 = z/norm(z)
        // s -> fill with zeroes
        // s(0) = norm(z)
        update_v_naught_and_s(nrows, restart, z_sh, V_sh, s_sh, item_ct1);
        item_ct1.barrier(sycl::access::fence_space::local_space);

        bool inner_loop_break_flag = false;

        for (int inner_iter = 0; inner_iter < restart; inner_iter++) {
            // w_temp = A * v_i
            // w = precond * w_temp
            // i = inner_iter
            // for k = 0 to inner_iter
            //     Hessenburg(k,i) =  w' * v_k
            //     w = w - Hessenburg(k,i) * v_k
            // end
            // Hessenburg(i+1, i) = norm(w)
            // KrylovBasis_i+1 = w / Hessenburg(i+1,i)
            arnoldi(a_global_entry, inner_iter, V_sh, H_sh, h_stride, w_sh,
                    helper_sh, prec_shared, item_ct1);
            item_ct1.barrier(sycl::access::fence_space::local_space);

            for (int k = 0; k < inner_iter; k++) {
                // temp = cs(k) * Hessenberg( k, inner_iter )  +   sn(k) *
                // Hessenberg(k + 1, inner_iter)
                // Hessenberg(k + 1, inner_iter) = -1 * conj(sn(k)) *
                // Hessenberg( k , inner_iter) + conj(cs(k)) * Hessenberg(k
                // + 1 , inner_iter) Hessenberg(k,inner_iter) = temp
                apply_plane_rotation(
                    &cs_sh[k], &sn_sh[k], &H_sh[k * h_stride + inner_iter],
                    &H_sh[(k + 1) * h_stride + inner_iter], item_ct1);
                item_ct1.barrier(sycl::access::fence_space::local_space);
            }

            // compute sine and cos
            // temp = cs(inner_iter) * s(inner_iter)
            // s(inner_iter + 1) = -1 * conj(sn(inner_iter)) * s(inner_iter)
            // s(inner_iter) = temp
            // Hessenberg(inner_iter , inner_iter) = cs(inner_iter) *
            // Hessenberg(inner_iter , inner_iter) + sn(inner_iter) *
            // Hessenberg(inner_iter + 1, inner_iter)
            // Hessenberg(inner_iter + 1, inner_iter) = 0
            generate_and_apply_plane_rot_fused(
                &cs_sh[inner_iter], &sn_sh[inner_iter],
                &H_sh[inner_iter * h_stride + inner_iter],
                &H_sh[(inner_iter + 1) * h_stride + inner_iter],
                &s_sh[inner_iter], &s_sh[(inner_iter + 1)], item_ct1);
            item_ct1.barrier(sycl::access::fence_space::local_space);

            // estimate of residual norms
            if (tid == 0) {
                norms_res_sh[0] = abs(s_sh[(inner_iter + 1)]);
            }
            item_ct1.barrier(sycl::access::fence_space::local_space);

            total_iter++;

            if (stop.check_converged(norms_res_sh) || total_iter >= max_iter) {
                update_x(nrows, inner_iter, H_sh, h_stride, s_sh, V_sh, x_sh,
                         y_sh, item_ct1);
                inner_loop_break_flag = true;
                break;
            }
        }

        if (inner_loop_break_flag == true) {
            break;
        }

        // y = Hessenburg(0:restart - 1,0:restart - 1) \ s(0:restart-1)
        // x = x + KrylovBasis(:,0 : restart - 1) * y
        update_x(nrows, restart - 1, H_sh, h_stride, s_sh, V_sh, x_sh, y_sh,
                 item_ct1);
        item_ct1.barrier(sycl::access::fence_space::local_space);

        // r = b
        copy_kernel(nrows, b_global_entry, r_sh, item_ct1);
        item_ct1.barrier(sycl::access::fence_space::local_space);
        // r = r - A*x
        single_advanced_matvec_kernel(
            static_cast<ValueType>(-1.0), a_global_entry, x_sh,
            static_cast<ValueType>(1.0), r_sh, item_ct1);
        item_ct1.barrier(sycl::access::fence_space::local_space);

        if constexpr(sg_kernel_all){
            if (sg_id == 0) {
                compute_norm2_sg_kernel(nrows, r_sh, norms_res_sh[0], item_ct1);
            }
        }

        prec_shared.apply(nrows, r_sh, z_sh, item_ct1);
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    // copy x back to global memory
    copy_kernel(nrows, x_sh, x_global_entry, item_ct1);

    logger.log_iteration(ibatch, total_iter, norms_res_sh[0]);
}
*/
