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

template <const bool sg_kernels_only, typename PrecType, typename ValueType,
          typename BatchMatrixType>
__dpct_inline__ void initialize(
    const int num_rows, const BatchMatrixType& A_global_entry,
    const ValueType* const __restrict__ b_global_entry,
    const ValueType* const __restrict__ x_global_entry,
    ValueType* const __restrict__ x_shared_entry,
    ValueType* const __restrict__ r_shared_entry, const PrecType& prec_shared,
    ValueType* const __restrict__ z_shared_entry, ValueType& rho_old,
    ValueType* const __restrict__ p_shared_entry,
    gko::remove_complex<ValueType>& rhs_norms, sycl::nd_item<3> item_ct1)
{
    auto sg = item_ct1.get_sub_group();
    auto group = item_ct1.get_group();
    const auto sg_id = sg.get_group_id();
    const auto tid = item_ct1.get_local_linear_id();
    const auto group_size = item_ct1.get_local_range().size();
    // copy x from global to shared memory
    // r = b
    for (int iz = tid; iz < num_rows; iz += group_size) {
        x_shared_entry[iz] = x_global_entry[iz];
        r_shared_entry[iz] = b_global_entry[iz];
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // r = b - A*x
    single_advanced_matvec_kernel(static_cast<ValueType>(-1.0), A_global_entry,
                                  x_shared_entry, static_cast<ValueType>(1.0),
                                  r_shared_entry, item_ct1);
    item_ct1.barrier(sycl::access::fence_space::local_space);


    // z = precond * r
    prec_shared.apply(num_rows, r_shared_entry, z_shared_entry, item_ct1);
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Compute norms of rhs
    // and rho_old = r' * z
    if constexpr (sg_kernels_only) {
        if (sg_id == 0)
            compute_norm2_sg_kernel(num_rows, b_global_entry, rhs_norms,
                                    item_ct1);
        else if (sg_id == 1)
            compute_dot_product_sg_kernel(num_rows, r_shared_entry,
                                          z_shared_entry, rho_old, item_ct1);
        item_ct1.barrier(sycl::access::fence_space::local_space);
    } else {
        compute_norm2_kernel(num_rows, b_global_entry, rhs_norms, item_ct1);
        compute_dot_product_kernel(num_rows, r_shared_entry, z_shared_entry,
                                   rho_old, item_ct1);
    }

    // p = z
    for (int iz = tid; iz < num_rows; iz += group_size) {
        p_shared_entry[iz] = z_shared_entry[iz];
    }
}


template <typename ValueType>
__dpct_inline__ void update_p(
    const int num_rows, const ValueType& rho_new_shared_entry,
    const ValueType& rho_old_shared_entry,
    const ValueType* const __restrict__ z_shared_entry,
    ValueType* const __restrict__ p_shared_entry, sycl::nd_item<3> item_ct1)
{
    const ValueType beta = rho_new_shared_entry / rho_old_shared_entry;
    for (int li = item_ct1.get_local_linear_id(); li < num_rows;
         li += item_ct1.get_local_range().size()) {
        p_shared_entry[li] = z_shared_entry[li] + beta * p_shared_entry[li];
    }
}

template <const int sg_kernels_only, typename ValueType>
__dpct_inline__ void update_x_and_r(
    const int num_rows, const ValueType rho_old_shared_entry,
    const ValueType* const __restrict__ p_shared_entry,
    const ValueType* const __restrict__ Ap_shared_entry,
    ValueType& alpha_shared_entry, ValueType* const __restrict__ x_shared_entry,
    ValueType* const __restrict__ r_shared_entry, sycl::nd_item<3> item_ct1)
{
    auto group = item_ct1.get_group();
    auto sg = item_ct1.get_sub_group();
    if constexpr (sg_kernels_only) {
        if (sg.get_group_id() == 0)
            compute_dot_product_sg_kernel(num_rows, p_shared_entry,
                                          Ap_shared_entry, alpha_shared_entry,
                                          item_ct1);
        item_ct1.barrier(sycl::access::fence_space::local_space);
    } else {
        compute_dot_product_kernel(num_rows, p_shared_entry, Ap_shared_entry,
                                   alpha_shared_entry, item_ct1);
    }
    alpha_shared_entry = rho_old_shared_entry / alpha_shared_entry;

    for (int li = item_ct1.get_local_linear_id(); li < num_rows;
         li += item_ct1.get_local_range().size()) {
        x_shared_entry[li] += alpha_shared_entry * p_shared_entry[li];
        r_shared_entry[li] -= alpha_shared_entry * Ap_shared_entry[li];
    }
}


}  // namespace


template <typename StopType, const bool vecs_shared_all,
          const bool sg_kernels_only, typename PrecType, typename LogType,
          typename BatchMatrixType, typename ValueType>
__dpct_inline__ void apply_kernel(
    const gko::kernels::batch_cg::StorageConfig sconf, const int max_iter,
    const gko::remove_complex<ValueType> tol, LogType logger,
    PrecType prec_shared, const BatchMatrixType& A_global_entry,
    const ValueType* const __restrict__ b_global_entry,
    ValueType* const __restrict__ x_global_entry, const size_type nrows,
    const size_type nnz, ValueType* const __restrict__ slm_values,
    sycl::nd_item<3> item_ct1,
    ValueType* const __restrict__ workspace = nullptr)
{
    using real_type = typename gko::remove_complex<ValueType>;

    const auto sg = item_ct1.get_sub_group();
    const int sg_id = sg.get_group_id();
    const int sg_size = sg.get_local_range().size();
    const int num_sg = sg.get_group_range().size();

    const auto group = item_ct1.get_group();
    const auto ibatch = item_ct1.get_group_linear_id();

    // The whole workgroup have the same values for these variables, but these
    // variables are stored in reg. mem, not on SLM
    ValueType* rho_old_sh;
    ValueType* rho_new_sh;
    ValueType* alpha_sh;
    real_type* norms_rhs_sh;
    real_type* norms_res_sh;
    if constexpr (sg_kernels_only) {
        using tile_value_t = ValueType[3];
        tile_value_t& values =
            *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_value_t>(
                group);
        using tile_real_t = real_type[3];
        tile_real_t& reals =
            *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_real_t>(
                group);
        rho_old_sh = &values[0];
        rho_new_sh = &values[1];
        alpha_sh = &values[2];
        norms_rhs_sh = &reals[0];
        norms_res_sh = &reals[1];
    } else {
        ValueType values[3];
        real_type reals[2];
        rho_old_sh = &values[0];
        rho_new_sh = &values[1];
        alpha_sh = &values[2];
        norms_rhs_sh = &reals[0];
        norms_res_sh = &reals[1];
    }
    const int gmem_offset =
        ibatch * sconf.gmem_stride_bytes / sizeof(ValueType);
    ValueType* r_sh;
    ValueType* z_sh;
    ValueType* p_sh;
    ValueType* Ap_sh;
    ValueType* x_sh;
    ValueType* prec_work_sh;
    if constexpr (vecs_shared_all) {
        r_sh = slm_values;
        z_sh = r_sh + sconf.padded_vec_len;
        p_sh = z_sh + sconf.padded_vec_len;
        Ap_sh = p_sh + sconf.padded_vec_len;
        prec_work_sh = Ap_sh + sconf.padded_vec_len;
        x_sh = prec_work_sh + PrecType::dynamic_work_size(nrows, nnz);
    } else {
        if (sconf.n_shared >= 1) {
            r_sh = slm_values;
        } else {
            r_sh = workspace + gmem_offset;
        }
        if (sconf.n_shared == 1) {
            z_sh = workspace + gmem_offset;
        } else {
            z_sh = r_sh + sconf.padded_vec_len;
        }
        if (sconf.n_shared == 2) {
            p_sh = workspace + gmem_offset;
        } else {
            p_sh = z_sh + sconf.padded_vec_len;
        }
        if (sconf.n_shared == 3) {
            Ap_sh = workspace + gmem_offset;
        } else {
            Ap_sh = p_sh + sconf.padded_vec_len;
        }
        if (!sconf.prec_shared && sconf.n_shared == 4) {
            prec_work_sh = workspace + gmem_offset;
        } else {
            prec_work_sh = Ap_sh + sconf.padded_vec_len;
        }
        if (sconf.n_shared == 4 && sconf.prec_shared) {
            x_sh = workspace + gmem_offset;
        } else {
            x_sh = prec_work_sh + PrecType::dynamic_work_size(nrows, nnz);
        }
    }

    // generate preconditioner
    prec_shared.generate(ibatch, A_global_entry, prec_work_sh, item_ct1);

    // initialization
    // compute b norms
    // r = b - A*x
    // z = precond*r
    // rho_old = r' * z (' is for hermitian transpose)
    // p = z
    initialize<sg_kernels_only>(nrows, A_global_entry, b_global_entry,
                                x_global_entry, x_sh, r_sh, prec_shared, z_sh,
                                rho_old_sh[0], p_sh, norms_rhs_sh[0], item_ct1);
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // stopping criterion object
    StopType stop(tol, norms_rhs_sh);
    norms_res_sh[0] = sqrt(abs(rho_old_sh[0]));
    if (stop.check_converged(norms_res_sh)) {
        return;
    }

    int iter = 0;
    for (; iter < max_iter; iter++) {
        // Ap = A * p
        single_matvec_kernel(A_global_entry, p_sh, Ap_sh, item_ct1);
        item_ct1.barrier(sycl::access::fence_space::local_space);

        // alpha = rho_old / (p' * Ap)
        // x = x + alpha * p
        // r = r - alpha * Ap
        update_x_and_r<sg_kernels_only>(nrows, rho_old_sh[0], p_sh, Ap_sh,
                                        alpha_sh[0], x_sh, r_sh, item_ct1);
        item_ct1.barrier(sycl::access::fence_space::local_space);


        // z = precond * r
        prec_shared.apply(nrows, r_sh, z_sh, item_ct1);
        item_ct1.barrier(sycl::access::fence_space::local_space);

        //  rho_new =  (r)' * (z)
        if constexpr (sg_kernels_only) {
            if (sg_id == 0)
                compute_dot_product_sg_kernel(nrows, r_sh, z_sh, rho_new_sh[0],
                                              item_ct1);
            item_ct1.barrier(sycl::access::fence_space::local_space);
        } else {
            compute_dot_product_kernel(nrows, r_sh, z_sh, rho_new_sh[0],
                                       item_ct1);
        }
        norms_res_sh[0] = sqrt(abs(rho_new_sh[0]));
        if (stop.check_converged(norms_res_sh)) {
            break;
        }

        // beta = rho_new / rho_old
        // p = z + beta * p
        update_p(nrows, rho_new_sh[0], rho_old_sh[0], z_sh, p_sh, item_ct1);
        rho_old_sh[0] = rho_new_sh[0];
        // item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    logger.log_iteration(ibatch, iter, norms_res_sh[0]);

    // copy x back to global memory
    copy_kernel(nrows, x_sh, x_global_entry, item_ct1);
    item_ct1.barrier(sycl::access::fence_space::local_space);
}
