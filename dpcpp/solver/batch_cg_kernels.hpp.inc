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


template <typename BatchMatrixType_entry, typename PrecType, typename ValueType>
__dpct_inline__ void initialize(
    const int num_rows, const BatchMatrixType_entry& A_global_entry,
    const ValueType* const b_global_entry,
    const ValueType* const x_global_entry, ValueType* const x_shared_entry,
    ValueType* const r_shared_entry, const PrecType& prec_shared,
    ValueType* const z_shared_entry, ValueType& rho_old,
    ValueType* const p_shared_entry, gko::remove_complex<ValueType>& rhs_norms,
    sycl::nd_item<3> item_ct1)
{
    auto sg = item_ct1.get_sub_group();
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

    //    if (sg_id == 0) {
    // Compute norms of rhs
    compute_norm2_kernel(num_rows, b_global_entry, rhs_norms, item_ct1);
    //    } else if (sg_id == 1) {
    // rho_old = r' * z
    compute_dot_product_kernel(num_rows, r_shared_entry, z_shared_entry,
                               rho_old, item_ct1);
    //}
    // item_ct1.barrier(sycl::access::fence_space::local_space);
    // p = z
    for (int iz = tid; iz < num_rows; iz += group_size) {
        p_shared_entry[iz] = z_shared_entry[iz];
    }
    //    item_ct1.barrier(sycl::access::fence_space::local_space);
}


template <typename ValueType>
__dpct_inline__ void update_p(const int num_rows,
                              const ValueType& rho_new_shared_entry,
                              const ValueType& rho_old_shared_entry,
                              const ValueType* const z_shared_entry,
                              ValueType* const p_shared_entry,
                              sycl::nd_item<3> item_ct1)
{
    const ValueType beta = rho_new_shared_entry / rho_old_shared_entry;
    for (int li = item_ct1.get_local_linear_id(); li < num_rows;
         li += item_ct1.get_local_range().size()) {
        p_shared_entry[li] = z_shared_entry[li] + beta * p_shared_entry[li];
    }
}


template <typename ValueType>
__dpct_inline__ void update_x_and_r(const int num_rows,
                                    const ValueType& rho_old_shared_entry,
                                    const ValueType* const p_shared_entry,
                                    const ValueType* const Ap_shared_entry,
                                    ValueType* const x_shared_entry,
                                    ValueType* const r_shared_entry,
                                    sycl::nd_item<3> item_ct1)
{
    auto sg = item_ct1.get_sub_group();
    const auto sg_id = sg.get_group_id();

    ValueType alpha;
    compute_dot_product_kernel(num_rows, p_shared_entry, Ap_shared_entry, alpha,
                               item_ct1);
    alpha = rho_old_shared_entry / alpha;

    for (int li = item_ct1.get_local_linear_id(); li < num_rows;
         li += item_ct1.get_local_range().size()) {
        x_shared_entry[li] += alpha * p_shared_entry[li];
        r_shared_entry[li] -= alpha * Ap_shared_entry[li];
    }
}


}  // namespace


template <typename StopType, typename PrecType, typename LogType,
          typename BatchMatrixType, typename ValueType>
void small_apply_kernel(const gko::kernels::batch_cg::StorageConfig sconf,
                        const int max_iter,
                        const gko::remove_complex<ValueType> tol,
                        LogType logger, PrecType prec_shared,
                        const BatchMatrixType A_global_entry,
                        const ValueType* const __restrict__ b_global_entry,
                        ValueType* const __restrict__ x_global_entry,
                        const size_type nrows, const size_type nnz,
                        ValueType* slm_values, sycl::nd_item<3> item_ct1,
                        ValueType* const __restrict__ workspace = nullptr)
{
    using real_type = typename gko::remove_complex<ValueType>;

    const auto sg = item_ct1.get_sub_group();
    const int sg_id = sg.get_group_id();
    const int sg_size = sg.get_local_range().size();
    const int num_sg = sg.get_group_range().size();

    const auto ibatch = item_ct1.get_group_linear_id();

    // The whole workgroup have the same values for these variables, but these
    // variables are stored in reg. mem, not on SLM
    ValueType rho_old_sh;
    ValueType rho_new_sh;
    remove_complex<ValueType> norms_rhs_sh;
    remove_complex<ValueType> norms_res_sh;

    const int gmem_offset =
        ibatch * sconf.gmem_stride_bytes / sizeof(ValueType);
    ValueType* r_sh;
    ValueType* z_sh;
    ValueType* p_sh;
    ValueType* Ap_sh;
    ValueType* x_sh;
    ValueType* prec_work_sh;
    r_sh = slm_values;
    z_sh = r_sh + sconf.padded_vec_len;
    p_sh = z_sh + sconf.padded_vec_len;
    Ap_sh = p_sh + sconf.padded_vec_len;
    prec_work_sh = Ap_sh + sconf.padded_vec_len;
    x_sh = prec_work_sh + PrecType::dynamic_work_size(nrows, nnz);
    // generate preconditioner
    prec_shared.generate(ibatch, A_global_entry, prec_work_sh, item_ct1);

    // initialization
    // compute b norms
    // r = b - A*x
    // z = precond*r
    // rho_old = r' * z (' is for hermitian transpose)
    // p = z
    initialize(nrows, A_global_entry, b_global_entry, x_global_entry, x_sh,
               r_sh, prec_shared, z_sh, rho_old_sh, p_sh, norms_rhs_sh,
               item_ct1);
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // stopping criterion object
    StopType stop(tol, &norms_rhs_sh);

    int iter = 0;
    for (; iter < max_iter; iter++) {
        norms_res_sh = sqrt(abs(rho_old_sh));
        item_ct1.barrier(sycl::access::fence_space::local_space);
        if (stop.check_converged(&norms_res_sh)) {
            break;
        }

        // Ap = A * p
        single_matvec_kernel(A_global_entry, p_sh, Ap_sh, item_ct1);
        item_ct1.barrier(sycl::access::fence_space::local_space);

        // alpha = rho_old / (p' * Ap)
        // x = x + alpha * p
        // r = r - alpha * Ap
        update_x_and_r(nrows, rho_old_sh, p_sh, Ap_sh, x_sh, r_sh, item_ct1);
        item_ct1.barrier(sycl::access::fence_space::local_space);

        // z = precond * r
        prec_shared.apply(nrows, r_sh, z_sh, item_ct1);
        item_ct1.barrier(sycl::access::fence_space::local_space);

        compute_dot_product_kernel(nrows, r_sh, z_sh, rho_new_sh, item_ct1);

        // beta = rho_new / rho_old
        // p = z + beta * p
        update_p(nrows, rho_new_sh, rho_old_sh, z_sh, p_sh, item_ct1);
        item_ct1.barrier(sycl::access::fence_space::local_space);

        // rho_old = rho_new
        rho_old_sh = rho_new_sh;
    }

    logger.log_iteration(ibatch, iter, norms_res_sh);

    // copy x back to global memory
    copy_kernel(nrows, x_sh, x_global_entry, item_ct1);
    item_ct1.barrier(sycl::access::fence_space::local_space);
}


template <typename StopType, typename PrecType, typename LogType,
          typename BatchMatrixType, typename ValueType>
void apply_kernel(const gko::kernels::batch_cg::StorageConfig sconf,
                  const int max_iter, const gko::remove_complex<ValueType> tol,
                  LogType logger, PrecType prec_shared,
                  const BatchMatrixType A_global_entry,
                  const ValueType* const __restrict__ b_global_entry,
                  ValueType* const __restrict__ x_global_entry,
                  const size_type nrows, const size_type nnz,
                  ValueType* slm_values, sycl::nd_item<3> item_ct1,
                  ValueType* const __restrict__ workspace = nullptr)
{
    using real_type = typename gko::remove_complex<ValueType>;

    const auto sg = item_ct1.get_sub_group();
    const int sg_id = sg.get_group_id();
    const int sg_size = sg.get_local_range().size();
    const int num_sg = sg.get_group_range().size();

    const auto ibatch = item_ct1.get_group_linear_id();

    // The whole workgroup have the same values for these variables, but these
    // variables are stored in reg. mem, not on SLM
    ValueType rho_old_sh;
    ValueType rho_new_sh;
    remove_complex<ValueType> norms_rhs_sh;
    remove_complex<ValueType> norms_res_sh;

    const int gmem_offset =
        ibatch * sconf.gmem_stride_bytes / sizeof(ValueType);
    ValueType* r_sh;
    ValueType* z_sh;
    ValueType* p_sh;
    ValueType* Ap_sh;
    ValueType* x_sh;
    ValueType* prec_work_sh;
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
    // generate preconditioner
    prec_shared.generate(ibatch, A_global_entry, prec_work_sh, item_ct1);

    // initialization
    // compute b norms
    // r = b - A*x
    // z = precond*r
    // rho_old = r' * z (' is for hermitian transpose)
    // p = z
    initialize(nrows, A_global_entry, b_global_entry, x_global_entry, x_sh,
               r_sh, prec_shared, z_sh, rho_old_sh, p_sh, norms_rhs_sh,
               item_ct1);
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // stopping criterion object
    StopType stop(tol, &norms_rhs_sh);

    int iter = 0;
    for (; iter < max_iter; iter++) {
        norms_res_sh = sqrt(abs(rho_old_sh));
        item_ct1.barrier(sycl::access::fence_space::local_space);
        if (stop.check_converged(&norms_res_sh)) {
            break;
        }

        // Ap = A * p
        single_matvec_kernel(A_global_entry, p_sh, Ap_sh, item_ct1);
        item_ct1.barrier(sycl::access::fence_space::local_space);

        // alpha = rho_old / (p' * Ap)
        // x = x + alpha * p
        // r = r - alpha * Ap
        update_x_and_r(nrows, rho_old_sh, p_sh, Ap_sh, x_sh, r_sh, item_ct1);
        item_ct1.barrier(sycl::access::fence_space::local_space);

        // z = precond * r
        prec_shared.apply(nrows, r_sh, z_sh, item_ct1);
        item_ct1.barrier(sycl::access::fence_space::local_space);

        //  rho_new =  (r)' * (z)
        compute_dot_product_kernel(nrows, r_sh, z_sh, rho_new_sh, item_ct1);

        // beta = rho_new / rho_old
        // p = z + beta * p
        update_p(nrows, rho_new_sh, rho_old_sh, z_sh, p_sh, item_ct1);
        item_ct1.barrier(sycl::access::fence_space::local_space);

        rho_old_sh = rho_new_sh;
    }

    logger.log_iteration(ibatch, iter, norms_res_sh);

    // copy x back to global memory
    copy_kernel(nrows, x_sh, x_global_entry, item_ct1);
    item_ct1.barrier(sycl::access::fence_space::local_space);
}
