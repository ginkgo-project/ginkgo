// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_DPCPP_SOLVER_BATCH_CG_KERNELS_HPP_
#define GKO_DPCPP_SOLVER_BATCH_CG_KERNELS_HPP_


#include <memory>

#include <CL/sycl.hpp>

#include "core/base/batch_struct.hpp"
#include "core/matrix/batch_struct.hpp"
#include "dpcpp/base/batch_multi_vector_kernels.hpp"
#include "dpcpp/base/batch_struct.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/dpct.hpp"
#include "dpcpp/base/helper.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"
#include "dpcpp/components/reduction.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"
#include "dpcpp/matrix/batch_csr_kernels.hpp"
#include "dpcpp/matrix/batch_dense_kernels.hpp"
#include "dpcpp/matrix/batch_ell_kernels.hpp"
#include "dpcpp/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace batch_single_kernels {


template <typename PrecType, typename ValueType, typename BatchMatrixType>
__dpct_inline__ void initialize(
    const int num_rows, const BatchMatrixType& mat_global_entry,
    const ValueType* const __restrict__ b_global_entry,
    const ValueType* const __restrict__ x_global_entry,
    ValueType* const __restrict__ x_shared_entry,
    ValueType* const __restrict__ r_shared_entry, const PrecType& prec_shared,
    ValueType* const __restrict__ z_shared_entry, ValueType& rho_old,
    ValueType* const __restrict__ p_shared_entry,
    gko::remove_complex<ValueType>& rhs_norms, sycl::nd_item<3> item_ct1)
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
    item_ct1.barrier(sycl::access::fence_space::global_and_local);

    // r = b - A*x
    advanced_apply(static_cast<ValueType>(-1.0), mat_global_entry,
                   x_shared_entry, static_cast<ValueType>(1.0), r_shared_entry,
                   item_ct1);
    item_ct1.barrier(sycl::access::fence_space::global_and_local);


    // z = precond * r
    prec_shared.apply(num_rows, r_shared_entry, z_shared_entry, item_ct1);
    item_ct1.barrier(sycl::access::fence_space::global_and_local);

    // Compute norms of rhs
    // and rho_old = r' * z
    if (sg_id == 0) {
        single_rhs_compute_norm2_sg(num_rows, b_global_entry, rhs_norms,
                                    item_ct1);
    } else if (sg_id == 1) {
        single_rhs_compute_conj_dot_sg(num_rows, r_shared_entry, z_shared_entry,
                                       rho_old, item_ct1);
    }
    item_ct1.barrier(sycl::access::fence_space::global_and_local);

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

template <typename ValueType>
__dpct_inline__ void update_x_and_r(
    const int num_rows, const ValueType rho_old_shared_entry,
    const ValueType* const __restrict__ p_shared_entry,
    const ValueType* const __restrict__ Ap_shared_entry,
    ValueType& alpha_shared_entry, ValueType* const __restrict__ x_shared_entry,
    ValueType* const __restrict__ r_shared_entry, sycl::nd_item<3> item_ct1)
{
    auto sg = item_ct1.get_sub_group();
    const auto tid = item_ct1.get_local_linear_id();
    if (sg.get_group_id() == 0) {
        single_rhs_compute_conj_dot_sg(num_rows, p_shared_entry,
                                       Ap_shared_entry, alpha_shared_entry,
                                       item_ct1);
    }
    item_ct1.barrier(sycl::access::fence_space::global_and_local);
    if (tid == 0) {
        alpha_shared_entry = rho_old_shared_entry / alpha_shared_entry;
    }
    item_ct1.barrier(sycl::access::fence_space::global_and_local);

    for (int li = item_ct1.get_local_linear_id(); li < num_rows;
         li += item_ct1.get_local_range().size()) {
        x_shared_entry[li] += alpha_shared_entry * p_shared_entry[li];
        r_shared_entry[li] -= alpha_shared_entry * Ap_shared_entry[li];
    }
    item_ct1.barrier(sycl::access::fence_space::global_and_local);
}


template <typename StopType, const int n_shared_total, typename PrecType,
          typename LogType, typename BatchMatrixType, typename ValueType>
__dpct_inline__ void apply_kernel(
    const gko::kernels::batch_cg::storage_config sconf, const int max_iter,
    const gko::remove_complex<ValueType> tol, LogType logger,
    PrecType prec_shared, const BatchMatrixType& mat_global_entry,
    const ValueType* const __restrict__ b_global_entry,
    ValueType* const __restrict__ x_global_entry, const size_type num_rows,
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
    const auto batch_id = item_ct1.get_group_linear_id();

    // The whole workgroup have the same values for these variables, but
    // these variables are stored in reg. mem, not on SLM
    using tile_value_t = ValueType[3];
    tile_value_t& values =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_value_t>(
            group);
    using tile_real_t = real_type[2];
    tile_real_t& reals =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_real_t>(
            group);
    ValueType* rho_old_sh = &values[0];
    ValueType* rho_new_sh = &values[1];
    ValueType* alpha_sh = &values[2];
    real_type* norms_rhs_sh = &reals[0];
    real_type* norms_res_sh = &reals[1];
    const int gmem_offset =
        batch_id * sconf.gmem_stride_bytes / sizeof(ValueType);
    ValueType* r_sh;
    ValueType* z_sh;
    ValueType* p_sh;
    ValueType* Ap_sh;
    ValueType* x_sh;
    ValueType* prec_work_sh;

    if constexpr (n_shared_total >= 1) {
        r_sh = slm_values;
    } else {
        r_sh = workspace + gmem_offset;
    }
    if constexpr (n_shared_total == 1) {
        z_sh = workspace + gmem_offset;
    } else {
        z_sh = r_sh + sconf.padded_vec_len;
    }
    if constexpr (n_shared_total == 2) {
        p_sh = workspace + gmem_offset;
    } else {
        p_sh = z_sh + sconf.padded_vec_len;
    }
    if constexpr (n_shared_total == 3) {
        Ap_sh = workspace + gmem_offset;
    } else {
        Ap_sh = p_sh + sconf.padded_vec_len;
    }
    if constexpr (n_shared_total == 4) {
        x_sh = workspace + gmem_offset;
    } else {
        x_sh = Ap_sh + sconf.padded_vec_len;
    }
    if constexpr (n_shared_total == 5) {
        prec_work_sh = workspace + gmem_offset;
    } else {
        prec_work_sh = x_sh + sconf.padded_vec_len;
    }

    // generate preconditioner
    prec_shared.generate(batch_id, mat_global_entry, prec_work_sh, item_ct1);

    // initialization
    // compute b norms
    // r = b - A*x
    // z = precond*r
    // rho_old = r' * z (' is for hermitian transpose)
    // p = z
    initialize(num_rows, mat_global_entry, b_global_entry, x_global_entry, x_sh,
               r_sh, prec_shared, z_sh, rho_old_sh[0], p_sh, norms_rhs_sh[0],
               item_ct1);
    item_ct1.barrier(sycl::access::fence_space::global_and_local);

    // stopping criterion object
    StopType stop(tol, norms_rhs_sh);

    int iter = 0;
    for (; iter < max_iter; iter++) {
        if (sg.leader()) {
            norms_res_sh[0] = sqrt(abs(rho_old_sh[0]));
        }
        item_ct1.barrier(sycl::access::fence_space::global_and_local);
        if (stop.check_converged(norms_res_sh)) {
            logger.log_iteration(batch_id, iter, norms_res_sh[0]);
            break;
        }
        // Ap = A * p
        simple_apply(mat_global_entry, p_sh, Ap_sh, item_ct1);
        item_ct1.barrier(sycl::access::fence_space::global_and_local);

        // alpha = rho_old / (p' * Ap)
        // x = x + alpha * p
        // r = r - alpha * Ap
        update_x_and_r(num_rows, rho_old_sh[0], p_sh, Ap_sh, alpha_sh[0], x_sh,
                       r_sh, item_ct1);
        item_ct1.barrier(sycl::access::fence_space::global_and_local);


        // z = precond * r
        prec_shared.apply(num_rows, r_sh, z_sh, item_ct1);
        item_ct1.barrier(sycl::access::fence_space::global_and_local);

        //  rho_new =  (r)' * (z)
        if (sg_id == 0) {
            single_rhs_compute_conj_dot_sg(num_rows, r_sh, z_sh, rho_new_sh[0],
                                           item_ct1);
        }
        item_ct1.barrier(sycl::access::fence_space::global_and_local);

        // beta = rho_new / rho_old
        // p = z + beta * p
        update_p(num_rows, rho_new_sh[0], rho_old_sh[0], z_sh, p_sh, item_ct1);
        item_ct1.barrier(sycl::access::fence_space::global_and_local);
        if (sg.leader()) {
            rho_old_sh[0] = rho_new_sh[0];
        }
        item_ct1.barrier(sycl::access::fence_space::global_and_local);
    }

    logger.log_iteration(batch_id, iter, norms_res_sh[0]);

    // copy x back to global memory
    copy_kernel(num_rows, x_sh, x_global_entry, item_ct1);
    item_ct1.barrier(sycl::access::fence_space::global_and_local);
}


}  // namespace batch_single_kernels
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko


#endif
