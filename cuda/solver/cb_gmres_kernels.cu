// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/cb_gmres_kernels.hpp"


#include <algorithm>


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/range.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>


#include "accessor/cuda_helper.hpp"
#include "accessor/range.hpp"
#include "accessor/reduced_row_major.hpp"
#include "accessor/scaled_reduced_row_major.hpp"
#include "core/base/array_access.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "core/solver/cb_gmres_accessor.hpp"
#include "cuda/base/config.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/atomic.cuh"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/reduction.cuh"
#include "cuda/components/thread_ids.cuh"
#include "cuda/components/uninitialized_array.hpp"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The CB_GMRES solver namespace.
 *
 * @ingroup cb_gmres
 */
namespace cb_gmres {


constexpr int default_block_size = 512;
constexpr int default_dot_dim = 32;
constexpr int default_dot_size = default_dot_dim * default_dot_dim;


#include "common/cuda_hip/solver/cb_gmres_kernels.hpp.inc"


template <typename ValueType>
void zero_matrix(std::shared_ptr<const DefaultExecutor> exec, size_type m,
                 size_type n, size_type stride, ValueType* array)
{
    const auto block_size = default_block_size;
    const auto grid_size = ceildiv(n, block_size);
    zero_matrix_kernel<<<grid_size, block_size, 0, exec->get_stream()>>>(
        m, n, stride, as_device_type(array));
}


template <typename ValueType>
void initialize(std::shared_ptr<const DefaultExecutor> exec,
                const matrix::Dense<ValueType>* b,
                matrix::Dense<ValueType>* residual,
                matrix::Dense<ValueType>* givens_sin,
                matrix::Dense<ValueType>* givens_cos,
                array<stopping_status>* stop_status, size_type krylov_dim)
{
    const auto num_threads = std::max(b->get_size()[0] * b->get_stride(),
                                      krylov_dim * b->get_size()[1]);
    const auto grid_dim = ceildiv(num_threads, default_block_size);
    const auto block_dim = default_block_size;
    constexpr auto block_size = default_block_size;

    initialize_kernel<block_size>
        <<<grid_dim, block_dim, 0, exec->get_stream()>>>(
            b->get_size()[0], b->get_size()[1], krylov_dim,
            as_device_type(b->get_const_values()), b->get_stride(),
            as_device_type(residual->get_values()), residual->get_stride(),
            as_device_type(givens_sin->get_values()), givens_sin->get_stride(),
            as_device_type(givens_cos->get_values()), givens_cos->get_stride(),
            as_device_type(stop_status->get_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CB_GMRES_INITIALIZE_KERNEL);


template <typename ValueType, typename Accessor3d>
void restart(std::shared_ptr<const DefaultExecutor> exec,
             const matrix::Dense<ValueType>* residual,
             matrix::Dense<remove_complex<ValueType>>* residual_norm,
             matrix::Dense<ValueType>* residual_norm_collection,
             matrix::Dense<remove_complex<ValueType>>* arnoldi_norm,
             Accessor3d krylov_bases,
             matrix::Dense<ValueType>* next_krylov_basis,
             array<size_type>* final_iter_nums, array<char>& reduction_tmp,
             size_type krylov_dim)
{
    constexpr bool use_scalar =
        gko::cb_gmres::detail::has_3d_scaled_accessor<Accessor3d>::value;
    const auto num_rows = residual->get_size()[0];
    const auto num_rhs = residual->get_size()[1];
    const auto krylov_stride =
        gko::cb_gmres::helper_functions_accessor<Accessor3d>::get_stride(
            krylov_bases);
    const auto grid_dim_1 =
        ceildiv((krylov_dim + 1) * krylov_stride[0], default_block_size);
    const auto block_dim = default_block_size;
    constexpr auto block_size = default_block_size;
    const auto stride_arnoldi = arnoldi_norm->get_stride();

    restart_1_kernel<block_size>
        <<<grid_dim_1, block_dim, 0, exec->get_stream()>>>(
            residual->get_size()[0], residual->get_size()[1], krylov_dim,
            acc::as_cuda_range(krylov_bases),
            as_device_type(residual_norm_collection->get_values()),
            residual_norm_collection->get_stride());
    kernels::cuda::dense::compute_norm2_dispatch(exec, residual, residual_norm,
                                                 reduction_tmp);

    if (use_scalar) {
        components::fill_array(exec,
                               arnoldi_norm->get_values() + 2 * stride_arnoldi,
                               num_rhs, zero<remove_complex<ValueType>>());
        const dim3 grid_size_nrm(ceildiv(num_rhs, default_dot_dim),
                                 exec->get_num_multiprocessor() * 2);
        const dim3 block_size_nrm(default_dot_dim, default_dot_dim);
        multinorminf_without_stop_kernel<<<grid_size_nrm, block_size_nrm, 0,
                                           exec->get_stream()>>>(
            num_rows, num_rhs, as_device_type(residual->get_const_values()),
            residual->get_stride(),
            as_device_type(arnoldi_norm->get_values() + 2 * stride_arnoldi), 0);
    }

    if (gko::cb_gmres::detail::has_3d_scaled_accessor<Accessor3d>::value) {
        set_scalar_kernel<default_block_size>
            <<<ceildiv(num_rhs * (krylov_dim + 1), default_block_size),
               default_block_size, 0, exec->get_stream()>>>(
                num_rhs, krylov_dim + 1,
                as_device_type(residual_norm->get_const_values()),
                residual_norm->get_stride(),
                as_device_type(arnoldi_norm->get_const_values() +
                               2 * stride_arnoldi),
                stride_arnoldi, acc::as_cuda_range(krylov_bases));
    }

    const auto grid_dim_2 =
        ceildiv(std::max<size_type>(num_rows, 1) * krylov_stride[1],
                default_block_size);
    restart_2_kernel<block_size>
        <<<grid_dim_2, block_dim, 0, exec->get_stream()>>>(
            residual->get_size()[0], residual->get_size()[1],
            as_device_type(residual->get_const_values()),
            residual->get_stride(),
            as_device_type(residual_norm->get_const_values()),
            as_device_type(residual_norm_collection->get_values()),
            acc::as_cuda_range(krylov_bases),
            as_device_type(next_krylov_basis->get_values()),
            next_krylov_basis->get_stride(),
            as_device_type(final_iter_nums->get_data()));
}

GKO_INSTANTIATE_FOR_EACH_CB_GMRES_TYPE(GKO_DECLARE_CB_GMRES_RESTART_KERNEL);


template <typename ValueType, typename Accessor3dim>
void finish_arnoldi_CGS(std::shared_ptr<const DefaultExecutor> exec,
                        matrix::Dense<ValueType>* next_krylov_basis,
                        Accessor3dim krylov_bases,
                        matrix::Dense<ValueType>* hessenberg_iter,
                        matrix::Dense<ValueType>* buffer_iter,
                        matrix::Dense<remove_complex<ValueType>>* arnoldi_norm,
                        size_type iter, const stopping_status* stop_status,
                        stopping_status* reorth_status,
                        array<size_type>* num_reorth)
{
    const auto dim_size = next_krylov_basis->get_size();
    if (dim_size[1] == 0) {
        return;
    }
    using non_complex = remove_complex<ValueType>;
    // optimization parameter
    constexpr int singledot_block_size = default_dot_dim;
    constexpr bool use_scalar =
        gko::cb_gmres::detail::has_3d_scaled_accessor<Accessor3dim>::value;
    const auto stride_next_krylov = next_krylov_basis->get_stride();
    const auto stride_hessenberg = hessenberg_iter->get_stride();
    const auto stride_buffer = buffer_iter->get_stride();
    const auto stride_arnoldi = arnoldi_norm->get_stride();
    const dim3 grid_size(ceildiv(dim_size[1], default_dot_dim),
                         exec->get_num_multiprocessor() * 2);
    const dim3 grid_size_num_iters(ceildiv(dim_size[1], default_dot_dim),
                                   exec->get_num_multiprocessor() * 2,
                                   iter + 1);
    const dim3 block_size(default_dot_dim, default_dot_dim);
    // Note: having iter first (instead of row_idx information) is likely
    //       beneficial for avoiding atomic_add conflicts, but that needs
    //       further investigation.
    const dim3 grid_size_iters_single(exec->get_num_multiprocessor() * 2,
                                      iter + 1);
    const auto block_size_iters_single = singledot_block_size;
    size_type num_reorth_host;

    components::fill_array(exec, arnoldi_norm->get_values(), dim_size[1],
                           zero<non_complex>());
    multinorm2_kernel<<<grid_size, block_size, 0, exec->get_stream()>>>(
        dim_size[0], dim_size[1],
        as_device_type(next_krylov_basis->get_const_values()),
        stride_next_krylov, as_device_type(arnoldi_norm->get_values()),
        as_device_type(stop_status));
    zero_matrix(exec, iter + 1, dim_size[1], stride_hessenberg,
                hessenberg_iter->get_values());
    if (dim_size[1] > 1) {
        multidot_kernel<default_dot_dim>
            <<<grid_size_num_iters, block_size, 0, exec->get_stream()>>>(
                dim_size[0], dim_size[1],
                as_device_type(next_krylov_basis->get_const_values()),
                stride_next_krylov, acc::as_cuda_range(krylov_bases),
                as_device_type(hessenberg_iter->get_values()),
                stride_hessenberg, as_device_type(stop_status));
    } else {
        singledot_kernel<singledot_block_size>
            <<<grid_size_iters_single, block_size_iters_single, 0,
               exec->get_stream()>>>(
                dim_size[0],
                as_device_type(next_krylov_basis->get_const_values()),
                stride_next_krylov, acc::as_cuda_range(krylov_bases),
                as_device_type(hessenberg_iter->get_values()),
                stride_hessenberg, as_device_type(stop_status));
    }
    // for i in 1:iter
    //     hessenberg(iter, i) = next_krylov_basis' * krylov_bases(:, i)
    // end
    update_next_krylov_kernel<default_block_size>
        <<<ceildiv(dim_size[0] * stride_next_krylov, default_block_size),
           default_block_size, 0, exec->get_stream()>>>(
            iter + 1, dim_size[0], dim_size[1],
            as_device_type(next_krylov_basis->get_values()), stride_next_krylov,
            acc::as_cuda_range(krylov_bases),
            as_device_type(hessenberg_iter->get_const_values()),
            stride_hessenberg, as_device_type(stop_status));

    // for i in 1:iter
    //     next_krylov_basis  -= hessenberg(iter, i) * krylov_bases(:, i)
    // end
    components::fill_array(exec, arnoldi_norm->get_values() + stride_arnoldi,
                           dim_size[1], zero<non_complex>());
    if (use_scalar) {
        components::fill_array(exec,
                               arnoldi_norm->get_values() + 2 * stride_arnoldi,
                               dim_size[1], zero<non_complex>());
    }
    multinorm2_inf_kernel<use_scalar>
        <<<grid_size, block_size, 0, exec->get_stream()>>>(
            dim_size[0], dim_size[1],
            as_device_type(next_krylov_basis->get_const_values()),
            stride_next_krylov,
            as_device_type(arnoldi_norm->get_values() + stride_arnoldi),
            as_device_type(arnoldi_norm->get_values() + 2 * stride_arnoldi),
            as_device_type(stop_status));
    // nrmN = norm(next_krylov_basis)
    components::fill_array(exec, num_reorth->get_data(), 1, zero<size_type>());
    check_arnoldi_norms<default_block_size>
        <<<ceildiv(dim_size[1], default_block_size), default_block_size, 0,
           exec->get_stream()>>>(
            dim_size[1], as_device_type(arnoldi_norm->get_values()),
            stride_arnoldi, as_device_type(hessenberg_iter->get_values()),
            stride_hessenberg, iter + 1, acc::as_cuda_range(krylov_bases),
            as_device_type(stop_status), as_device_type(reorth_status),
            as_device_type(num_reorth->get_data()));
    num_reorth_host = get_element(*num_reorth, 0);
    // num_reorth_host := number of next_krylov vector to be reorthogonalization
    for (size_type l = 1; (num_reorth_host > 0) && (l < 3); l++) {
        zero_matrix(exec, iter + 1, dim_size[1], stride_buffer,
                    buffer_iter->get_values());
        if (dim_size[1] > 1) {
            multidot_kernel<default_dot_dim>
                <<<grid_size_num_iters, block_size, 0, exec->get_stream()>>>(
                    dim_size[0], dim_size[1],
                    as_device_type(next_krylov_basis->get_const_values()),
                    stride_next_krylov, acc::as_cuda_range(krylov_bases),
                    as_device_type(buffer_iter->get_values()), stride_buffer,
                    as_device_type(stop_status));
        } else {
            singledot_kernel<singledot_block_size>
                <<<grid_size_iters_single, block_size_iters_single, 0,
                   exec->get_stream()>>>(
                    dim_size[0],
                    as_device_type(next_krylov_basis->get_const_values()),
                    stride_next_krylov, acc::as_cuda_range(krylov_bases),
                    as_device_type(buffer_iter->get_values()), stride_buffer,
                    as_device_type(stop_status));
        }
        // for i in 1:iter
        //     hessenberg(iter, i) = next_krylov_basis' * krylov_bases(:, i)
        // end
        update_next_krylov_and_add_kernel<default_block_size>
            <<<ceildiv(dim_size[0] * stride_next_krylov, default_block_size),
               default_block_size, 0, exec->get_stream()>>>(
                iter + 1, dim_size[0], dim_size[1],
                as_device_type(next_krylov_basis->get_values()),
                stride_next_krylov, acc::as_cuda_range(krylov_bases),
                as_device_type(hessenberg_iter->get_values()),
                stride_hessenberg,
                as_device_type(buffer_iter->get_const_values()), stride_buffer,
                as_device_type(stop_status), as_device_type(reorth_status));
        // for i in 1:iter
        //     next_krylov_basis  -= hessenberg(iter, i) * krylov_bases(:, i)
        // end
        components::fill_array(exec,
                               arnoldi_norm->get_values() + stride_arnoldi,
                               dim_size[1], zero<non_complex>());
        if (use_scalar) {
            components::fill_array(
                exec, arnoldi_norm->get_values() + 2 * stride_arnoldi,
                dim_size[1], zero<non_complex>());
        }
        multinorm2_inf_kernel<use_scalar>
            <<<grid_size, block_size, 0, exec->get_stream()>>>(
                dim_size[0], dim_size[1],
                as_device_type(next_krylov_basis->get_const_values()),
                stride_next_krylov,
                as_device_type(arnoldi_norm->get_values() + stride_arnoldi),
                as_device_type(arnoldi_norm->get_values() + 2 * stride_arnoldi),
                as_device_type(stop_status));
        // nrmN = norm(next_krylov_basis)
        components::fill_array(exec, num_reorth->get_data(), 1,
                               zero<size_type>());
        check_arnoldi_norms<default_block_size>
            <<<ceildiv(dim_size[1], default_block_size), default_block_size, 0,
               exec->get_stream()>>>(
                dim_size[1], as_device_type(arnoldi_norm->get_values()),
                stride_arnoldi, as_device_type(hessenberg_iter->get_values()),
                stride_hessenberg, iter + 1, acc::as_cuda_range(krylov_bases),
                as_device_type(stop_status), as_device_type(reorth_status),
                num_reorth->get_data());
        num_reorth_host = get_element(*num_reorth, 0);
    }

    update_krylov_next_krylov_kernel<default_block_size>
        <<<ceildiv(dim_size[0] * stride_next_krylov, default_block_size),
           default_block_size, 0, exec->get_stream()>>>(
            iter, dim_size[0], dim_size[1],
            as_device_type(next_krylov_basis->get_values()), stride_next_krylov,
            acc::as_cuda_range(krylov_bases),
            as_device_type(hessenberg_iter->get_const_values()),
            stride_hessenberg, as_device_type(stop_status));
    // next_krylov_basis /= hessenberg(iter, iter + 1)
    // krylov_bases(:, iter + 1) = next_krylov_basis
    // End of arnoldi
}

template <typename ValueType>
void givens_rotation(std::shared_ptr<const DefaultExecutor> exec,
                     matrix::Dense<ValueType>* givens_sin,
                     matrix::Dense<ValueType>* givens_cos,
                     matrix::Dense<ValueType>* hessenberg_iter,
                     matrix::Dense<remove_complex<ValueType>>* residual_norm,
                     matrix::Dense<ValueType>* residual_norm_collection,
                     size_type iter, const array<stopping_status>* stop_status)
{
    // TODO: tune block_size for optimal performance
    constexpr auto block_size = default_block_size;
    const auto num_cols = hessenberg_iter->get_size()[1];
    const auto block_dim = block_size;
    const auto grid_dim =
        static_cast<unsigned int>(ceildiv(num_cols, block_size));

    givens_rotation_kernel<block_size>
        <<<grid_dim, block_dim, 0, exec->get_stream()>>>(
            hessenberg_iter->get_size()[0], hessenberg_iter->get_size()[1],
            iter, as_device_type(hessenberg_iter->get_values()),
            hessenberg_iter->get_stride(),
            as_device_type(givens_sin->get_values()), givens_sin->get_stride(),
            as_device_type(givens_cos->get_values()), givens_cos->get_stride(),
            as_device_type(residual_norm->get_values()),
            as_device_type(residual_norm_collection->get_values()),
            residual_norm_collection->get_stride(),
            stop_status->get_const_data());
}


template <typename ValueType, typename Accessor3d>
void arnoldi(std::shared_ptr<const DefaultExecutor> exec,
             matrix::Dense<ValueType>* next_krylov_basis,
             matrix::Dense<ValueType>* givens_sin,
             matrix::Dense<ValueType>* givens_cos,
             matrix::Dense<remove_complex<ValueType>>* residual_norm,
             matrix::Dense<ValueType>* residual_norm_collection,
             Accessor3d krylov_bases, matrix::Dense<ValueType>* hessenberg_iter,
             matrix::Dense<ValueType>* buffer_iter,
             matrix::Dense<remove_complex<ValueType>>* arnoldi_norm,
             size_type iter, array<size_type>* final_iter_nums,
             const array<stopping_status>* stop_status,
             array<stopping_status>* reorth_status,
             array<size_type>* num_reorth)
{
    increase_final_iteration_numbers_kernel<<<
        static_cast<unsigned int>(
            ceildiv(final_iter_nums->get_size(), default_block_size)),
        default_block_size, 0, exec->get_stream()>>>(
        as_device_type(final_iter_nums->get_data()),
        stop_status->get_const_data(), final_iter_nums->get_size());
    finish_arnoldi_CGS(exec, next_krylov_basis, krylov_bases, hessenberg_iter,
                       buffer_iter, arnoldi_norm, iter,
                       stop_status->get_const_data(), reorth_status->get_data(),
                       num_reorth);
    givens_rotation(exec, givens_sin, givens_cos, hessenberg_iter,
                    residual_norm, residual_norm_collection, iter, stop_status);
}

GKO_INSTANTIATE_FOR_EACH_CB_GMRES_TYPE(GKO_DECLARE_CB_GMRES_ARNOLDI_KERNEL);


template <typename ValueType>
void solve_upper_triangular(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Dense<ValueType>* residual_norm_collection,
    const matrix::Dense<ValueType>* hessenberg, matrix::Dense<ValueType>* y,
    const array<size_type>* final_iter_nums)
{
    // TODO: tune block_size for optimal performance
    constexpr auto block_size = default_block_size;
    const auto num_rhs = residual_norm_collection->get_size()[1];
    const auto block_dim = block_size;
    const auto grid_dim =
        static_cast<unsigned int>(ceildiv(num_rhs, block_size));

    solve_upper_triangular_kernel<block_size>
        <<<grid_dim, block_dim, 0, exec->get_stream()>>>(
            hessenberg->get_size()[1], num_rhs,
            as_device_type(residual_norm_collection->get_const_values()),
            residual_norm_collection->get_stride(),
            as_device_type(hessenberg->get_const_values()),
            hessenberg->get_stride(), as_device_type(y->get_values()),
            y->get_stride(), as_device_type(final_iter_nums->get_const_data()));
}


template <typename ValueType, typename ConstAccessor3d>
void calculate_qy(std::shared_ptr<const DefaultExecutor> exec,
                  ConstAccessor3d krylov_bases, size_type num_krylov_bases,
                  const matrix::Dense<ValueType>* y,
                  matrix::Dense<ValueType>* before_preconditioner,
                  const array<size_type>* final_iter_nums)
{
    const auto num_rows = before_preconditioner->get_size()[0];
    const auto num_cols = before_preconditioner->get_size()[1];
    const auto stride_before_preconditioner =
        before_preconditioner->get_stride();

    constexpr auto block_size = default_block_size;
    const auto grid_dim = static_cast<unsigned int>(
        ceildiv(num_rows * stride_before_preconditioner, block_size));
    const auto block_dim = block_size;

    calculate_Qy_kernel<block_size>
        <<<grid_dim, block_dim, 0, exec->get_stream()>>>(
            num_rows, num_cols, acc::as_cuda_range(krylov_bases),
            as_device_type(y->get_const_values()), y->get_stride(),
            as_device_type(before_preconditioner->get_values()),
            stride_before_preconditioner,
            as_device_type(final_iter_nums->get_const_data()));
    // Calculate qy
    // before_preconditioner = krylov_bases * y
}


template <typename ValueType, typename ConstAccessor3d>
void solve_krylov(std::shared_ptr<const DefaultExecutor> exec,
                  const matrix::Dense<ValueType>* residual_norm_collection,
                  ConstAccessor3d krylov_bases,
                  const matrix::Dense<ValueType>* hessenberg,
                  matrix::Dense<ValueType>* y,
                  matrix::Dense<ValueType>* before_preconditioner,
                  const array<size_type>* final_iter_nums)
{
    if (before_preconditioner->get_size()[1] == 0) {
        return;
    }
    // since hessenberg has dims:  iters x iters * num_rhs
    // krylov_bases has dims:  (iters + 1) x sysmtx[0] x num_rhs
    const auto iters =
        hessenberg->get_size()[1] / before_preconditioner->get_size()[1];
    const auto num_krylov_bases = iters + 1;
    solve_upper_triangular(exec, residual_norm_collection, hessenberg, y,
                           final_iter_nums);
    calculate_qy(exec, krylov_bases, num_krylov_bases, y, before_preconditioner,
                 final_iter_nums);
}

GKO_INSTANTIATE_FOR_EACH_CB_GMRES_CONST_TYPE(
    GKO_DECLARE_CB_GMRES_SOLVE_KRYLOV_KERNEL);


}  // namespace cb_gmres
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
