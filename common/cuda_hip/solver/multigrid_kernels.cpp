// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/multigrid_kernels.hpp"

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>

#include "accessor/cuda_hip_helper.hpp"
#include "common/cuda_hip/base/math.hpp"
#include "common/cuda_hip/base/runtime.hpp"
#include "common/cuda_hip/base/types.hpp"
#include "common/cuda_hip/components/thread_ids.hpp"
#include "common/unified/base/arithmetic_accessor.hpp"
#include "core/base/array_access.hpp"
#include "core/components/fill_array_kernels.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The MULTIGRID solver namespace.
 *
 */
namespace multigrid {


constexpr int default_block_size = 512;


namespace kernel {


// grid_nrows is the number of rows handled in the whole grid at the same time.
// Thus, the threads whose index is larger than grid_nrows * nrhs are not used.
// Let the thread handle the same col (has same scalar) in whole loop.
template <typename ValueType, typename ValueAccessor, typename OutputAccessor>
__global__ __launch_bounds__(default_block_size) void kcycle_step_1_kernel(
    const size_type num_rows, const size_type nrhs, const size_type stride,
    const size_type grid_nrows, ValueAccessor alpha, ValueAccessor rho,
    ValueAccessor v, OutputAccessor g, ValueType* __restrict__ d,
    ValueType* __restrict__ e)
{
    using arithmetic_type = arithmetic_type_t<OutputAccessor>;
    // d and e are accessed in the storage type, so that the copy stays exact
    const auto tidx = thread::get_thread_id_flat();
    const auto col = tidx % nrhs;
    const auto num_elems = grid_nrows * nrhs;
    if (tidx >= num_elems) {
        return;
    }
    const auto total_elems = num_rows * stride;
    const auto grid_stride = grid_nrows * stride;
    const auto temp = alpha[col] / rho[col];
    const bool update = is_finite(temp);
    for (auto idx = tidx / nrhs * stride + col; idx < total_elems;
         idx += grid_stride) {
        if (update) {
            g[idx] -= temp * v[idx];
            e[idx] = static_cast<ValueType>(
                static_cast<arithmetic_type>(e[idx]) * temp);
        }
        // copies the stored value, so that d is exactly e
        d[idx] = e[idx];
    }
}


template <typename ValueAccessor, typename OutputAccessor>
__global__ __launch_bounds__(default_block_size) void kcycle_step_2_kernel(
    const size_type num_rows, const size_type nrhs, const size_type stride,
    const size_type grid_nrows, ValueAccessor alpha, ValueAccessor rho,
    ValueAccessor gamma, ValueAccessor beta, ValueAccessor zeta,
    ValueAccessor d, OutputAccessor e)
{
    using arithmetic_type = arithmetic_type_t<OutputAccessor>;
    const auto tidx = thread::get_thread_id_flat();
    const auto col = tidx % nrhs;
    const auto num_elems = grid_nrows * nrhs;
    if (tidx >= num_elems) {
        return;
    }
    const auto total_elems = num_rows * stride;
    const auto grid_stride = grid_nrows * stride;
    const auto scalar_d =
        zeta[col] / (beta[col] - gamma[col] * gamma[col] / rho[col]);
    const auto scalar_e =
        one<arithmetic_type>() - gamma[col] / alpha[col] * scalar_d;
    if (is_finite(scalar_d) && is_finite(scalar_e)) {
        for (auto idx = tidx / nrhs * stride + col; idx < total_elems;
             idx += grid_stride) {
            e[idx] = scalar_e * e[idx] + scalar_d * d[idx];
        }
    }
}


template <typename NormAccessor, typename ArithmeticType>
__global__ __launch_bounds__(default_block_size) void kcycle_check_stop_kernel(
    const size_type nrhs, NormAccessor old_norm, NormAccessor new_norm,
    const ArithmeticType rel_tol, bool* __restrict__ is_stop)
{
    auto tidx = thread::get_thread_id_flat();
    if (tidx >= nrhs) {
        return;
    }
    if (new_norm[tidx] > rel_tol * old_norm[tidx]) {
        *is_stop = false;
    }
}


}  // namespace kernel


template <typename ValueType>
void kcycle_step_1(std::shared_ptr<const DefaultExecutor> exec,
                   matrix::view::dense<const ValueType> alpha,
                   matrix::view::dense<const ValueType> rho,
                   matrix::view::dense<const ValueType> v,
                   matrix::view::dense<ValueType> g,
                   matrix::view::dense<ValueType> d,
                   matrix::view::dense<ValueType> e)
{
    const auto nrows = e.size[0];
    const auto nrhs = e.size[1];
    constexpr int max_size = (1U << 31) - 1;
    const size_type grid_nrows =
        max_size / nrhs < nrows ? max_size / nrhs : nrows;
    const auto grid = ceildiv(grid_nrows * nrhs, default_block_size);
    const auto alpha_vals = arithmetic<ValueType, 1>(alpha);
    const auto rho_vals = arithmetic<ValueType, 1>(rho);
    const auto v_vals = arithmetic<ValueType, 1>(v);
    const auto g_vals = arithmetic<ValueType, 1>(g);
    if (grid > 0) {
        kernel::kcycle_step_1_kernel<<<grid, default_block_size, 0,
                                       exec->get_stream()>>>(
            nrows, nrhs, e.stride, grid_nrows, acc::as_device_range(alpha_vals),
            acc::as_device_range(rho_vals), acc::as_device_range(v_vals),
            acc::as_device_range(g_vals), as_device_type(d.values),
            as_device_type(e.values));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MULTIGRID_KCYCLE_STEP_1_KERNEL);


template <typename ValueType>
void kcycle_step_2(std::shared_ptr<const DefaultExecutor> exec,
                   matrix::view::dense<const ValueType> alpha,
                   matrix::view::dense<const ValueType> rho,
                   matrix::view::dense<const ValueType> gamma,
                   matrix::view::dense<const ValueType> beta,
                   matrix::view::dense<const ValueType> zeta,
                   matrix::view::dense<const ValueType> d,
                   matrix::view::dense<ValueType> e)
{
    const auto nrows = e.size[0];
    const auto nrhs = e.size[1];
    constexpr int max_size = (1U << 31) - 1;
    const size_type grid_nrows =
        max_size / nrhs < nrows ? max_size / nrhs : nrows;
    const auto grid = ceildiv(grid_nrows * nrhs, default_block_size);
    const auto alpha_vals = arithmetic<ValueType, 1>(alpha);
    const auto rho_vals = arithmetic<ValueType, 1>(rho);
    const auto gamma_vals = arithmetic<ValueType, 1>(gamma);
    const auto beta_vals = arithmetic<ValueType, 1>(beta);
    const auto zeta_vals = arithmetic<ValueType, 1>(zeta);
    const auto d_vals = arithmetic<ValueType, 1>(d);
    const auto e_vals = arithmetic<ValueType, 1>(e);
    if (grid > 0) {
        kernel::kcycle_step_2_kernel<<<grid, default_block_size, 0,
                                       exec->get_stream()>>>(
            nrows, nrhs, e.stride, grid_nrows, acc::as_device_range(alpha_vals),
            acc::as_device_range(rho_vals), acc::as_device_range(gamma_vals),
            acc::as_device_range(beta_vals), acc::as_device_range(zeta_vals),
            acc::as_device_range(d_vals), acc::as_device_range(e_vals));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MULTIGRID_KCYCLE_STEP_2_KERNEL);


template <typename ValueType>
void kcycle_check_stop(std::shared_ptr<const DefaultExecutor> exec,
                       matrix::view::dense<const ValueType> old_norm,
                       matrix::view::dense<const ValueType> new_norm,
                       const ValueType rel_tol, bool& is_stop)
{
    gko::array<bool> dis_stop(exec, 1);
    components::fill_array(exec, dis_stop.get_data(), dis_stop.get_size(),
                           true);
    const auto nrhs = new_norm.size[1];
    const auto grid = ceildiv(nrhs, default_block_size);
    const auto old_norm_vals = arithmetic<ValueType, 1>(old_norm);
    const auto new_norm_vals = arithmetic<ValueType, 1>(new_norm);
    if (grid > 0) {
        kernel::kcycle_check_stop_kernel<<<grid, default_block_size, 0,
                                           exec->get_stream()>>>(
            nrhs, acc::as_device_range(old_norm_vals),
            acc::as_device_range(new_norm_vals), arithmetic<ValueType>(rel_tol),
            as_device_type(dis_stop.get_data()));
    }
    is_stop = get_element(dis_stop, 0);
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(
    GKO_DECLARE_MULTIGRID_KCYCLE_CHECK_STOP_KERNEL);


}  // namespace multigrid
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
