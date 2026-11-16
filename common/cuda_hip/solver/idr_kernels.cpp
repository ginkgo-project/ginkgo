// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/idr_kernels.hpp"

#include <ctime>
#include <random>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>

#include "accessor/cuda_hip_helper.hpp"
#include "common/cuda_hip/base/blas_bindings.hpp"
#include "common/cuda_hip/base/config.hpp"
#include "common/cuda_hip/base/math.hpp"
#include "common/cuda_hip/base/randlib_bindings.hpp"
#include "common/cuda_hip/base/runtime.hpp"
#include "common/cuda_hip/base/types.hpp"
#include "common/cuda_hip/components/atomic.hpp"
#include "common/cuda_hip/components/cooperative_groups.hpp"
#include "common/cuda_hip/components/reduction.hpp"
#include "common/cuda_hip/components/thread_ids.hpp"
#include "common/unified/base/arithmetic_accessor.hpp"
#include "core/components/fill_array_kernels.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The IDR solver namespace.
 *
 */
namespace idr {


constexpr int default_block_size = 512;
constexpr int default_dot_dim = 32;
constexpr int default_dot_size = default_dot_dim * default_dot_dim;


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void initialize_m_kernel(
    size_type subspace_dim, size_type nrhs, ValueType* __restrict__ m_values,
    size_type m_stride, stopping_status* __restrict__ stop_status)
{
    const auto global_id = thread::get_thread_id_flat();
    const auto row = global_id / m_stride;
    const auto col = global_id % m_stride;

    if (global_id < nrhs) {
        stop_status[global_id].reset();
    }

    if (row < subspace_dim && col < nrhs * subspace_dim) {
        m_values[row * m_stride + col] =
            (row == col / nrhs) ? one<ValueType>() : zero<ValueType>();
    }
}


template <size_type block_size, typename ValueAccessor>
__global__
__launch_bounds__(block_size) void orthonormalize_subspace_vectors_kernel(
    size_type num_rows, size_type num_cols, ValueAccessor values,
    size_type stride)
{
    using arithmetic_type = arithmetic_type_t<ValueAccessor>;
    using real_arithmetic_type = remove_complex<arithmetic_type>;
    const auto tidx = thread::get_thread_id_flat();

    __shared__ uninitialized_array<arithmetic_type, block_size>
        reduction_helper_array;
    // they are not be used in the same time.
    arithmetic_type* reduction_helper = reduction_helper_array;
    auto reduction_helper_real =
        reinterpret_cast<real_arithmetic_type*>(reduction_helper);

    for (size_type row = 0; row < num_rows; row++) {
        for (size_type i = 0; i < row; i++) {
            auto dot = zero<arithmetic_type>();
            for (size_type j = tidx; j < num_cols; j += block_size) {
                dot += values[row * stride + j] * conj(values[i * stride + j]);
            }

            // Ensure already finish reading this shared memory
            __syncthreads();
            reduction_helper[tidx] = dot;
            reduce(group::this_thread_block(), reduction_helper,
                   [](const arithmetic_type& a, const arithmetic_type& b) {
                       return a + b;
                   });
            __syncthreads();

            dot = reduction_helper[0];
            for (size_type j = tidx; j < num_cols; j += block_size) {
                values[row * stride + j] -= dot * values[i * stride + j];
            }
        }

        auto norm = zero<real_arithmetic_type>();
        for (size_type j = tidx; j < num_cols; j += block_size) {
            norm += squared_norm(values[row * stride + j]);
        }
        // Ensure already finish reading this shared memory
        __syncthreads();
        reduction_helper_real[tidx] = norm;
        reduce(group::this_thread_block(), reduction_helper_real,
               [](const real_arithmetic_type& a,
                  const real_arithmetic_type& b) { return a + b; });
        __syncthreads();

        norm = sqrt(reduction_helper_real[0]);
        for (size_type j = tidx; j < num_cols; j += block_size) {
            values[row * stride + j] /= norm;
        }
    }
}


template <typename MValuesAccessor, typename FValuesAccessor,
          typename CValuesAccessor>
__global__
__launch_bounds__(default_block_size) void solve_lower_triangular_kernel(
    size_type subspace_dim, size_type nrhs, MValuesAccessor m_values,
    size_type m_stride, FValuesAccessor f_values, size_type f_stride,
    CValuesAccessor c_values, size_type c_stride,
    const stopping_status* __restrict__ stop_status)
{
    const auto global_id = thread::get_thread_id_flat();

    if (global_id >= nrhs) {
        return;
    }

    if (!stop_status[global_id].has_stopped()) {
        for (size_type row = 0; row < subspace_dim; row++) {
            auto temp = f_values[row * f_stride + global_id];
            for (size_type col = 0; col < row; col++) {
                temp -= m_values[row * m_stride + col * nrhs + global_id] *
                        c_values[col * c_stride + global_id];
            }
            c_values[row * c_stride + global_id] =
                temp / m_values[row * m_stride + row * nrhs + global_id];
        }
    }
}


template <typename ResidualAccessor, typename CAccessor, typename GAccessor,
          typename OutputAccessor>
__global__ __launch_bounds__(default_block_size) void step_1_kernel(
    size_type k, size_type num_rows, size_type subspace_dim, size_type nrhs,
    ResidualAccessor residual_values, size_type residual_stride,
    CAccessor c_values, size_type c_stride, GAccessor g_values,
    size_type g_stride, OutputAccessor v_values, size_type v_stride,
    const stopping_status* __restrict__ stop_status)
{
    const auto global_id = thread::get_thread_id_flat();
    const auto row = global_id / nrhs;
    const auto col = global_id % nrhs;

    if (row >= num_rows) {
        return;
    }

    if (!stop_status[col].has_stopped()) {
        auto temp = residual_values[row * residual_stride + col];
        for (size_type j = k; j < subspace_dim; j++) {
            temp -= c_values[j * c_stride + col] *
                    g_values[row * g_stride + j * nrhs + col];
        }
        v_values[row * v_stride + col] = temp;
    }
}


template <typename OmegaValuesAccessor, typename VValuesAccessor,
          typename CValuesAccessor, typename UValuesAccessor>
__global__ __launch_bounds__(default_block_size) void step_2_kernel(
    size_type k, size_type num_rows, size_type subspace_dim, size_type nrhs,
    OmegaValuesAccessor omega_values, VValuesAccessor v_values,
    size_type v_stride, CValuesAccessor c_values, size_type c_stride,
    UValuesAccessor u_values, size_type u_stride,
    const stopping_status* __restrict__ stop_status)
{
    const auto global_id = thread::get_thread_id_flat();
    const auto row = global_id / nrhs;
    const auto col = global_id % nrhs;

    if (row >= num_rows) {
        return;
    }

    if (!stop_status[col].has_stopped()) {
        auto temp = omega_values[col] * v_values[row * v_stride + col];
        for (size_type j = k; j < subspace_dim; j++) {
            temp += c_values[j * c_stride + col] *
                    u_values[row * u_stride + j * nrhs + col];
        }
        u_values[row * u_stride + k * nrhs + col] = temp;
    }
}


// alpha is written with atomic_add, which takes the storage type
template <typename PIAccessor, typename GKAccessor, typename ValueType>
__global__ __launch_bounds__(default_dot_size) void multidot_kernel(
    size_type num_rows, size_type nrhs, PIAccessor p_i, GKAccessor g_k,
    size_type g_k_stride, ValueType* __restrict__ alpha,
    const stopping_status* __restrict__ stop_status)
{
    const auto tidx = threadIdx.x;
    const auto tidy = threadIdx.y;
    const auto rhs = blockIdx.x * default_dot_dim + tidx;
    const auto num = ceildiv(num_rows, gridDim.y);
    const auto start_row = blockIdx.y * num;
    const auto end_row =
        ((blockIdx.y + 1) * num > num_rows) ? num_rows : (blockIdx.y + 1) * num;
    // Used that way to get around dynamic initialization warning and
    // template error when using `reduction_helper_array` directly in `reduce`
    using arithmetic_type = arithmetic_type_t<PIAccessor>;
    __shared__ uninitialized_array<arithmetic_type,
                                   default_dot_dim*(default_dot_dim + 1)>
        reduction_helper_array;
    arithmetic_type* __restrict__ reduction_helper = reduction_helper_array;

    arithmetic_type local_res = zero<arithmetic_type>();
    if (rhs < nrhs && !stop_status[rhs].has_stopped()) {
        for (size_type i = start_row + tidy; i < end_row;
             i += default_dot_dim) {
            const auto g_idx = i * g_k_stride + rhs;
            local_res += p_i[i] * g_k[g_idx];
        }
    }
    reduction_helper[tidx * (default_dot_dim + 1) + tidy] = local_res;
    __syncthreads();
    local_res = reduction_helper[tidy * (default_dot_dim + 1) + tidx];
    const auto tile_block =
        group::tiled_partition<default_dot_dim>(group::this_thread_block());
    const auto sum = reduce(tile_block, local_res,
                            [](const arithmetic_type& a,
                               const arithmetic_type& b) { return a + b; });
    const auto new_rhs = blockIdx.x * default_dot_dim + tidy;
    if (tidx == 0 && new_rhs < nrhs && !stop_status[new_rhs].has_stopped()) {
        atomic_add(alpha + new_rhs, static_cast<ValueType>(sum));
    }
}


template <size_type block_size, typename AlphaAccessor,
          typename MValuesAccessor, typename GValuesAccessor,
          typename GKValuesAccessor, typename UValuesAccessor>
__global__ __launch_bounds__(block_size) void update_g_k_and_u_kernel(
    size_type k, size_type i, size_type size, size_type nrhs,
    AlphaAccessor alpha, MValuesAccessor m_values, size_type m_stride,
    GValuesAccessor g_values, size_type g_stride, GKValuesAccessor g_k_values,
    size_type g_k_stride, UValuesAccessor u_values, size_type u_stride,
    const stopping_status* __restrict__ stop_status)
{
    const auto tidx = thread::get_thread_id_flat();
    const auto row = tidx / g_k_stride;
    const auto rhs = tidx % g_k_stride;

    if (row >= size || rhs >= nrhs) {
        return;
    }

    if (!stop_status[rhs].has_stopped()) {
        const auto fact = alpha[rhs] / m_values[i * m_stride + i * nrhs + rhs];
        g_k_values[row * g_k_stride + rhs] -=
            fact * g_values[row * g_stride + i * nrhs + rhs];
        u_values[row * u_stride + k * nrhs + rhs] -=
            fact * u_values[row * u_stride + i * nrhs + rhs];
    }
}


template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void update_g_kernel(
    size_type k, size_type size, size_type nrhs,
    const ValueType* __restrict__ g_k_values, size_type g_k_stride,
    ValueType* __restrict__ g_values, size_type g_stride,
    const stopping_status* __restrict__ stop_status)
{
    const auto tidx = thread::get_thread_id_flat();
    const auto row = tidx / g_k_stride;
    const auto rhs = tidx % nrhs;

    if (row >= size || rhs >= nrhs) {
        return;
    }

    if (!stop_status[rhs].has_stopped()) {
        g_values[row * g_stride + k * nrhs + rhs] =
            g_k_values[row * g_k_stride + rhs];
    }
}


template <typename MValuesAccessor, typename GValuesAccessor,
          typename UValuesAccessor, typename FValuesAccessor,
          typename RValuesAccessor, typename XValuesAccessor>
__global__ __launch_bounds__(default_block_size) void update_x_r_and_f_kernel(
    size_type k, size_type size, size_type subspace_dim, size_type nrhs,
    MValuesAccessor m_values, size_type m_stride, GValuesAccessor g_values,
    size_type g_stride, UValuesAccessor u_values, size_type u_stride,
    FValuesAccessor f_values, size_type f_stride, RValuesAccessor r_values,
    size_type r_stride, XValuesAccessor x_values, size_type x_stride,
    const stopping_status* __restrict__ stop_status)
{
    const auto global_id = thread::get_thread_id_flat();
    const auto row = global_id / x_stride;
    const auto col = global_id % x_stride;

    if (row >= size || col >= nrhs) {
        return;
    }

    if (!stop_status[col].has_stopped()) {
        const auto beta = f_values[k * f_stride + col] /
                          m_values[k * m_stride + k * nrhs + col];
        r_values[row * r_stride + col] -=
            beta * g_values[row * g_stride + k * nrhs + col];
        x_values[row * x_stride + col] +=
            beta * u_values[row * u_stride + k * nrhs + col];

        if (k < row && k + 1 < subspace_dim && row < subspace_dim) {
            f_values[row * f_stride + col] -=
                beta * m_values[row * m_stride + k * nrhs + col];
        }
    }
}


template <typename RealArithmeticType, typename ValueAccessor,
          typename NormAccessor, typename OutputAccessor>
__global__ __launch_bounds__(config::warp_size) void compute_omega_kernel(
    size_type nrhs, const RealArithmeticType kappa, ValueAccessor tht,
    NormAccessor residual_norm, OutputAccessor omega,
    const stopping_status* __restrict__ stop_status)
{
    using arithmetic_type = arithmetic_type_t<OutputAccessor>;
    const auto global_id = thread::get_thread_id_flat();

    if (global_id >= nrhs) {
        return;
    }

    if (!stop_status[global_id].has_stopped()) {
        auto thr = static_cast<arithmetic_type>(omega[global_id]);
        const auto normt = sqrt(real(tht[global_id]));
        if (normt == zero<RealArithmeticType>()) {
            omega[global_id] = zero<arithmetic_type>();
            return;
        }
        omega[global_id] /= tht[global_id];
        auto absrho = abs(thr / (normt * residual_norm[global_id]));

        if (absrho < kappa) {
            omega[global_id] *= kappa / absrho;
        }
    }
}


namespace {


template <typename ValueType>
void initialize_m(std::shared_ptr<const DefaultExecutor> exec,
                  const size_type nrhs, matrix::view::dense<ValueType> m,
                  array<stopping_status>& stop_status)
{
    const auto subspace_dim = m.size[0];
    const auto m_stride = m.stride;

    const auto grid_dim = ceildiv(m_stride * subspace_dim, default_block_size);
    initialize_m_kernel<<<grid_dim, default_block_size, 0,
                          exec->get_stream()>>>(
        subspace_dim, nrhs, as_device_type(m.values), m_stride,
        as_device_type(stop_status.get_data()));
}


template <typename ValueType>
void initialize_subspace_vectors(
    std::shared_ptr<const DefaultExecutor> exec,
    matrix::view::dense<ValueType> subspace_vectors, bool deterministic)
{
    if (!deterministic) {
        auto gen = randlib::rand_generator(std::random_device{}(),
                                           RANDLIB_RNG_PSEUDO_DEFAULT,
                                           exec->get_stream());
        randlib::rand_vector(gen,
                             subspace_vectors.size[0] * subspace_vectors.stride,
                             0.0, 1.0, subspace_vectors.values);
        randlib::destroy(gen);
    }
}


template <typename ValueType>
void orthonormalize_subspace_vectors(
    std::shared_ptr<const DefaultExecutor> exec,
    matrix::view::dense<ValueType> subspace_vectors)
{
    const auto subspace_vectors_vals =
        arithmetic<ValueType, 1>(subspace_vectors);
    orthonormalize_subspace_vectors_kernel<default_block_size>
        <<<1, default_block_size, 0, exec->get_stream()>>>(
            subspace_vectors.size[0], subspace_vectors.size[1],
            acc::as_device_range(subspace_vectors_vals),
            subspace_vectors.stride);
}


template <typename ValueType>
void solve_lower_triangular(std::shared_ptr<const DefaultExecutor> exec,
                            const size_type nrhs,
                            matrix::view::dense<const ValueType> m,
                            matrix::view::dense<const ValueType> f,
                            matrix::view::dense<ValueType> c,
                            const array<stopping_status>& stop_status)
{
    const auto subspace_dim = m.size[0];

    const auto grid_dim = ceildiv(nrhs, default_block_size);
    const auto m_vals = arithmetic<ValueType, 1>(m);
    const auto f_vals = arithmetic<ValueType, 1>(f);
    const auto c_vals = arithmetic<ValueType, 1>(c);
    solve_lower_triangular_kernel<<<grid_dim, default_block_size, 0,
                                    exec->get_stream()>>>(
        subspace_dim, nrhs, acc::as_device_range(m_vals), m.stride,
        acc::as_device_range(f_vals), f.stride, acc::as_device_range(c_vals),
        c.stride, stop_status.get_const_data());
}


template <typename ValueType>
void update_g_and_u(std::shared_ptr<const DefaultExecutor> exec,
                    const size_type nrhs, const size_type k,
                    matrix::view::dense<const ValueType> p,
                    matrix::view::dense<const ValueType> m,
                    matrix::view::dense<ValueType> alpha,
                    matrix::view::dense<ValueType> g,
                    matrix::view::dense<ValueType> g_k,
                    matrix::view::dense<ValueType> u,
                    const array<stopping_status>& stop_status)
{
    if (nrhs == 0) {
        return;
    }
    const auto size = g.size[0];
    const auto p_stride = p.stride;

    const dim3 grid_dim(ceildiv(nrhs, default_dot_dim),
                        exec->get_num_multiprocessor() * 2);
    const dim3 block_dim(default_dot_dim, default_dot_dim);

    for (size_type i = 0; i < k; i++) {
        const auto p_i = p.values + i * p_stride;
        if (nrhs > 1 || is_complex<ValueType>()) {
            components::fill_array(exec, alpha.values, nrhs, zero<ValueType>());
// not support 16 bit atomic
#if !defined(CUDA_VERSION)
            if constexpr (sizeof(remove_complex<ValueType>) == sizeof(int16)) {
                GKO_NOT_SUPPORTED(alpha);
            } else
#else
            const auto compute_capability =
                as<CudaExecutor>(exec)->get_compute_capability();
            if (compute_capability < 70 &&
                std::is_same_v<remove_complex<ValueType>, half>) {
                GKO_NOT_SUPPORTED(alpha);
            } else if (compute_capability < 80 &&
                       std::is_same_v<remove_complex<ValueType>, bfloat16>) {
                GKO_NOT_SUPPORTED(alpha);
            } else
#endif
            {
                const auto p_i_vals = arithmetic<ValueType>(p_i, size);
                const auto g_k_vals = arithmetic<ValueType, 1>(g_k);
                multidot_kernel<<<grid_dim, block_dim, 0, exec->get_stream()>>>(
                    size, nrhs, acc::as_device_range(p_i_vals),
                    acc::as_device_range(g_k_vals), g_k.stride,
                    as_device_type(alpha.values), stop_status.get_const_data());
            }
        } else {
            blas::dot(exec->get_blas_handle(), size, p_i, 1, g_k.values,
                      g_k.stride, alpha.values);
        }
        const auto alpha_vals = arithmetic<ValueType, 1>(alpha);
        const auto m_vals = arithmetic<ValueType, 1>(m);
        const auto g_vals = arithmetic<ValueType, 1>(g);
        const auto g_k_vals = arithmetic<ValueType, 1>(g_k);
        const auto u_vals = arithmetic<ValueType, 1>(u);
        update_g_k_and_u_kernel<default_block_size>
            <<<ceildiv(size * g_k.stride, default_block_size),
               default_block_size, 0, exec->get_stream()>>>(
                k, i, size, nrhs, acc::as_device_range(alpha_vals),
                acc::as_device_range(m_vals), m.stride,
                acc::as_device_range(g_vals), g.stride,
                acc::as_device_range(g_k_vals), g_k.stride,
                acc::as_device_range(u_vals), u.stride,
                stop_status.get_const_data());
    }
    update_g_kernel<default_block_size>
        <<<ceildiv(size * g_k.stride, default_block_size), default_block_size,
           0, exec->get_stream()>>>(k, size, nrhs, as_device_type(g_k.values),
                                    g_k.stride, as_device_type(g.values),
                                    g.stride, stop_status.get_const_data());
}


template <typename ValueType>
void update_m(std::shared_ptr<const DefaultExecutor> exec, const size_type nrhs,
              const size_type k, matrix::view::dense<const ValueType> p,
              matrix::view::dense<const ValueType> g_k,
              matrix::view::dense<ValueType> m,
              const array<stopping_status>& stop_status)
{
    if (nrhs == 0) {
        return;
    }
    const auto size = g_k.size[0];
    const auto subspace_dim = m.size[0];
    const auto p_stride = p.stride;
    const auto m_stride = m.stride;

    const dim3 grid_dim(ceildiv(nrhs, default_dot_dim),
                        exec->get_num_multiprocessor() * 2);
    const dim3 block_dim(default_dot_dim, default_dot_dim);

    for (size_type i = k; i < subspace_dim; i++) {
        const auto p_i = p.values + i * p_stride;
        auto m_i = m.values + i * m_stride + k * nrhs;
        if (nrhs > 1 || is_complex<ValueType>()) {
            components::fill_array(exec, m_i, nrhs, zero<ValueType>());

// not support 16 bit atomic
#if !defined(CUDA_VERSION)
            if constexpr (sizeof(remove_complex<ValueType>) == sizeof(int16)) {
                GKO_NOT_SUPPORTED(m_i);
            } else
#else
            const auto compute_capability =
                as<CudaExecutor>(exec)->get_compute_capability();
            if (compute_capability < 70 &&
                std::is_same_v<remove_complex<ValueType>, half>) {
                GKO_NOT_SUPPORTED(m_i);
            } else if (compute_capability < 80 &&
                       std::is_same_v<remove_complex<ValueType>, bfloat16>) {
                GKO_NOT_SUPPORTED(m_i);
            } else
#endif
            {
                const auto p_i_vals = arithmetic<ValueType>(p_i, size);
                const auto g_k_vals = arithmetic<ValueType, 1>(g_k);
                multidot_kernel<<<grid_dim, block_dim, 0, exec->get_stream()>>>(
                    size, nrhs, acc::as_device_range(p_i_vals),
                    acc::as_device_range(g_k_vals), g_k.stride,
                    as_device_type(m_i), stop_status.get_const_data());
            }
        } else {
            blas::dot(exec->get_blas_handle(), size, p_i, 1, g_k.values,
                      g_k.stride, m_i);
        }
    }
}


template <typename ValueType>
void update_x_r_and_f(std::shared_ptr<const DefaultExecutor> exec,
                      const size_type nrhs, const size_type k,
                      matrix::view::dense<const ValueType> m,
                      matrix::view::dense<const ValueType> g,
                      matrix::view::dense<const ValueType> u,
                      matrix::view::dense<ValueType> f,
                      matrix::view::dense<ValueType> r,
                      matrix::view::dense<ValueType> x,
                      const array<stopping_status>& stop_status)
{
    const auto size = x.size[0];
    const auto subspace_dim = m.size[0];

    const auto grid_dim = ceildiv(size * x.stride, default_block_size);
    const auto m_vals = arithmetic<ValueType, 1>(m);
    const auto g_vals = arithmetic<ValueType, 1>(g);
    const auto u_vals = arithmetic<ValueType, 1>(u);
    const auto f_vals = arithmetic<ValueType, 1>(f);
    const auto r_vals = arithmetic<ValueType, 1>(r);
    const auto x_vals = arithmetic<ValueType, 1>(x);
    update_x_r_and_f_kernel<<<grid_dim, default_block_size, 0,
                              exec->get_stream()>>>(
        k, size, subspace_dim, nrhs, acc::as_device_range(m_vals), m.stride,
        acc::as_device_range(g_vals), g.stride, acc::as_device_range(u_vals),
        u.stride, acc::as_device_range(f_vals), f.stride,
        acc::as_device_range(r_vals), r.stride, acc::as_device_range(x_vals),
        x.stride, stop_status.get_const_data());
    components::fill_array(exec, f.values + k * f.stride, nrhs,
                           zero<ValueType>());
}


}  // namespace


template <typename ValueType>
void initialize(std::shared_ptr<const DefaultExecutor> exec,
                const size_type nrhs, matrix::view::dense<ValueType> m,
                matrix::view::dense<ValueType> subspace_vectors,
                bool deterministic, array<stopping_status>& stop_status)
{
    initialize_m(exec, nrhs, m, stop_status);
    initialize_subspace_vectors(exec, subspace_vectors, deterministic);
    orthonormalize_subspace_vectors(exec, subspace_vectors);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR_INITIALIZE_KERNEL);


template <typename ValueType>
void step_1(std::shared_ptr<const DefaultExecutor> exec, const size_type nrhs,
            const size_type k, matrix::view::dense<const ValueType> m,
            matrix::view::dense<const ValueType> f,
            matrix::view::dense<const ValueType> residual,
            matrix::view::dense<const ValueType> g,
            matrix::view::dense<ValueType> c, matrix::view::dense<ValueType> v,
            const array<stopping_status>& stop_status)
{
    solve_lower_triangular(exec, nrhs, m, f, c, stop_status);

    const auto num_rows = v.size[0];
    const auto subspace_dim = m.size[0];

    const auto grid_dim = ceildiv(nrhs * num_rows, default_block_size);
    const auto residual_vals = arithmetic<ValueType, 1>(residual);
    const auto c_vals = arithmetic<ValueType, 1>(c);
    const auto g_vals = arithmetic<ValueType, 1>(g);
    const auto v_vals = arithmetic<ValueType, 1>(v);
    step_1_kernel<<<grid_dim, default_block_size, 0, exec->get_stream()>>>(
        k, num_rows, subspace_dim, nrhs, acc::as_device_range(residual_vals),
        residual.stride, acc::as_device_range(c_vals), c.stride,
        acc::as_device_range(g_vals), g.stride, acc::as_device_range(v_vals),
        v.stride, stop_status.get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR_STEP_1_KERNEL);


template <typename ValueType>
void step_2(std::shared_ptr<const DefaultExecutor> exec, const size_type nrhs,
            const size_type k, matrix::view::dense<const ValueType> omega,
            matrix::view::dense<const ValueType> preconditioned_vector,
            matrix::view::dense<const ValueType> c,
            matrix::view::dense<ValueType> u,
            const array<stopping_status>& stop_status)
{
    if (nrhs == 0) {
        return;
    }
    const auto num_rows = preconditioned_vector.size[0];
    const auto subspace_dim = u.size[1] / nrhs;

    const auto grid_dim = ceildiv(nrhs * num_rows, default_block_size);
    const auto omega_vals = arithmetic<ValueType, 1>(omega);
    const auto preconditioned_vector_vals =
        arithmetic<ValueType, 1>(preconditioned_vector);
    const auto c_vals = arithmetic<ValueType, 1>(c);
    const auto u_vals = arithmetic<ValueType, 1>(u);
    step_2_kernel<<<grid_dim, default_block_size, 0, exec->get_stream()>>>(
        k, num_rows, subspace_dim, nrhs, acc::as_device_range(omega_vals),
        acc::as_device_range(preconditioned_vector_vals),
        preconditioned_vector.stride, acc::as_device_range(c_vals), c.stride,
        acc::as_device_range(u_vals), u.stride, stop_status.get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR_STEP_2_KERNEL);


template <typename ValueType>
void step_3(std::shared_ptr<const DefaultExecutor> exec, const size_type nrhs,
            const size_type k, matrix::view::dense<const ValueType> p,
            matrix::view::dense<ValueType> g,
            matrix::view::dense<ValueType> g_k,
            matrix::view::dense<ValueType> u, matrix::view::dense<ValueType> m,
            matrix::view::dense<ValueType> f,
            matrix::view::dense<ValueType> alpha,
            matrix::view::dense<ValueType> residual,
            matrix::view::dense<ValueType> x,
            const array<stopping_status>& stop_status)
{
    update_g_and_u(exec, nrhs, k, p, m.as_const(), alpha, g, g_k, u,
                   stop_status);
    update_m(exec, nrhs, k, p.as_const(), g_k.as_const(), m, stop_status);
    update_x_r_and_f(exec, nrhs, k, m.as_const(), g.as_const(), u.as_const(), f,
                     residual, x, stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR_STEP_3_KERNEL);


template <typename ValueType>
void compute_omega(
    std::shared_ptr<const DefaultExecutor> exec, const size_type nrhs,
    const remove_complex<ValueType> kappa,
    matrix::view::dense<const ValueType> tht,
    matrix::view::dense<const remove_complex<ValueType>> residual_norm,
    matrix::view::dense<ValueType> omega,
    const array<stopping_status>& stop_status)
{
    const auto grid_dim = ceildiv(nrhs, config::warp_size);
    const auto tht_vals = arithmetic<ValueType, 1>(tht);
    const auto residual_norm_vals = arithmetic<ValueType, 1>(residual_norm);
    const auto omega_vals = arithmetic<ValueType, 1>(omega);
    compute_omega_kernel<<<grid_dim, config::warp_size, 0,
                           exec->get_stream()>>>(
        nrhs, arithmetic<ValueType>(kappa), acc::as_device_range(tht_vals),
        acc::as_device_range(residual_norm_vals),
        acc::as_device_range(omega_vals), stop_status.get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR_COMPUTE_OMEGA_KERNEL);


}  // namespace idr
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
