// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/idr_kernels.hpp"

#include <ctime>
#include <random>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>

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
#include "core/components/fill_array_kernels.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The IDR solver namespace.
 *
 * @ingroup idr
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


template <size_type block_size, typename ValueType>
__global__
__launch_bounds__(block_size) void orthonormalize_subspace_vectors_kernel(
    size_type num_rows, size_type num_cols, ValueType* __restrict__ values,
    size_type stride)
{
    const auto tidx = thread::get_thread_id_flat();

    __shared__ uninitialized_array<ValueType, block_size>
        reduction_helper_array;
    // they are not be used in the same time.
    ValueType* reduction_helper = reduction_helper_array;
    auto reduction_helper_real =
        reinterpret_cast<remove_complex<ValueType>*>(reduction_helper);

    for (size_type row = 0; row < num_rows; row++) {
        for (size_type i = 0; i < row; i++) {
            auto dot = zero<ValueType>();
            for (size_type j = tidx; j < num_cols; j += block_size) {
                dot += values[row * stride + j] * conj(values[i * stride + j]);
            }

            // Ensure already finish reading this shared memory
            __syncthreads();
            reduction_helper[tidx] = dot;
            reduce(
                group::this_thread_block(), reduction_helper,
                [](const ValueType& a, const ValueType& b) { return a + b; });
            __syncthreads();

            dot = reduction_helper[0];
            for (size_type j = tidx; j < num_cols; j += block_size) {
                values[row * stride + j] -= dot * values[i * stride + j];
            }
        }

        auto norm = zero<remove_complex<ValueType>>();
        for (size_type j = tidx; j < num_cols; j += block_size) {
            norm += squared_norm(values[row * stride + j]);
        }
        // Ensure already finish reading this shared memory
        __syncthreads();
        reduction_helper_real[tidx] = norm;
        reduce(group::this_thread_block(), reduction_helper_real,
               [](const remove_complex<ValueType>& a,
                  const remove_complex<ValueType>& b) { return a + b; });
        __syncthreads();

        norm = sqrt(reduction_helper_real[0]);
        for (size_type j = tidx; j < num_cols; j += block_size) {
            values[row * stride + j] /= norm;
        }
    }
}


template <typename ValueType>
__global__
__launch_bounds__(default_block_size) void solve_lower_triangular_kernel(
    size_type subspace_dim, size_type nrhs,
    const ValueType* __restrict__ m_values, size_type m_stride,
    const ValueType* __restrict__ f_values, size_type f_stride,
    ValueType* __restrict__ c_values, size_type c_stride,
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


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void step_1_kernel(
    size_type k, size_type num_rows, size_type subspace_dim, size_type nrhs,
    const ValueType* __restrict__ residual_values, size_type residual_stride,
    const ValueType* __restrict__ c_values, size_type c_stride,
    const ValueType* __restrict__ g_values, size_type g_stride,
    ValueType* __restrict__ v_values, size_type v_stride,
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


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void step_2_kernel(
    size_type k, size_type num_rows, size_type subspace_dim, size_type nrhs,
    const ValueType* __restrict__ omega_values,
    const ValueType* __restrict__ v_values, size_type v_stride,
    const ValueType* __restrict__ c_values, size_type c_stride,
    ValueType* __restrict__ u_values, size_type u_stride,
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


template <typename ValueType>
__global__ __launch_bounds__(default_dot_size) void multidot_kernel(
    size_type num_rows, size_type nrhs, const ValueType* __restrict__ p_i,
    const ValueType* __restrict__ g_k, size_type g_k_stride,
    ValueType* __restrict__ alpha,
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
    __shared__
        uninitialized_array<ValueType, default_dot_dim*(default_dot_dim + 1)>
            reduction_helper_array;
    ValueType* __restrict__ reduction_helper = reduction_helper_array;

    ValueType local_res = zero<ValueType>();
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
    const auto sum =
        reduce(tile_block, local_res,
               [](const ValueType& a, const ValueType& b) { return a + b; });
    const auto new_rhs = blockIdx.x * default_dot_dim + tidy;
    if (tidx == 0 && new_rhs < nrhs && !stop_status[new_rhs].has_stopped()) {
        atomic_add(alpha + new_rhs, sum);
    }
}


template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void update_g_k_and_u_kernel(
    size_type k, size_type i, size_type size, size_type nrhs,
    const ValueType* __restrict__ alpha, const ValueType* __restrict__ m_values,
    size_type m_stride, const ValueType* __restrict__ g_values,
    size_type g_stride, ValueType* __restrict__ g_k_values,
    size_type g_k_stride, ValueType* __restrict__ u_values, size_type u_stride,
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


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void update_x_r_and_f_kernel(
    size_type k, size_type size, size_type subspace_dim, size_type nrhs,
    const ValueType* __restrict__ m_values, size_type m_stride,
    const ValueType* __restrict__ g_values, size_type g_stride,
    const ValueType* __restrict__ u_values, size_type u_stride,
    ValueType* __restrict__ f_values, size_type f_stride,
    ValueType* __restrict__ r_values, size_type r_stride,
    ValueType* __restrict__ x_values, size_type x_stride,
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


template <typename ValueType>
__global__ __launch_bounds__(config::warp_size) void compute_omega_kernel(
    size_type nrhs, const remove_complex<ValueType> kappa,
    const ValueType* __restrict__ tht,
    const remove_complex<ValueType>* __restrict__ residual_norm,
    ValueType* __restrict__ omega,
    const stopping_status* __restrict__ stop_status)
{
    const auto global_id = thread::get_thread_id_flat();

    if (global_id >= nrhs) {
        return;
    }

    if (!stop_status[global_id].has_stopped()) {
        auto thr = omega[global_id];
        const auto normt = sqrt(real(tht[global_id]));
        if (normt == zero<remove_complex<ValueType>>()) {
            omega[global_id] = zero<ValueType>();
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
                  const size_type nrhs, matrix::Dense<ValueType>* m,
                  array<stopping_status>* stop_status)
{
    const auto subspace_dim = m->get_size()[0];
    const auto m_stride = m->get_stride();

    const auto grid_dim = ceildiv(m_stride * subspace_dim, default_block_size);
    initialize_m_kernel<<<grid_dim, default_block_size, 0,
                          exec->get_stream()>>>(
        subspace_dim, nrhs, as_device_type(m->get_values()), m_stride,
        as_device_type(stop_status->get_data()));
}


template <typename ValueType>
void initialize_subspace_vectors(std::shared_ptr<const DefaultExecutor> exec,
                                 matrix::Dense<ValueType>* subspace_vectors,
                                 bool deterministic)
{
    if (!deterministic) {
        auto gen = randlib::rand_generator(std::random_device{}(),
                                           RANDLIB_RNG_PSEUDO_DEFAULT,
                                           exec->get_stream());
        randlib::rand_vector(
            gen,
            subspace_vectors->get_size()[0] * subspace_vectors->get_stride(),
            0.0, 1.0, subspace_vectors->get_values());
        randlib::destroy(gen);
    }
}


template <typename ValueType>
void orthonormalize_subspace_vectors(
    std::shared_ptr<const DefaultExecutor> exec,
    matrix::Dense<ValueType>* subspace_vectors)
{
    orthonormalize_subspace_vectors_kernel<default_block_size>
        <<<1, default_block_size, 0, exec->get_stream()>>>(
            subspace_vectors->get_size()[0], subspace_vectors->get_size()[1],
            as_device_type(subspace_vectors->get_values()),
            subspace_vectors->get_stride());
}


template <typename ValueType>
void solve_lower_triangular(std::shared_ptr<const DefaultExecutor> exec,
                            const size_type nrhs,
                            const matrix::Dense<ValueType>* m,
                            const matrix::Dense<ValueType>* f,
                            matrix::Dense<ValueType>* c,
                            const array<stopping_status>* stop_status)
{
    const auto subspace_dim = m->get_size()[0];

    const auto grid_dim = ceildiv(nrhs, default_block_size);
    solve_lower_triangular_kernel<<<grid_dim, default_block_size, 0,
                                    exec->get_stream()>>>(
        subspace_dim, nrhs, as_device_type(m->get_const_values()),
        m->get_stride(), as_device_type(f->get_const_values()), f->get_stride(),
        as_device_type(c->get_values()), c->get_stride(),
        stop_status->get_const_data());
}


template <typename ValueType>
void update_g_and_u(std::shared_ptr<const DefaultExecutor> exec,
                    const size_type nrhs, const size_type k,
                    const matrix::Dense<ValueType>* p,
                    const matrix::Dense<ValueType>* m,
                    matrix::Dense<ValueType>* alpha,
                    matrix::Dense<ValueType>* g, matrix::Dense<ValueType>* g_k,
                    matrix::Dense<ValueType>* u,
                    const array<stopping_status>* stop_status)
{
    if (nrhs == 0) {
        return;
    }
    const auto size = g->get_size()[0];
    const auto p_stride = p->get_stride();

    const dim3 grid_dim(ceildiv(nrhs, default_dot_dim),
                        exec->get_num_multiprocessor() * 2);
    const dim3 block_dim(default_dot_dim, default_dot_dim);

    for (size_type i = 0; i < k; i++) {
        const auto p_i = p->get_const_values() + i * p_stride;
        if (nrhs > 1 || is_complex<ValueType>()) {
            components::fill_array(exec, alpha->get_values(), nrhs,
                                   zero<ValueType>());
            // not support 16 bit atomic
#if !(defined(CUDA_VERSION) && (__CUDA_ARCH__ >= 700))
            if constexpr (sizeof(remove_complex<ValueType>) == sizeof(int16)) {
                GKO_NOT_SUPPORTED(alpha);
            } else
#endif
            {
                multidot_kernel<<<grid_dim, block_dim, 0, exec->get_stream()>>>(
                    size, nrhs, as_device_type(p_i),
                    as_device_type(g_k->get_values()), g_k->get_stride(),
                    as_device_type(alpha->get_values()),
                    stop_status->get_const_data());
            }
        } else {
            blas::dot(exec->get_blas_handle(), size, p_i, 1, g_k->get_values(),
                      g_k->get_stride(), alpha->get_values());
        }
        update_g_k_and_u_kernel<default_block_size>
            <<<ceildiv(size * g_k->get_stride(), default_block_size),
               default_block_size, 0, exec->get_stream()>>>(
                k, i, size, nrhs, as_device_type(alpha->get_const_values()),
                as_device_type(m->get_const_values()), m->get_stride(),
                as_device_type(g->get_const_values()), g->get_stride(),
                as_device_type(g_k->get_values()), g_k->get_stride(),
                as_device_type(u->get_values()), u->get_stride(),
                stop_status->get_const_data());
    }
    update_g_kernel<default_block_size>
        <<<ceildiv(size * g_k->get_stride(), default_block_size),
           default_block_size, 0, exec->get_stream()>>>(
            k, size, nrhs, as_device_type(g_k->get_const_values()),
            g_k->get_stride(), as_device_type(g->get_values()), g->get_stride(),
            stop_status->get_const_data());
}


template <typename ValueType>
void update_m(std::shared_ptr<const DefaultExecutor> exec, const size_type nrhs,
              const size_type k, const matrix::Dense<ValueType>* p,
              const matrix::Dense<ValueType>* g_k, matrix::Dense<ValueType>* m,
              const array<stopping_status>* stop_status)
{
    if (nrhs == 0) {
        return;
    }
    const auto size = g_k->get_size()[0];
    const auto subspace_dim = m->get_size()[0];
    const auto p_stride = p->get_stride();
    const auto m_stride = m->get_stride();

    const dim3 grid_dim(ceildiv(nrhs, default_dot_dim),
                        exec->get_num_multiprocessor() * 2);
    const dim3 block_dim(default_dot_dim, default_dot_dim);

    for (size_type i = k; i < subspace_dim; i++) {
        const auto p_i = p->get_const_values() + i * p_stride;
        auto m_i = m->get_values() + i * m_stride + k * nrhs;
        if (nrhs > 1 || is_complex<ValueType>()) {
            components::fill_array(exec, m_i, nrhs, zero<ValueType>());
            // not support 16 bit atomic
#if !(defined(CUDA_VERSION) && (__CUDA_ARCH__ >= 700))
            if constexpr (std::is_same_v<remove_complex<ValueType>, float16>) {
                GKO_NOT_SUPPORTED(m_i);
            } else
#endif
            {
                multidot_kernel<<<grid_dim, block_dim, 0, exec->get_stream()>>>(
                    size, nrhs, as_device_type(p_i),
                    as_device_type(g_k->get_const_values()), g_k->get_stride(),
                    as_device_type(m_i), stop_status->get_const_data());
            }
        } else {
            blas::dot(exec->get_blas_handle(), size, p_i, 1,
                      g_k->get_const_values(), g_k->get_stride(), m_i);
        }
    }
}


template <typename ValueType>
void update_x_r_and_f(std::shared_ptr<const DefaultExecutor> exec,
                      const size_type nrhs, const size_type k,
                      const matrix::Dense<ValueType>* m,
                      const matrix::Dense<ValueType>* g,
                      const matrix::Dense<ValueType>* u,
                      matrix::Dense<ValueType>* f, matrix::Dense<ValueType>* r,
                      matrix::Dense<ValueType>* x,
                      const array<stopping_status>* stop_status)
{
    const auto size = x->get_size()[0];
    const auto subspace_dim = m->get_size()[0];

    const auto grid_dim = ceildiv(size * x->get_stride(), default_block_size);
    update_x_r_and_f_kernel<<<grid_dim, default_block_size, 0,
                              exec->get_stream()>>>(
        k, size, subspace_dim, nrhs, as_device_type(m->get_const_values()),
        m->get_stride(), as_device_type(g->get_const_values()), g->get_stride(),
        as_device_type(u->get_const_values()), u->get_stride(),
        as_device_type(f->get_values()), f->get_stride(),
        as_device_type(r->get_values()), r->get_stride(),
        as_device_type(x->get_values()), x->get_stride(),
        stop_status->get_const_data());
    components::fill_array(exec, f->get_values() + k * f->get_stride(), nrhs,
                           zero<ValueType>());
}


}  // namespace


template <typename ValueType>
void initialize(std::shared_ptr<const DefaultExecutor> exec,
                const size_type nrhs, matrix::Dense<ValueType>* m,
                matrix::Dense<ValueType>* subspace_vectors, bool deterministic,
                array<stopping_status>* stop_status)
{
    initialize_m(exec, nrhs, m, stop_status);
    initialize_subspace_vectors(exec, subspace_vectors, deterministic);
    orthonormalize_subspace_vectors(exec, subspace_vectors);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR_INITIALIZE_KERNEL);


template <typename ValueType>
void step_1(std::shared_ptr<const DefaultExecutor> exec, const size_type nrhs,
            const size_type k, const matrix::Dense<ValueType>* m,
            const matrix::Dense<ValueType>* f,
            const matrix::Dense<ValueType>* residual,
            const matrix::Dense<ValueType>* g, matrix::Dense<ValueType>* c,
            matrix::Dense<ValueType>* v,
            const array<stopping_status>* stop_status)
{
    solve_lower_triangular(exec, nrhs, m, f, c, stop_status);

    const auto num_rows = v->get_size()[0];
    const auto subspace_dim = m->get_size()[0];

    const auto grid_dim = ceildiv(nrhs * num_rows, default_block_size);
    step_1_kernel<<<grid_dim, default_block_size, 0, exec->get_stream()>>>(
        k, num_rows, subspace_dim, nrhs,
        as_device_type(residual->get_const_values()), residual->get_stride(),
        as_device_type(c->get_const_values()), c->get_stride(),
        as_device_type(g->get_const_values()), g->get_stride(),
        as_device_type(v->get_values()), v->get_stride(),
        stop_status->get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR_STEP_1_KERNEL);


template <typename ValueType>
void step_2(std::shared_ptr<const DefaultExecutor> exec, const size_type nrhs,
            const size_type k, const matrix::Dense<ValueType>* omega,
            const matrix::Dense<ValueType>* preconditioned_vector,
            const matrix::Dense<ValueType>* c, matrix::Dense<ValueType>* u,
            const array<stopping_status>* stop_status)
{
    if (nrhs == 0) {
        return;
    }
    const auto num_rows = preconditioned_vector->get_size()[0];
    const auto subspace_dim = u->get_size()[1] / nrhs;

    const auto grid_dim = ceildiv(nrhs * num_rows, default_block_size);
    step_2_kernel<<<grid_dim, default_block_size, 0, exec->get_stream()>>>(
        k, num_rows, subspace_dim, nrhs,
        as_device_type(omega->get_const_values()),
        as_device_type(preconditioned_vector->get_const_values()),
        preconditioned_vector->get_stride(),
        as_device_type(c->get_const_values()), c->get_stride(),
        as_device_type(u->get_values()), u->get_stride(),
        stop_status->get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR_STEP_2_KERNEL);


template <typename ValueType>
void step_3(std::shared_ptr<const DefaultExecutor> exec, const size_type nrhs,
            const size_type k, const matrix::Dense<ValueType>* p,
            matrix::Dense<ValueType>* g, matrix::Dense<ValueType>* g_k,
            matrix::Dense<ValueType>* u, matrix::Dense<ValueType>* m,
            matrix::Dense<ValueType>* f, matrix::Dense<ValueType>* alpha,
            matrix::Dense<ValueType>* residual, matrix::Dense<ValueType>* x,
            const array<stopping_status>* stop_status)
{
    update_g_and_u(exec, nrhs, k, p, m, alpha, g, g_k, u, stop_status);
    update_m(exec, nrhs, k, p, g_k, m, stop_status);
    update_x_r_and_f(exec, nrhs, k, m, g, u, f, residual, x, stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR_STEP_3_KERNEL);


template <typename ValueType>
void compute_omega(
    std::shared_ptr<const DefaultExecutor> exec, const size_type nrhs,
    const remove_complex<ValueType> kappa, const matrix::Dense<ValueType>* tht,
    const matrix::Dense<remove_complex<ValueType>>* residual_norm,
    matrix::Dense<ValueType>* omega, const array<stopping_status>* stop_status)
{
    const auto grid_dim = ceildiv(nrhs, config::warp_size);
    compute_omega_kernel<<<grid_dim, config::warp_size, 0,
                           exec->get_stream()>>>(
        nrhs, as_device_type(kappa), as_device_type(tht->get_const_values()),
        as_device_type(residual_norm->get_const_values()),
        as_device_type(omega->get_values()), stop_status->get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR_COMPUTE_OMEGA_KERNEL);


}  // namespace idr
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
