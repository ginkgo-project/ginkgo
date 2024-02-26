// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/idr_kernels.hpp"


#include <ctime>
#include <random>


#include <CL/sycl.hpp>
#include <oneapi/dpl/random>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>


#include "core/components/fill_array_kernels.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/onemkl_bindings.hpp"
#include "dpcpp/components/atomic.dp.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"
#include "dpcpp/components/reduction.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The IDR solver namespace.
 *
 * @ingroup idr
 */
namespace idr {


constexpr int default_block_size = 256;
constexpr int default_dot_dim = 16;
constexpr int default_dot_size = default_dot_dim * default_dot_dim;


template <typename ValueType>
void initialize_m_kernel(size_type subspace_dim, size_type nrhs,
                         ValueType* __restrict__ m_values, size_type m_stride,
                         stopping_status* __restrict__ stop_status,
                         sycl::nd_item<3> item_ct1)
{
    const auto global_id = thread::get_thread_id_flat(item_ct1);
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

template <typename ValueType>
void initialize_m_kernel(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                         sycl::queue* stream, size_type subspace_dim,
                         size_type nrhs, ValueType* m_values,
                         size_type m_stride, stopping_status* stop_status)
{
    if (nrhs == 0) {
        return;
    }
    stream->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                initialize_m_kernel(subspace_dim, nrhs, m_values, m_stride,
                                    stop_status, item_ct1);
            });
    });
}


template <size_type block_size, typename ValueType>
void orthonormalize_subspace_vectors_kernel(
    size_type num_rows, size_type num_cols, ValueType* __restrict__ values,
    size_type stride, sycl::nd_item<3> item_ct1,
    uninitialized_array<ValueType, block_size>& reduction_helper_array)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1);

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
            item_ct1.barrier(sycl::access::fence_space::local_space);
            reduction_helper[tidx] = dot;
            ::gko::kernels::dpcpp::reduce(
                group::this_thread_block(item_ct1), reduction_helper,
                [](const ValueType& a, const ValueType& b) { return a + b; });
            item_ct1.barrier(sycl::access::fence_space::local_space);

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
        item_ct1.barrier(sycl::access::fence_space::local_space);
        reduction_helper_real[tidx] = norm;
        ::gko::kernels::dpcpp::reduce(
            group::this_thread_block(item_ct1), reduction_helper_real,
            [](const remove_complex<ValueType>& a,
               const remove_complex<ValueType>& b) { return a + b; });
        item_ct1.barrier(sycl::access::fence_space::local_space);

        norm = std::sqrt(reduction_helper_real[0]);
        for (size_type j = tidx; j < num_cols; j += block_size) {
            values[row * stride + j] /= norm;
        }
    }
}

template <size_type block_size, typename ValueType>
void orthonormalize_subspace_vectors_kernel(
    dim3 grid, dim3 block, size_t dynamic_shared_memory, sycl::queue* stream,
    size_type num_rows, size_type num_cols, ValueType* values, size_type stride)
{
    stream->submit([&](sycl::handler& cgh) {
        sycl::accessor<uninitialized_array<ValueType, block_size>, 0,
                       sycl::access_mode::read_write,
                       sycl::access::target::local>
            reduction_helper_array_acc_ct1(cgh);

        cgh.parallel_for(
            sycl_nd_range(grid, block),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(config::warp_size)]] {
                    orthonormalize_subspace_vectors_kernel<block_size>(
                        num_rows, num_cols, values, stride, item_ct1,
                        *reduction_helper_array_acc_ct1.get_pointer());
                });
    });
}


template <typename ValueType>
void solve_lower_triangular_kernel(
    size_type subspace_dim, size_type nrhs,
    const ValueType* __restrict__ m_values, size_type m_stride,
    const ValueType* __restrict__ f_values, size_type f_stride,
    ValueType* __restrict__ c_values, size_type c_stride,
    const stopping_status* __restrict__ stop_status, sycl::nd_item<3> item_ct1)
{
    const auto global_id = thread::get_thread_id_flat(item_ct1);

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
void solve_lower_triangular_kernel(
    dim3 grid, dim3 block, size_t dynamic_shared_memory, sycl::queue* stream,
    size_type subspace_dim, size_type nrhs, const ValueType* m_values,
    size_type m_stride, const ValueType* f_values, size_type f_stride,
    ValueType* c_values, size_type c_stride, const stopping_status* stop_status)
{
    stream->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                solve_lower_triangular_kernel(
                    subspace_dim, nrhs, m_values, m_stride, f_values, f_stride,
                    c_values, c_stride, stop_status, item_ct1);
            });
    });
}


template <typename ValueType>
void step_1_kernel(size_type k, size_type num_rows, size_type subspace_dim,
                   size_type nrhs,
                   const ValueType* __restrict__ residual_values,
                   size_type residual_stride,
                   const ValueType* __restrict__ c_values, size_type c_stride,
                   const ValueType* __restrict__ g_values, size_type g_stride,
                   ValueType* __restrict__ v_values, size_type v_stride,
                   const stopping_status* __restrict__ stop_status,
                   sycl::nd_item<3> item_ct1)
{
    const auto global_id = thread::get_thread_id_flat(item_ct1);
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
void step_1_kernel(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                   sycl::queue* stream, size_type k, size_type num_rows,
                   size_type subspace_dim, size_type nrhs,
                   const ValueType* residual_values, size_type residual_stride,
                   const ValueType* c_values, size_type c_stride,
                   const ValueType* g_values, size_type g_stride,
                   ValueType* v_values, size_type v_stride,
                   const stopping_status* stop_status)
{
    if (nrhs == 0) {
        return;
    }
    stream->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                step_1_kernel(k, num_rows, subspace_dim, nrhs, residual_values,
                              residual_stride, c_values, c_stride, g_values,
                              g_stride, v_values, v_stride, stop_status,
                              item_ct1);
            });
    });
}


template <typename ValueType>
void step_2_kernel(size_type k, size_type num_rows, size_type subspace_dim,
                   size_type nrhs, const ValueType* __restrict__ omega_values,
                   const ValueType* __restrict__ v_values, size_type v_stride,
                   const ValueType* __restrict__ c_values, size_type c_stride,
                   ValueType* __restrict__ u_values, size_type u_stride,
                   const stopping_status* __restrict__ stop_status,
                   sycl::nd_item<3> item_ct1)
{
    const auto global_id = thread::get_thread_id_flat(item_ct1);
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
void step_2_kernel(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                   sycl::queue* stream, size_type k, size_type num_rows,
                   size_type subspace_dim, size_type nrhs,
                   const ValueType* omega_values, const ValueType* v_values,
                   size_type v_stride, const ValueType* c_values,
                   size_type c_stride, ValueType* u_values, size_type u_stride,
                   const stopping_status* stop_status)
{
    if (nrhs == 0) {
        return;
    }
    stream->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                step_2_kernel(k, num_rows, subspace_dim, nrhs, omega_values,
                              v_values, v_stride, c_values, c_stride, u_values,
                              u_stride, stop_status, item_ct1);
            });
    });
}


template <typename ValueType>
void multidot_kernel(
    size_type num_rows, size_type nrhs, const ValueType* __restrict__ p_i,
    const ValueType* __restrict__ g_k, size_type g_k_stride,
    ValueType* __restrict__ alpha,
    const stopping_status* __restrict__ stop_status, sycl::nd_item<3> item_ct1,
    uninitialized_array<ValueType, default_dot_dim*(default_dot_dim + 1)>&
        reduction_helper_array)
{
    const auto tidx = item_ct1.get_local_id(2);
    const auto tidy = item_ct1.get_local_id(1);
    const auto rhs = item_ct1.get_group(2) * default_dot_dim + tidx;
    const auto num = ceildiv(num_rows, item_ct1.get_group_range(1));
    const auto start_row = item_ct1.get_group(1) * num;
    const auto end_row = ((item_ct1.get_group(1) + 1) * num > num_rows)
                             ? num_rows
                             : (item_ct1.get_group(1) + 1) * num;
    // Used that way to get around dynamic initialization warning and
    // template error when using `reduction_helper_array` directly in `reduce`
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
    item_ct1.barrier(sycl::access::fence_space::local_space);
    local_res = reduction_helper[tidy * (default_dot_dim + 1) + tidx];
    const auto tile_block = group::tiled_partition<default_dot_dim>(
        group::this_thread_block(item_ct1));
    const auto sum = ::gko::kernels::dpcpp::reduce(
        tile_block, local_res,
        [](const ValueType& a, const ValueType& b) { return a + b; });
    const auto new_rhs = item_ct1.get_group(2) * default_dot_dim + tidy;
    if (tidx == 0 && new_rhs < nrhs && !stop_status[new_rhs].has_stopped()) {
        atomic_add(alpha + new_rhs, sum);
    }
}

template <typename ValueType>
void multidot_kernel(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                     sycl::queue* stream, size_type num_rows, size_type nrhs,
                     const ValueType* p_i, const ValueType* g_k,
                     size_type g_k_stride, ValueType* alpha,
                     const stopping_status* stop_status)
{
    stream->submit([&](sycl::handler& cgh) {
        sycl::accessor<uninitialized_array<ValueType, default_dot_dim*(
                                                          default_dot_dim + 1)>,
                       0, sycl::access_mode::read_write,
                       sycl::access::target::local>
            reduction_helper_array_acc_ct1(cgh);

        cgh.parallel_for(
            sycl_nd_range(grid, block),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(default_dot_dim)]] {
                    multidot_kernel(
                        num_rows, nrhs, p_i, g_k, g_k_stride, alpha,
                        stop_status, item_ct1,
                        *reduction_helper_array_acc_ct1.get_pointer());
                });
    });
}


template <size_type block_size, typename ValueType>
void update_g_k_and_u_kernel(
    size_type k, size_type i, size_type size, size_type nrhs,
    const ValueType* __restrict__ alpha, const ValueType* __restrict__ m_values,
    size_type m_stride, const ValueType* __restrict__ g_values,
    size_type g_stride, ValueType* __restrict__ g_k_values,
    size_type g_k_stride, ValueType* __restrict__ u_values, size_type u_stride,
    const stopping_status* __restrict__ stop_status, sycl::nd_item<3> item_ct1)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1);
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
void update_g_k_and_u_kernel(dim3 grid, dim3 block,
                             size_t dynamic_shared_memory, sycl::queue* stream,
                             size_type k, size_type i, size_type size,
                             size_type nrhs, const ValueType* alpha,
                             const ValueType* m_values, size_type m_stride,
                             const ValueType* g_values, size_type g_stride,
                             ValueType* g_k_values, size_type g_k_stride,
                             ValueType* u_values, size_type u_stride,
                             const stopping_status* stop_status)
{
    if (g_k_stride == 0) {
        return;
    }
    stream->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl_nd_range(grid, block),
                         [=](sycl::nd_item<3> item_ct1) {
                             update_g_k_and_u_kernel<block_size>(
                                 k, i, size, nrhs, alpha, m_values, m_stride,
                                 g_values, g_stride, g_k_values, g_k_stride,
                                 u_values, u_stride, stop_status, item_ct1);
                         });
    });
}


template <size_type block_size, typename ValueType>
void update_g_kernel(size_type k, size_type size, size_type nrhs,
                     const ValueType* __restrict__ g_k_values,
                     size_type g_k_stride, ValueType* __restrict__ g_values,
                     size_type g_stride,
                     const stopping_status* __restrict__ stop_status,
                     sycl::nd_item<3> item_ct1)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1);
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

template <size_type block_size, typename ValueType>
void update_g_kernel(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                     sycl::queue* stream, size_type k, size_type size,
                     size_type nrhs, const ValueType* g_k_values,
                     size_type g_k_stride, ValueType* g_values,
                     size_type g_stride, const stopping_status* stop_status)
{
    if (g_k_stride == 0) {
        return;
    }
    stream->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                update_g_kernel<block_size>(k, size, nrhs, g_k_values,
                                            g_k_stride, g_values, g_stride,
                                            stop_status, item_ct1);
            });
    });
}


template <typename ValueType>
void update_x_r_and_f_kernel(
    size_type k, size_type size, size_type subspace_dim, size_type nrhs,
    const ValueType* __restrict__ m_values, size_type m_stride,
    const ValueType* __restrict__ g_values, size_type g_stride,
    const ValueType* __restrict__ u_values, size_type u_stride,
    ValueType* __restrict__ f_values, size_type f_stride,
    ValueType* __restrict__ r_values, size_type r_stride,
    ValueType* __restrict__ x_values, size_type x_stride,
    const stopping_status* __restrict__ stop_status, sycl::nd_item<3> item_ct1)
{
    const auto global_id = thread::get_thread_id_flat(item_ct1);
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
void update_x_r_and_f_kernel(
    dim3 grid, dim3 block, size_t dynamic_shared_memory, sycl::queue* stream,
    size_type k, size_type size, size_type subspace_dim, size_type nrhs,
    const ValueType* m_values, size_type m_stride, const ValueType* g_values,
    size_type g_stride, const ValueType* u_values, size_type u_stride,
    ValueType* f_values, size_type f_stride, ValueType* r_values,
    size_type r_stride, ValueType* x_values, size_type x_stride,
    const stopping_status* stop_status)
{
    stream->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                update_x_r_and_f_kernel(
                    k, size, subspace_dim, nrhs, m_values, m_stride, g_values,
                    g_stride, u_values, u_stride, f_values, f_stride, r_values,
                    r_stride, x_values, x_stride, stop_status, item_ct1);
            });
    });
}


template <typename ValueType>
void compute_omega_kernel(
    size_type nrhs, const remove_complex<ValueType> kappa,
    const ValueType* __restrict__ tht,
    const remove_complex<ValueType>* __restrict__ residual_norm,
    ValueType* __restrict__ omega,
    const stopping_status* __restrict__ stop_status, sycl::nd_item<3> item_ct1)
{
    const auto global_id = thread::get_thread_id_flat(item_ct1);

    if (global_id >= nrhs) {
        return;
    }

    if (!stop_status[global_id].has_stopped()) {
        auto thr = omega[global_id];
        omega[global_id] /= tht[global_id];
        auto absrho = std::abs(
            thr / (std::sqrt(real(tht[global_id])) * residual_norm[global_id]));

        if (absrho < kappa) {
            omega[global_id] *= kappa / absrho;
        }
    }
}

template <typename ValueType>
void compute_omega_kernel(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                          sycl::queue* stream, size_type nrhs,
                          const remove_complex<ValueType> kappa,
                          const ValueType* tht,
                          const remove_complex<ValueType>* residual_norm,
                          ValueType* omega, const stopping_status* stop_status)
{
    stream->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                compute_omega_kernel(nrhs, kappa, tht, residual_norm, omega,
                                     stop_status, item_ct1);
            });
    });
}


namespace {


template <typename ValueType>
void initialize_m(std::shared_ptr<const DpcppExecutor> exec,
                  const size_type nrhs, matrix::Dense<ValueType>* m,
                  array<stopping_status>* stop_status)
{
    const auto subspace_dim = m->get_size()[0];
    const auto m_stride = m->get_stride();

    const auto grid_dim = ceildiv(m_stride * subspace_dim, default_block_size);
    initialize_m_kernel(grid_dim, default_block_size, 0, exec->get_queue(),
                        subspace_dim, nrhs, m->get_values(), m_stride,
                        stop_status->get_data());
}


template <typename ValueType>
void initialize_subspace_vectors(std::shared_ptr<const DpcppExecutor> exec,
                                 matrix::Dense<ValueType>* subspace_vectors,
                                 bool deterministic)
{
    if (!deterministic) {
        auto seed = std::random_device{}();
        auto work = reinterpret_cast<remove_complex<ValueType>*>(
            subspace_vectors->get_values());
        auto n =
            subspace_vectors->get_size()[0] * subspace_vectors->get_stride();
        n = is_complex<ValueType>() ? 2 * n : n;
        exec->get_queue()->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(n), [=](sycl::item<1> idx) {
                std::uint64_t offset = idx.get_linear_id();
                oneapi::dpl::minstd_rand engine(seed, offset);
                oneapi::dpl::normal_distribution<remove_complex<ValueType>>
                    distr(0, 1);
                auto res = distr(engine);

                work[idx] = res;
            });
        });
    }
}


template <typename ValueType>
void orthonormalize_subspace_vectors(std::shared_ptr<const DpcppExecutor> exec,
                                     matrix::Dense<ValueType>* subspace_vectors)
{
    orthonormalize_subspace_vectors_kernel<default_block_size>(
        1, default_block_size, 0, exec->get_queue(),
        subspace_vectors->get_size()[0], subspace_vectors->get_size()[1],
        subspace_vectors->get_values(), subspace_vectors->get_stride());
}


template <typename ValueType>
void solve_lower_triangular(std::shared_ptr<const DpcppExecutor> exec,
                            const size_type nrhs,
                            const matrix::Dense<ValueType>* m,
                            const matrix::Dense<ValueType>* f,
                            matrix::Dense<ValueType>* c,
                            const array<stopping_status>* stop_status)
{
    const auto subspace_dim = m->get_size()[0];

    const auto grid_dim = ceildiv(nrhs, default_block_size);
    solve_lower_triangular_kernel(
        grid_dim, default_block_size, 0, exec->get_queue(), subspace_dim, nrhs,
        m->get_const_values(), m->get_stride(), f->get_const_values(),
        f->get_stride(), c->get_values(), c->get_stride(),
        stop_status->get_const_data());
}


template <typename ValueType>
void update_g_and_u(std::shared_ptr<const DpcppExecutor> exec,
                    const size_type nrhs, const size_type k,
                    const matrix::Dense<ValueType>* p,
                    const matrix::Dense<ValueType>* m,
                    matrix::Dense<ValueType>* alpha,
                    matrix::Dense<ValueType>* g, matrix::Dense<ValueType>* g_k,
                    matrix::Dense<ValueType>* u,
                    const array<stopping_status>* stop_status)
{
    const auto size = g->get_size()[0];
    const auto p_stride = p->get_stride();

    const dim3 grid_dim(ceildiv(nrhs, default_dot_dim),
                        exec->get_num_computing_units() * 2);
    const dim3 block_dim(default_dot_dim, default_dot_dim);

    for (size_type i = 0; i < k; i++) {
        const auto p_i = p->get_const_values() + i * p_stride;
        if (nrhs > 1 || is_complex<ValueType>()) {
            components::fill_array(exec, alpha->get_values(), nrhs,
                                   zero<ValueType>());
            multidot_kernel(grid_dim, block_dim, 0, exec->get_queue(), size,
                            nrhs, p_i, g_k->get_values(), g_k->get_stride(),
                            alpha->get_values(), stop_status->get_const_data());
        } else {
            onemkl::dot(*exec->get_queue(), size, p_i, 1, g_k->get_values(),
                        g_k->get_stride(), alpha->get_values());
        }
        update_g_k_and_u_kernel<default_block_size>(
            ceildiv(size * g_k->get_stride(), default_block_size),
            default_block_size, 0, exec->get_queue(), k, i, size, nrhs,
            alpha->get_const_values(), m->get_const_values(), m->get_stride(),
            g->get_const_values(), g->get_stride(), g_k->get_values(),
            g_k->get_stride(), u->get_values(), u->get_stride(),
            stop_status->get_const_data());
    }
    update_g_kernel<default_block_size>(
        ceildiv(size * g_k->get_stride(), default_block_size),
        default_block_size, 0, exec->get_queue(), k, size, nrhs,
        g_k->get_const_values(), g_k->get_stride(), g->get_values(),
        g->get_stride(), stop_status->get_const_data());
}


template <typename ValueType>
void update_m(std::shared_ptr<const DpcppExecutor> exec, const size_type nrhs,
              const size_type k, const matrix::Dense<ValueType>* p,
              const matrix::Dense<ValueType>* g_k, matrix::Dense<ValueType>* m,
              const array<stopping_status>* stop_status)
{
    const auto size = g_k->get_size()[0];
    const auto subspace_dim = m->get_size()[0];
    const auto p_stride = p->get_stride();
    const auto m_stride = m->get_stride();

    const dim3 grid_dim(ceildiv(nrhs, default_dot_dim),
                        exec->get_num_computing_units() * 2);
    const dim3 block_dim(default_dot_dim, default_dot_dim);

    for (size_type i = k; i < subspace_dim; i++) {
        const auto p_i = p->get_const_values() + i * p_stride;
        auto m_i = m->get_values() + i * m_stride + k * nrhs;
        if (nrhs > 1 || is_complex<ValueType>()) {
            components::fill_array(exec, m_i, nrhs, zero<ValueType>());
            multidot_kernel(grid_dim, block_dim, 0, exec->get_queue(), size,
                            nrhs, p_i, g_k->get_const_values(),
                            g_k->get_stride(), m_i,
                            stop_status->get_const_data());
        } else {
            onemkl::dot(*exec->get_queue(), size, p_i, 1,
                        g_k->get_const_values(), g_k->get_stride(), m_i);
        }
    }
}


template <typename ValueType>
void update_x_r_and_f(std::shared_ptr<const DpcppExecutor> exec,
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
    update_x_r_and_f_kernel(grid_dim, default_block_size, 0, exec->get_queue(),
                            k, size, subspace_dim, nrhs, m->get_const_values(),
                            m->get_stride(), g->get_const_values(),
                            g->get_stride(), u->get_const_values(),
                            u->get_stride(), f->get_values(), f->get_stride(),
                            r->get_values(), r->get_stride(), x->get_values(),
                            x->get_stride(), stop_status->get_const_data());
    components::fill_array(exec, f->get_values() + k * f->get_stride(), nrhs,
                           zero<ValueType>());
}


}  // namespace


template <typename ValueType>
void initialize(std::shared_ptr<const DpcppExecutor> exec, const size_type nrhs,
                matrix::Dense<ValueType>* m,
                matrix::Dense<ValueType>* subspace_vectors, bool deterministic,
                array<stopping_status>* stop_status)
{
    initialize_m(exec, nrhs, m, stop_status);
    initialize_subspace_vectors(exec, subspace_vectors, deterministic);
    orthonormalize_subspace_vectors(exec, subspace_vectors);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR_INITIALIZE_KERNEL);


template <typename ValueType>
void step_1(std::shared_ptr<const DpcppExecutor> exec, const size_type nrhs,
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
    step_1_kernel(grid_dim, default_block_size, 0, exec->get_queue(), k,
                  num_rows, subspace_dim, nrhs, residual->get_const_values(),
                  residual->get_stride(), c->get_const_values(),
                  c->get_stride(), g->get_const_values(), g->get_stride(),
                  v->get_values(), v->get_stride(),
                  stop_status->get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR_STEP_1_KERNEL);


template <typename ValueType>
void step_2(std::shared_ptr<const DpcppExecutor> exec, const size_type nrhs,
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
    step_2_kernel(grid_dim, default_block_size, 0, exec->get_queue(), k,
                  num_rows, subspace_dim, nrhs, omega->get_const_values(),
                  preconditioned_vector->get_const_values(),
                  preconditioned_vector->get_stride(), c->get_const_values(),
                  c->get_stride(), u->get_values(), u->get_stride(),
                  stop_status->get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR_STEP_2_KERNEL);


template <typename ValueType>
void step_3(std::shared_ptr<const DpcppExecutor> exec, const size_type nrhs,
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
    std::shared_ptr<const DpcppExecutor> exec, const size_type nrhs,
    const remove_complex<ValueType> kappa, const matrix::Dense<ValueType>* tht,
    const matrix::Dense<remove_complex<ValueType>>* residual_norm,
    matrix::Dense<ValueType>* omega, const array<stopping_status>* stop_status)
{
    const auto grid_dim = ceildiv(nrhs, config::warp_size);
    compute_omega_kernel(grid_dim, config::warp_size, 0, exec->get_queue(),
                         nrhs, kappa, tht->get_const_values(),
                         residual_norm->get_const_values(), omega->get_values(),
                         stop_status->get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR_COMPUTE_OMEGA_KERNEL);


}  // namespace idr
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
