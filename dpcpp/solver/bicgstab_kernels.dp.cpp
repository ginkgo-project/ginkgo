/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include "core/solver/bicgstab_kernels.hpp"


#include <CL/sycl.hpp>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>


#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/math.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The BICGSTAB solver namespace.
 *
 * @ingroup bicgstab
 */
namespace bicgstab {


constexpr int default_block_size = 256;


// #include "common/solver/bicgstab_kernels.hpp.inc"
template <typename ValueType>
void initialize_kernel(
    size_type num_rows, size_type num_cols, size_type stride,
    const ValueType *__restrict__ b, ValueType *__restrict__ r,
    ValueType *__restrict__ rr, ValueType *__restrict__ y,
    ValueType *__restrict__ s, ValueType *__restrict__ t,
    ValueType *__restrict__ z, ValueType *__restrict__ v,
    ValueType *__restrict__ p, ValueType *__restrict__ prev_rho,
    ValueType *__restrict__ rho, ValueType *__restrict__ alpha,
    ValueType *__restrict__ beta, ValueType *__restrict__ gamma,
    ValueType *__restrict__ omega, stopping_status *__restrict__ stop_status,
    sycl::nd_item<3> item_ct1)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1);

    if (tidx < num_cols) {
        prev_rho[tidx] = one<ValueType>();
        rho[tidx] = one<ValueType>();
        alpha[tidx] = one<ValueType>();
        beta[tidx] = one<ValueType>();
        gamma[tidx] = one<ValueType>();
        omega[tidx] = one<ValueType>();
        stop_status[tidx].reset();
    }

    if (tidx < num_rows * stride) {
        r[tidx] = b[tidx];
        rr[tidx] = zero<ValueType>();
        y[tidx] = zero<ValueType>();
        s[tidx] = zero<ValueType>();
        t[tidx] = zero<ValueType>();
        z[tidx] = zero<ValueType>();
        v[tidx] = zero<ValueType>();
        p[tidx] = zero<ValueType>();
    }
}

template <typename ValueType>
void initialize_kernel(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                       sycl::queue *stream, size_type num_rows,
                       size_type num_cols, size_type stride, const ValueType *b,
                       ValueType *r, ValueType *rr, ValueType *y, ValueType *s,
                       ValueType *t, ValueType *z, ValueType *v, ValueType *p,
                       ValueType *prev_rho, ValueType *rho, ValueType *alpha,
                       ValueType *beta, ValueType *gamma, ValueType *omega,
                       stopping_status *stop_status)
{
    stream->submit([&](sycl::handler &cgh) {
        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(sycl::nd_range<3>(global_range, local_range),
                         [=](sycl::nd_item<3> item_ct1) {
                             initialize_kernel(num_rows, num_cols, stride, b, r,
                                               rr, y, s, t, z, v, p, prev_rho,
                                               rho, alpha, beta, gamma, omega,
                                               stop_status, item_ct1);
                         });
    });
}


template <typename ValueType>
void step_1_kernel(size_type num_rows, size_type num_cols, size_type stride,
                   const ValueType *__restrict__ r, ValueType *__restrict__ p,
                   const ValueType *__restrict__ v,
                   const ValueType *__restrict__ rho,
                   const ValueType *__restrict__ prev_rho,
                   const ValueType *__restrict__ alpha,
                   const ValueType *__restrict__ omega,
                   const stopping_status *__restrict__ stop_status,
                   sycl::nd_item<3> item_ct1)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1);
    const auto col = tidx % stride;
    if (col >= num_cols || tidx >= num_rows * stride ||
        stop_status[col].has_stopped()) {
        return;
    }
    auto res = r[tidx];
    if (prev_rho[col] * omega[col] != zero<ValueType>()) {
        const auto tmp = (rho[col] / prev_rho[col]) * (alpha[col] / omega[col]);
        res += tmp * (p[tidx] - omega[col] * v[tidx]);
    }
    p[tidx] = res;
}

template <typename ValueType>
void step_1_kernel(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                   sycl::queue *stream, size_type num_rows, size_type num_cols,
                   size_type stride, const ValueType *r, ValueType *p,
                   const ValueType *v, const ValueType *rho,
                   const ValueType *prev_rho, const ValueType *alpha,
                   const ValueType *omega, const stopping_status *stop_status)
{
    stream->submit([&](sycl::handler &cgh) {
        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(sycl::nd_range<3>(global_range, local_range),
                         [=](sycl::nd_item<3> item_ct1) {
                             step_1_kernel(num_rows, num_cols, stride, r, p, v,
                                           rho, prev_rho, alpha, omega,
                                           stop_status, item_ct1);
                         });
    });
}


template <typename ValueType>
void step_2_kernel(size_type num_rows, size_type num_cols, size_type stride,
                   const ValueType *__restrict__ r, ValueType *__restrict__ s,
                   const ValueType *__restrict__ v,
                   const ValueType *__restrict__ rho,
                   ValueType *__restrict__ alpha,
                   const ValueType *__restrict__ beta,
                   const stopping_status *__restrict__ stop_status,
                   sycl::nd_item<3> item_ct1)
{
    const size_type tidx = thread::get_thread_id_flat(item_ct1);
    const size_type col = tidx % stride;
    if (col >= num_cols || tidx >= num_rows * stride ||
        stop_status[col].has_stopped()) {
        return;
    }
    auto t_alpha = zero<ValueType>();
    auto t_s = r[tidx];
    if (beta[col] != zero<ValueType>()) {
        t_alpha = rho[col] / beta[col];
        t_s -= t_alpha * v[tidx];
    }
    alpha[col] = t_alpha;
    s[tidx] = t_s;
}

template <typename ValueType>
void step_2_kernel(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                   sycl::queue *stream, size_type num_rows, size_type num_cols,
                   size_type stride, const ValueType *r, ValueType *s,
                   const ValueType *v, const ValueType *rho, ValueType *alpha,
                   const ValueType *beta, const stopping_status *stop_status)
{
    stream->submit([&](sycl::handler &cgh) {
        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(sycl::nd_range<3>(global_range, local_range),
                         [=](sycl::nd_item<3> item_ct1) {
                             step_2_kernel(num_rows, num_cols, stride, r, s, v,
                                           rho, alpha, beta, stop_status,
                                           item_ct1);
                         });
    });
}


template <typename ValueType>
void step_3_kernel(
    size_type num_rows, size_type num_cols, size_type stride,
    size_type x_stride, ValueType *__restrict__ x, ValueType *__restrict__ r,
    const ValueType *__restrict__ s, const ValueType *__restrict__ t,
    const ValueType *__restrict__ y, const ValueType *__restrict__ z,
    const ValueType *__restrict__ alpha, const ValueType *__restrict__ beta,
    const ValueType *__restrict__ gamma, ValueType *__restrict__ omega,
    const stopping_status *__restrict__ stop_status, sycl::nd_item<3> item_ct1)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1);
    const auto row = tidx / stride;
    const auto col = tidx % stride;
    if (col >= num_cols || tidx >= num_rows * stride ||
        stop_status[col].has_stopped()) {
        return;
    }
    const auto x_pos = row * x_stride + col;
    auto t_omega = zero<ValueType>();
    auto t_x = x[x_pos] + alpha[col] * y[tidx];
    auto t_r = s[tidx];
    if (beta[col] != zero<ValueType>()) {
        t_omega = gamma[col] / beta[col];
        t_x += t_omega * z[tidx];
        t_r -= t_omega * t[tidx];
    }
    omega[col] = t_omega;
    x[x_pos] = t_x;
    r[tidx] = t_r;
}

template <typename ValueType>
void step_3_kernel(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                   sycl::queue *stream, size_type num_rows, size_type num_cols,
                   size_type stride, size_type x_stride, ValueType *x,
                   ValueType *r, const ValueType *s, const ValueType *t,
                   const ValueType *y, const ValueType *z,
                   const ValueType *alpha, const ValueType *beta,
                   const ValueType *gamma, ValueType *omega,
                   const stopping_status *stop_status)
{
    stream->submit([&](sycl::handler &cgh) {
        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(sycl::nd_range<3>(global_range, local_range),
                         [=](sycl::nd_item<3> item_ct1) {
                             step_3_kernel(num_rows, num_cols, stride, x_stride,
                                           x, r, s, t, y, z, alpha, beta, gamma,
                                           omega, stop_status, item_ct1);
                         });
    });
}


template <typename ValueType>
void finalize_kernel(size_type num_rows, size_type num_cols, size_type stride,
                     size_type x_stride, ValueType *__restrict__ x,
                     const ValueType *__restrict__ y,
                     const ValueType *__restrict__ alpha,
                     stopping_status *__restrict__ stop_status,
                     sycl::nd_item<3> item_ct1)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1);
    const auto row = tidx / stride;
    const auto col = tidx % stride;
    if (col >= num_cols || tidx >= num_rows * stride ||
        stop_status[col].is_finalized() || !stop_status[col].has_stopped()) {
        return;
    }
    const auto x_pos = row * x_stride + col;
    x[x_pos] = x[x_pos] + alpha[col] * y[tidx];
    stop_status[col].finalize();
}

template <typename ValueType>
void finalize_kernel(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                     sycl::queue *stream, size_type num_rows,
                     size_type num_cols, size_type stride, size_type x_stride,
                     ValueType *x, const ValueType *y, const ValueType *alpha,
                     stopping_status *stop_status)
{
    stream->submit([&](sycl::handler &cgh) {
        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(sycl::nd_range<3>(global_range, local_range),
                         [=](sycl::nd_item<3> item_ct1) {
                             finalize_kernel(num_rows, num_cols, stride,
                                             x_stride, x, y, alpha, stop_status,
                                             item_ct1);
                         });
    });
}


template <typename ValueType>
void initialize(std::shared_ptr<const DpcppExecutor> exec,
                const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *r,
                matrix::Dense<ValueType> *rr, matrix::Dense<ValueType> *y,
                matrix::Dense<ValueType> *s, matrix::Dense<ValueType> *t,
                matrix::Dense<ValueType> *z, matrix::Dense<ValueType> *v,
                matrix::Dense<ValueType> *p, matrix::Dense<ValueType> *prev_rho,
                matrix::Dense<ValueType> *rho, matrix::Dense<ValueType> *alpha,
                matrix::Dense<ValueType> *beta, matrix::Dense<ValueType> *gamma,
                matrix::Dense<ValueType> *omega,
                Array<stopping_status> *stop_status)
{
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(b->get_size()[0] * b->get_stride(), block_size.x), 1, 1);

    // functioname initialize_kernel
    initialize_kernel(
        grid_size, block_size, 0, exec->get_queue(), b->get_size()[0],
        b->get_size()[1], b->get_stride(), b->get_const_values(),
        r->get_values(), rr->get_values(), y->get_values(), s->get_values(),
        t->get_values(), z->get_values(), v->get_values(), p->get_values(),
        prev_rho->get_values(), rho->get_values(), alpha->get_values(),
        beta->get_values(), gamma->get_values(), omega->get_values(),
        stop_status->get_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_INITIALIZE_KERNEL);


template <typename ValueType>
void step_1(std::shared_ptr<const DpcppExecutor> exec,
            const matrix::Dense<ValueType> *r, matrix::Dense<ValueType> *p,
            const matrix::Dense<ValueType> *v,
            const matrix::Dense<ValueType> *rho,
            const matrix::Dense<ValueType> *prev_rho,
            const matrix::Dense<ValueType> *alpha,
            const matrix::Dense<ValueType> *omega,
            const Array<stopping_status> *stop_status)
{
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(r->get_size()[0] * r->get_stride(), block_size.x), 1, 1);

    // functioname step_1_kernel
    step_1_kernel(grid_size, block_size, 0, exec->get_queue(), r->get_size()[0],
                  r->get_size()[1], r->get_stride(), r->get_const_values(),
                  p->get_values(), v->get_const_values(),
                  rho->get_const_values(), prev_rho->get_const_values(),
                  alpha->get_const_values(), omega->get_const_values(),
                  stop_status->get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_STEP_1_KERNEL);


template <typename ValueType>
void step_2(std::shared_ptr<const DpcppExecutor> exec,
            const matrix::Dense<ValueType> *r, matrix::Dense<ValueType> *s,
            const matrix::Dense<ValueType> *v,
            const matrix::Dense<ValueType> *rho,
            matrix::Dense<ValueType> *alpha,
            const matrix::Dense<ValueType> *beta,
            const Array<stopping_status> *stop_status)
{
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(r->get_size()[0] * r->get_stride(), block_size.x), 1, 1);

    // functioname step_2_kernel
    step_2_kernel(grid_size, block_size, 0, exec->get_queue(), r->get_size()[0],
                  r->get_size()[1], r->get_stride(), r->get_const_values(),
                  s->get_values(), v->get_const_values(),
                  rho->get_const_values(), alpha->get_values(),
                  beta->get_const_values(), stop_status->get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_STEP_2_KERNEL);


template <typename ValueType>
void step_3(
    std::shared_ptr<const DpcppExecutor> exec, matrix::Dense<ValueType> *x,
    matrix::Dense<ValueType> *r, const matrix::Dense<ValueType> *s,
    const matrix::Dense<ValueType> *t, const matrix::Dense<ValueType> *y,
    const matrix::Dense<ValueType> *z, const matrix::Dense<ValueType> *alpha,
    const matrix::Dense<ValueType> *beta, const matrix::Dense<ValueType> *gamma,
    matrix::Dense<ValueType> *omega, const Array<stopping_status> *stop_status)
{
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(r->get_size()[0] * r->get_stride(), block_size.x), 1, 1);

    // functioname step_3_kernel
    step_3_kernel(grid_size, block_size, 0, exec->get_queue(), r->get_size()[0],
                  r->get_size()[1], r->get_stride(), x->get_stride(),
                  x->get_values(), r->get_values(), s->get_const_values(),
                  t->get_const_values(), y->get_const_values(),
                  z->get_const_values(), alpha->get_const_values(),
                  beta->get_const_values(), gamma->get_const_values(),
                  omega->get_values(), stop_status->get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_STEP_3_KERNEL);


template <typename ValueType>
void finalize(std::shared_ptr<const DpcppExecutor> exec,
              matrix::Dense<ValueType> *x, const matrix::Dense<ValueType> *y,
              const matrix::Dense<ValueType> *alpha,
              Array<stopping_status> *stop_status)
{
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(y->get_size()[0] * y->get_stride(), block_size.x), 1, 1);

    // functioname finalize_kernel
    finalize_kernel(grid_size, block_size, 0, exec->get_queue(),
                    y->get_size()[0], y->get_size()[1], y->get_stride(),
                    x->get_stride(), x->get_values(), y->get_const_values(),
                    alpha->get_const_values(), stop_status->get_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_FINALIZE_KERNEL);


}  // namespace bicgstab
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
