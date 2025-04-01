// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/pipe_cg_kernels.hpp"

#include <ginkgo/core/base/math.hpp>

#include "common/unified/base/kernel_launch_solver.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The PIPE_CG solver namespace.
 *
 * @ingroup pipe_cg
 */
namespace pipe_cg {


template <typename ValueType>
void initialize_1(std::shared_ptr<const DefaultExecutor> exec,
                  const matrix::Dense<ValueType>* b,
                  matrix::Dense<ValueType>* r,
                  matrix::Dense<ValueType>* prev_rho,
                  array<stopping_status>* stop_status)
{
    GKO_NOT_IMPLEMENTED;
    // if (b->get_size()) {
    //     run_kernel_solver(
    //         exec,
    //         [] GKO_KERNEL(auto row, auto col, auto b, auto r, auto z, auto p,
    //                       auto q, auto prev_rho, auto rho, auto stop) {
    //             if (row == 0) {
    //                 rho[col] = zero(rho[col]);
    //                 prev_rho[col] = one(prev_rho[col]);
    //                 stop[col].reset();
    //             }
    //             r(row, col) = b(row, col);
    //             z(row, col) = p(row, col) = q(row, col) = zero(z(row, col));
    //         },
    //         b->get_size(), b->get_stride(), b, default_stride(r),
    //         default_stride(z), default_stride(p), default_stride(q),
    //         row_vector(prev_rho), row_vector(rho), *stop_status);
    // } else {
    //     // what is this case?
    //     run_kernel(
    //         exec,
    //         [] GKO_KERNEL(auto col, auto prev_rho, auto rho, auto stop) {
    //             rho[col] = zero(rho[col]);
    //             prev_rho[col] = one(prev_rho[col]);
    //             stop[col].reset();
    //         },
    //         b->get_size()[1], row_vector(prev_rho), row_vector(rho),
    //         *stop_status);
    // }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_PIPE_CG_INITIALIZE_1_KERNEL);

template <typename ValueType>
void initialize_2(std::shared_ptr<const DefaultExecutor> exec,
                  matrix::Dense<ValueType>* p, matrix::Dense<ValueType>* q,
                  matrix::Dense<ValueType>* f, matrix::Dense<ValueType>* g,
                  matrix::Dense<ValueType>* beta,
                  const matrix::Dense<ValueType>* z,
                  const matrix::Dense<ValueType>* w,
                  const matrix::Dense<ValueType>* m,
                  const matrix::Dense<ValueType>* n,
                  const matrix::Dense<ValueType>* delta)
{
    GKO_NOT_IMPLEMENTED;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_PIPE_CG_INITIALIZE_2_KERNEL);


template <typename ValueType>
void step_1(std::shared_ptr<const DefaultExecutor> exec,
            matrix::Dense<ValueType>* x, matrix::Dense<ValueType>* r,
            matrix::Dense<ValueType>* z, matrix::Dense<ValueType>* w,
            const matrix::Dense<ValueType>* p,
            const matrix::Dense<ValueType>* q,
            const matrix::Dense<ValueType>* f,
            const matrix::Dense<ValueType>* g,
            const matrix::Dense<ValueType>* rho,
            const matrix::Dense<ValueType>* beta,
            const array<stopping_status>* stop_status)
{
    GKO_NOT_IMPLEMENTED;
    // run_kernel_solver(
    //     exec,
    //     [] GKO_KERNEL(auto row, auto col, auto p, auto z, auto rho,
    //                   auto prev_rho, auto stop) {
    //         if (!stop[col].has_stopped()) {
    //             auto tmp = safe_divide(rho[col], prev_rho[col]);
    //             p(row, col) = z(row, col) + tmp * p(row, col);
    //         }
    //     },
    //     p->get_size(), p->get_stride(), default_stride(p), default_stride(z),
    //     row_vector(rho), row_vector(prev_rho), *stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_PIPE_CG_STEP_1_KERNEL);


template <typename ValueType>
void step_2(std::shared_ptr<const DefaultExecutor> exec,
            matrix::Dense<ValueType>* beta, matrix::Dense<ValueType>* p,
            matrix::Dense<ValueType>* q, matrix::Dense<ValueType>* f,
            matrix::Dense<ValueType>* g, const matrix::Dense<ValueType>* z,
            const matrix::Dense<ValueType>* w,
            const matrix::Dense<ValueType>* m,
            const matrix::Dense<ValueType>* n,
            const matrix::Dense<ValueType>* prev_rho,
            const matrix::Dense<ValueType>* rho,
            const matrix::Dense<ValueType>* delta,
            const array<stopping_status>* stop_status)
{
    GKO_NOT_IMPLEMENTED;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_PIPE_CG_STEP_2_KERNEL);


}  // namespace pipe_cg
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
