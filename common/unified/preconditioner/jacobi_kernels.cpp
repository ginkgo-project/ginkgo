// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/preconditioner/jacobi_kernels.hpp"


#include <ginkgo/core/base/math.hpp>


#include "common/unified/base/kernel_launch.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The Jacobi preconditioner namespace.
 *
 * @ingroup jacobi
 */
namespace jacobi {


template <typename ValueType>
void scalar_conj(std::shared_ptr<const DefaultExecutor> exec,
                 const array<ValueType>& diag, array<ValueType>& conj_diag)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto elem, auto diag, auto conj_diag) {
            conj_diag[elem] = conj(diag[elem]);
        },
        diag.get_size(), diag, conj_diag);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_JACOBI_SCALAR_CONJ_KERNEL);


template <typename ValueType>
void invert_diagonal(std::shared_ptr<const DefaultExecutor> exec,
                     const array<ValueType>& diag, array<ValueType>& inv_diag)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto elem, auto diag, auto inv_diag) {
            inv_diag[elem] = safe_divide(one(diag[elem]), diag[elem]);
        },
        diag.get_size(), diag, inv_diag);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_JACOBI_INVERT_DIAGONAL_KERNEL);


template <typename ValueType>
void scalar_apply(std::shared_ptr<const DefaultExecutor> exec,
                  const array<ValueType>& diag,
                  const matrix::Dense<ValueType>* alpha,
                  const matrix::Dense<ValueType>* b,
                  const matrix::Dense<ValueType>* beta,
                  matrix::Dense<ValueType>* x)
{
    if (alpha->get_size()[1] > 1) {
        run_kernel(
            exec,
            [] GKO_KERNEL(auto row, auto col, auto diag, auto alpha, auto b,
                          auto beta, auto x) {
                x(row, col) = beta[col] * x(row, col) +
                              alpha[col] * b(row, col) * diag[row];
            },
            x->get_size(), diag, alpha->get_const_values(), b,
            beta->get_const_values(), x);
    } else {
        run_kernel(
            exec,
            [] GKO_KERNEL(auto row, auto col, auto diag, auto alpha, auto b,
                          auto beta, auto x) {
                x(row, col) =
                    beta[0] * x(row, col) + alpha[0] * b(row, col) * diag[row];
            },
            x->get_size(), diag, alpha->get_const_values(), b,
            beta->get_const_values(), x);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_JACOBI_SCALAR_APPLY_KERNEL);


template <typename ValueType>
void simple_scalar_apply(std::shared_ptr<const DefaultExecutor> exec,
                         const array<ValueType>& diag,
                         const matrix::Dense<ValueType>* b,
                         matrix::Dense<ValueType>* x)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto diag, auto b, auto x) {
            x(row, col) = b(row, col) * diag[row];
        },
        x->get_size(), diag, b, x);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_JACOBI_SIMPLE_SCALAR_APPLY_KERNEL);


template <typename ValueType>
void scalar_convert_to_dense(std::shared_ptr<const DefaultExecutor> exec,
                             const array<ValueType>& blocks,
                             matrix::Dense<ValueType>* result)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto diag, auto result) {
            result(row, col) = zero(diag[row]);
            if (row == col) {
                result(row, col) = diag[row];
            }
        },
        result->get_size(), blocks, result);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_JACOBI_SCALAR_CONVERT_TO_DENSE_KERNEL);


}  // namespace jacobi
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
