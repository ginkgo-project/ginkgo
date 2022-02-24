/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#include "core/matrix/dense_kernels.hpp"


#include <ginkgo/core/base/device_matrix_data.hpp>
#include <ginkgo/core/base/math.hpp>


#include "common/unified/base/kernel_launch.hpp"
#include "common/unified/base/kernel_launch_reduction.hpp"
#include "core/base/mixed_precision_types.hpp"
#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The Dense matrix format namespace.
 *
 * @ingroup dense
 */
namespace dense {


template <typename ValueType, typename ScalarType>
void scale(std::shared_ptr<const DefaultExecutor> exec,
           const matrix::Dense<ScalarType>* alpha, matrix::Dense<ValueType>* x)
{
    if (alpha->get_size()[1] > 1) {
        run_kernel(
            exec,
            [] GKO_KERNEL(auto row, auto col, auto alpha, auto x) {
                x(row, col) *= alpha[col];
            },
            x->get_size(), alpha->get_const_values(), x);
    } else {
        run_kernel(
            exec,
            [] GKO_KERNEL(auto row, auto col, auto alpha, auto x) {
                x(row, col) *= alpha[0];
            },
            x->get_size(), alpha->get_const_values(), x);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_SCALAR_TYPE(GKO_DECLARE_DENSE_SCALE_KERNEL);


template <typename ValueType, typename ScalarType>
void inv_scale(std::shared_ptr<const DefaultExecutor> exec,
               const matrix::Dense<ScalarType>* alpha,
               matrix::Dense<ValueType>* x)
{
    if (alpha->get_size()[1] > 1) {
        run_kernel(
            exec,
            [] GKO_KERNEL(auto row, auto col, auto alpha, auto x) {
                x(row, col) /= alpha[col];
            },
            x->get_size(), alpha->get_const_values(), x);
    } else {
        run_kernel(
            exec,
            [] GKO_KERNEL(auto row, auto col, auto alpha, auto x) {
                x(row, col) /= alpha[0];
            },
            x->get_size(), alpha->get_const_values(), x);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_SCALAR_TYPE(
    GKO_DECLARE_DENSE_INV_SCALE_KERNEL);


template <typename ValueType, typename ScalarType>
void add_scaled(std::shared_ptr<const DefaultExecutor> exec,
                const matrix::Dense<ScalarType>* alpha,
                const matrix::Dense<ValueType>* x, matrix::Dense<ValueType>* y)
{
    if (alpha->get_size()[1] > 1) {
        run_kernel(
            exec,
            [] GKO_KERNEL(auto row, auto col, auto alpha, auto x, auto y) {
                y(row, col) += alpha[col] * x(row, col);
            },
            x->get_size(), alpha->get_const_values(), x, y);
    } else {
        run_kernel(
            exec,
            [] GKO_KERNEL(auto row, auto col, auto alpha, auto x, auto y) {
                y(row, col) += alpha[0] * x(row, col);
            },
            x->get_size(), alpha->get_const_values(), x, y);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_SCALAR_TYPE(
    GKO_DECLARE_DENSE_ADD_SCALED_KERNEL);


template <typename ValueType, typename ScalarType>
void sub_scaled(std::shared_ptr<const DefaultExecutor> exec,
                const matrix::Dense<ScalarType>* alpha,
                const matrix::Dense<ValueType>* x, matrix::Dense<ValueType>* y)
{
    if (alpha->get_size()[1] > 1) {
        run_kernel(
            exec,
            [] GKO_KERNEL(auto row, auto col, auto alpha, auto x, auto y) {
                y(row, col) -= alpha[col] * x(row, col);
            },
            x->get_size(), alpha->get_const_values(), x, y);
    } else {
        run_kernel(
            exec,
            [] GKO_KERNEL(auto row, auto col, auto alpha, auto x, auto y) {
                y(row, col) -= alpha[0] * x(row, col);
            },
            x->get_size(), alpha->get_const_values(), x, y);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_SCALAR_TYPE(
    GKO_DECLARE_DENSE_SUB_SCALED_KERNEL);


template <typename ValueType>
void add_scaled_diag(std::shared_ptr<const DefaultExecutor> exec,
                     const matrix::Dense<ValueType>* alpha,
                     const matrix::Diagonal<ValueType>* x,
                     matrix::Dense<ValueType>* y)
{
    const auto diag_values = x->get_const_values();
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto alpha, auto diag, auto y) {
            y(i, i) += alpha[0] * diag[i];
        },
        x->get_size()[0], alpha->get_const_values(), x->get_const_values(), y);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_ADD_SCALED_DIAG_KERNEL);


template <typename ValueType>
void sub_scaled_diag(std::shared_ptr<const DefaultExecutor> exec,
                     const matrix::Dense<ValueType>* alpha,
                     const matrix::Diagonal<ValueType>* x,
                     matrix::Dense<ValueType>* y)
{
    const auto diag_values = x->get_const_values();
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto alpha, auto diag, auto y) {
            y(i, i) -= alpha[0] * diag[i];
        },
        x->get_size()[0], alpha->get_const_values(), x->get_const_values(), y);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_SUB_SCALED_DIAG_KERNEL);


}  // namespace dense
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
