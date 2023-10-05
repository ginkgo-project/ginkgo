/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#include "core/matrix/batch_dense_kernels.hpp"


#include <algorithm>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>


#include "core/base/batch_struct.hpp"
#include "core/matrix/batch_struct.hpp"
#include "reference/base/batch_struct.hpp"
#include "reference/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The Dense matrix format namespace.
 * @ref Dense
 * @ingroup batch_dense
 */
namespace batch_dense {


#include "reference/matrix/batch_dense_kernels.hpp.inc"


template <typename ValueType>
void simple_apply(std::shared_ptr<const DefaultExecutor> exec,
                  const batch::matrix::Dense<ValueType>* mat,
                  const batch::MultiVector<ValueType>* b,
                  batch::MultiVector<ValueType>* x)
{
    const auto b_ub = host::get_batch_struct(b);
    const auto x_ub = host::get_batch_struct(x);
    const auto mat_ub = host::get_batch_struct(mat);
#pragma omp parallel for
    for (size_type batch = 0; batch < x->get_num_batch_items(); ++batch) {
        const auto mat_item = batch::matrix::extract_batch_item(mat_ub, batch);
        const auto b_item = batch::extract_batch_item(b_ub, batch);
        const auto x_item = batch::extract_batch_item(x_ub, batch);
        simple_apply_kernel(mat_item, b_item, x_item);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_SIMPLE_APPLY_KERNEL);


template <typename ValueType>
void advanced_apply(std::shared_ptr<const DefaultExecutor> exec,
                    const batch::MultiVector<ValueType>* alpha,
                    const batch::matrix::Dense<ValueType>* mat,
                    const batch::MultiVector<ValueType>* b,
                    const batch::MultiVector<ValueType>* beta,
                    batch::MultiVector<ValueType>* x)
{
    const auto b_ub = host::get_batch_struct(b);
    const auto x_ub = host::get_batch_struct(x);
    const auto mat_ub = host::get_batch_struct(mat);
    const auto alpha_ub = host::get_batch_struct(alpha);
    const auto beta_ub = host::get_batch_struct(beta);
#pragma omp parallel for
    for (size_type batch = 0; batch < x->get_num_batch_items(); ++batch) {
        const auto mat_item = batch::matrix::extract_batch_item(mat_ub, batch);
        const auto b_item = batch::extract_batch_item(b_ub, batch);
        const auto x_item = batch::extract_batch_item(x_ub, batch);
        const auto alpha_item = batch::extract_batch_item(alpha_ub, batch);
        const auto beta_item = batch::extract_batch_item(beta_ub, batch);
        advanced_apply_kernel(alpha_item.values[0], mat_item, b_item,
                              beta_item.values[0], x_item);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_ADVANCED_APPLY_KERNEL);


}  // namespace batch_dense
}  // namespace omp
}  // namespace kernels
}  // namespace gko
