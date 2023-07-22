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

#include "core/base/batch_multi_vector_kernels.hpp"


#include <algorithm>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/range_accessors.hpp>


#include "core/base/batch_struct.hpp"
#include "reference/base/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The BatchMultiVector matrix format namespace.
 * @ref BatchMultiVector
 * @ingroup batch_multi_vector
 */
namespace batch_multi_vector {


#include "reference/base/batch_multi_vector_kernels.hpp.inc"


template <typename ValueType>
void scale(std::shared_ptr<const DefaultExecutor> exec,
           const BatchMultiVector<ValueType>* alpha,
           BatchMultiVector<ValueType>* x)
{
    const auto x_ub = host::get_batch_struct(x);
    const auto alpha_ub = host::get_batch_struct(alpha);
    for (size_type batch = 0; batch < x->get_num_batch_entries(); ++batch) {
        const auto alpha_b = gko::batch::batch_entry(alpha_ub, batch);
        const auto x_b = gko::batch::batch_entry(x_ub, batch);
        scale_kernel(alpha_b, x_b);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_MULTI_VECTOR_SCALE_KERNEL);


template <typename ValueType>
void add_scaled(std::shared_ptr<const DefaultExecutor> exec,
                const BatchMultiVector<ValueType>* alpha,
                const BatchMultiVector<ValueType>* x,
                BatchMultiVector<ValueType>* y)
{
    const auto x_ub = host::get_batch_struct(x);
    const auto y_ub = host::get_batch_struct(y);
    const auto alpha_ub = host::get_batch_struct(alpha);
    for (size_type batch = 0; batch < y->get_num_batch_entries(); ++batch) {
        const auto alpha_b = gko::batch::batch_entry(alpha_ub, batch);
        const auto x_b = gko::batch::batch_entry(x_ub, batch);
        const auto y_b = gko::batch::batch_entry(y_ub, batch);
        add_scaled_kernel(alpha_b, x_b, y_b);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_MULTI_VECTOR_ADD_SCALED_KERNEL);


template <typename ValueType>
void compute_dot(std::shared_ptr<const DefaultExecutor> exec,
                 const BatchMultiVector<ValueType>* x,
                 const BatchMultiVector<ValueType>* y,
                 BatchMultiVector<ValueType>* result)
{
    const auto x_ub = host::get_batch_struct(x);
    const auto y_ub = host::get_batch_struct(y);
    const auto res_ub = host::get_batch_struct(result);
    for (size_type batch = 0; batch < result->get_num_batch_entries();
         ++batch) {
        const auto res_b = gko::batch::batch_entry(res_ub, batch);
        const auto x_b = gko::batch::batch_entry(x_ub, batch);
        const auto y_b = gko::batch::batch_entry(y_ub, batch);
        compute_dot_product_kernel(x_b, y_b, res_b);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_MULTI_VECTOR_COMPUTE_DOT_KERNEL);


template <typename ValueType>
void compute_conj_dot(std::shared_ptr<const DefaultExecutor> exec,
                      const BatchMultiVector<ValueType>* x,
                      const BatchMultiVector<ValueType>* y,
                      BatchMultiVector<ValueType>* result)
{
    const auto x_ub = host::get_batch_struct(x);
    const auto y_ub = host::get_batch_struct(y);
    const auto res_ub = host::get_batch_struct(result);
    for (size_type batch = 0; batch < result->get_num_batch_entries();
         ++batch) {
        const auto res_b = gko::batch::batch_entry(res_ub, batch);
        const auto x_b = gko::batch::batch_entry(x_ub, batch);
        const auto y_b = gko::batch::batch_entry(y_ub, batch);
        compute_conj_dot_product_kernel(x_b, y_b, res_b);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_MULTI_VECTOR_COMPUTE_CONJ_DOT_KERNEL);


template <typename ValueType>
void compute_norm2(std::shared_ptr<const DefaultExecutor> exec,
                   const BatchMultiVector<ValueType>* x,
                   BatchMultiVector<remove_complex<ValueType>>* result)
{
    const auto x_ub = host::get_batch_struct(x);
    const auto res_ub = host::get_batch_struct(result);
    for (size_type batch = 0; batch < result->get_num_batch_entries();
         ++batch) {
        const auto res_b = gko::batch::batch_entry(res_ub, batch);
        const auto x_b = gko::batch::batch_entry(x_ub, batch);
        compute_norm2_kernel(x_b, res_b);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_MULTI_VECTOR_COMPUTE_NORM2_KERNEL);


template <typename ValueType>
void copy(std::shared_ptr<const DefaultExecutor> exec,
          const BatchMultiVector<ValueType>* x,
          BatchMultiVector<ValueType>* result)
{
    const auto x_ub = host::get_batch_struct(x);
    const auto result_ub = host::get_batch_struct(result);
    for (size_type batch = 0; batch < x->get_num_batch_entries(); ++batch) {
        const auto result_b = gko::batch::batch_entry(result_ub, batch);
        const auto x_b = gko::batch::batch_entry(x_ub, batch);
        copy_kernel(x_b, result_b);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_MULTI_VECTOR_COPY_KERNEL);


}  // namespace batch_multi_vector
}  // namespace reference
}  // namespace kernels
}  // namespace gko
