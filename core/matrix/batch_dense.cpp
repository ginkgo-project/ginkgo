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

#include <ginkgo/core/matrix/batch_dense.hpp>


#include <algorithm>
#include <type_traits>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/utils.hpp>


#include "core/matrix/batch_dense_kernels.hpp"


namespace gko {
namespace batch {
namespace matrix {
namespace dense {


GKO_REGISTER_OPERATION(simple_apply, batch_dense::simple_apply);
GKO_REGISTER_OPERATION(advanced_apply, batch_dense::advanced_apply);


}  // namespace dense


namespace detail {


template <typename ValueType>
batch_dim<2> compute_batch_size(
    const std::vector<matrix::Dense<ValueType>*>& matrices)
{
    auto common_size = matrices[0]->get_size();
    for (size_type i = 1; i < matrices.size(); ++i) {
        GKO_ASSERT_EQUAL_DIMENSIONS(common_size, matrices[i]->get_size());
    }
    return batch_dim<2>{matrices.size(), common_size};
}


}  // namespace detail


template <typename ValueType>
std::unique_ptr<matrix::Dense<ValueType>>
BatchDense<ValueType>::create_view_for_item(size_type item_id)
{
    auto exec = this->get_executor();
    auto num_rows = this->get_common_size()[0];
    auto stride = this->get_common_size()[1];
    auto mat = unbatch_type::create(
        exec, this->get_common_size(),
        make_array_view(exec, num_rows * stride,
                        this->get_values_for_item(item_id)),
        stride);
    return mat;
}


template <typename ValueType>
std::unique_ptr<const matrix::Dense<ValueType>>
BatchDense<ValueType>::create_const_view_for_item(size_type item_id) const
{
    auto exec = this->get_executor();
    auto num_rows = this->get_common_size()[0];
    auto stride = this->get_common_size()[1];
    auto mat = unbatch_type::create_const(
        exec, this->get_common_size(),
        make_const_array_view(exec, num_rows * stride,
                              this->get_const_values_for_item(item_id)),
        stride);
    return mat;
}


template <typename ValueType>
std::unique_ptr<BatchDense<ValueType>>
BatchDense<ValueType>::create_with_config_of(ptr_param<const MultiVector> other)
{
    // De-referencing `other` before calling the functions (instead of
    // using operator `->`) is currently required to be compatible with
    // CUDA 10.1.
    // Otherwise, it results in a compile error.
    return (*other).create_with_same_config();
}


template <typename ValueType>
void BatchDense<ValueType>::set_size(const batch_dim<2>& value) noexcept
{
    batch_size_ = value;
}


template <typename ValueType>
std::unique_ptr<BatchDense<ValueType>>
BatchDense<ValueType>::create_with_same_config() const
{
    return BatchDense<ValueType>::create(this->get_executor(),
                                         this->get_size());
}


inline const batch_dim<2> get_col_sizes(const batch_dim<2>& sizes)
{
    return batch_dim<2>(sizes.get_num_batch_items(),
                        dim<2>(1, sizes.get_common_size()[1]));
}


template <typename ValueType>
void BatchDense<ValueType>::apply_impl(const MultiVector<ValueType>* b,
                                       MultiVector<ValueType>* x) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(b->get_common_size(), x->get_common_size());
    GKO_ASSERT_EQ(b->get_num_batch_items(), this->get_num_batch_items());
    GKO_ASSERT_CONFORMANT(this->get_common_size(), b->get_common_size());
    GKO_ASSERT_EQ(this->get_num_batch_items(), x->get_num_batch_items());
    GKO_ASSERT_CONFORMANT(this->get_common_size(), x->get_common_size());
    this->get_executor()->run(batch_dense::make_simple_apply(this, b, x));
}


template <typename ValueType>
void BatchDense<ValueType>::apply_impl(const MultiVector<ValueType>* alpha,
                                       const MultiVector<ValueType>* b,
                                       const MultiVector<ValueType>* beta,
                                       MultiVector<ValueType>* x) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(b->get_common_size(), x->get_common_size());
    GKO_ASSERT_EQ(b->get_num_batch_items(), this->get_num_batch_items());
    GKO_ASSERT_CONFORMANT(this->get_common_size(), b->get_common_size());
    GKO_ASSERT_EQ(this->get_num_batch_items(), x->get_num_batch_items());
    GKO_ASSERT_CONFORMANT(this->get_common_size(), x->get_common_size());
    GKO_ASSERT_EQUAL_COLS(alpha->get_common_size(), gko::dim<2>(1, 1));
    GKO_ASSERT_EQUAL_COLS(beta->get_common_size(), gko::dim<2>(1, 1));
    this->get_executor()->run(
        batch_dense::make_advanced_apply(alpha, this, b, beta, x));
}


template <typename ValueType>
void BatchDense<ValueType>::convert_to(
    BatchDense<next_precision<ValueType>>* result) const
{
    result->values_ = this->values_;
    result->set_size(this->get_size());
}


template <typename ValueType>
void BatchDense<ValueType>::move_to(
    BatchDense<next_precision<ValueType>>* result)
{
    this->convert_to(result);
}


#define GKO_DECLARE_BATCH_DENSE_MATRIX(_type) class BatchDense<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_MATRIX);


}  // namespace matrix
}  // namespace batch
}  // namespace gko
