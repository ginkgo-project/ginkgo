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

#include <ginkgo/core/matrix/batch_ell.hpp>


#include <algorithm>
#include <type_traits>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/ell.hpp>


#include "core/matrix/batch_ell_kernels.hpp"


namespace gko {
namespace batch {
namespace matrix {
namespace ell {
namespace {


GKO_REGISTER_OPERATION(simple_apply, batch_ell::simple_apply);
GKO_REGISTER_OPERATION(advanced_apply, batch_ell::advanced_apply);


}  // namespace
}  // namespace ell


namespace detail {


template <typename ValueType, typename IndexType>
batch_dim<2> compute_batch_size(
    const std::vector<gko::matrix::Ell<ValueType, IndexType>*>& matrices)
{
    auto common_size = matrices[0]->get_size();
    for (size_type i = 1; i < matrices.size(); ++i) {
        GKO_ASSERT_EQUAL_DIMENSIONS(common_size, matrices[i]->get_size());
    }
    return batch_dim<2>{matrices.size(), common_size};
}


}  // namespace detail


template <typename ValueType, typename IndexType>
std::unique_ptr<gko::matrix::Ell<ValueType, IndexType>>
Ell<ValueType, IndexType>::create_view_for_item(size_type item_id)
{
    auto exec = this->get_executor();
    auto num_rows = this->get_common_size()[0];
    auto stride = this->get_common_size()[1];
    auto mat = unbatch_type::create(
        exec, this->get_common_size(),
        make_array_view(exec, this->get_num_elements_per_item(),
                        this->get_values_for_item(item_id)),
        make_array_view(exec, this->get_num_elements_per_item(),
                        this->get_col_idxs_for_item(item_id)),
        this->get_num_stored_elements_per_row(), stride);
    return mat;
}


template <typename ValueType, typename IndexType>
std::unique_ptr<const gko::matrix::Ell<ValueType, IndexType>>
Ell<ValueType, IndexType>::create_const_view_for_item(size_type item_id) const
{
    auto exec = this->get_executor();
    auto num_rows = this->get_common_size()[0];
    auto stride = this->get_common_size()[1];
    auto mat = unbatch_type::create_const(
        exec, this->get_common_size(),
        make_const_array_view(exec, this->get_num_elements_per_item(),
                              this->get_const_values_for_item(item_id)),
        make_const_array_view(exec, this->get_num_elements_per_item(),
                              this->get_const_col_idxs_for_item(item_id)),
        this->get_num_stored_elements_per_row(), stride);
    return mat;
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Ell<ValueType, IndexType>>
Ell<ValueType, IndexType>::create_with_config_of(
    ptr_param<const Ell<ValueType, IndexType>> other)
{
    // De-referencing `other` before calling the functions (instead of
    // using operator `->`) is currently required to be compatible with
    // CUDA 10.1.
    // Otherwise, it results in a compile error.
    return (*other).create_with_same_config();
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Ell<ValueType, IndexType>>
Ell<ValueType, IndexType>::create_with_same_config() const
{
    return Ell<ValueType, IndexType>::create(
        this->get_executor(), this->get_size(),
        this->get_num_stored_elements_per_row());
}


template <typename ValueType, typename IndexType>
std::unique_ptr<const Ell<ValueType, IndexType>>
Ell<ValueType, IndexType>::create_const(
    std::shared_ptr<const Executor> exec, const batch_dim<2>& sizes,
    int num_elems_per_row, gko::detail::const_array_view<ValueType>&& values,
    gko::detail::const_array_view<IndexType>&& col_idxs)
{
    // cast const-ness away, but return a const object afterwards,
    // so we can ensure that no modifications take place.
    return std::unique_ptr<const Ell>(
        new Ell{exec, sizes, num_elems_per_row,
                gko::detail::array_const_cast(std::move(values)),
                gko::detail::array_const_cast(std::move(col_idxs))});
}


inline const batch_dim<2> get_col_sizes(const batch_dim<2>& sizes)
{
    return batch_dim<2>(sizes.get_num_batch_items(),
                        dim<2>(1, sizes.get_common_size()[1]));
}


template <typename ValueType, typename IndexType>
Ell<ValueType, IndexType>::Ell(std::shared_ptr<const Executor> exec,
                               const batch_dim<2>& size, int num_elems_per_row)
    : EnableBatchLinOp<Ell<ValueType, IndexType>>(exec, size),
      num_elems_per_row_(num_elems_per_row),
      values_(exec, compute_num_elems(size, num_elems_per_row)),
      col_idxs_(exec, compute_num_elems(size, num_elems_per_row))
{}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::apply_impl(const MultiVector<ValueType>* b,
                                           MultiVector<ValueType>* x) const
{
    GKO_ASSERT_EQ(b->get_num_batch_items(), this->get_num_batch_items());
    GKO_ASSERT_EQ(this->get_num_batch_items(), x->get_num_batch_items());

    GKO_ASSERT_CONFORMANT(this->get_common_size(), b->get_common_size());
    GKO_ASSERT_EQUAL_ROWS(this->get_common_size(), x->get_common_size());
    GKO_ASSERT_EQUAL_COLS(b->get_common_size(), x->get_common_size());
    this->get_executor()->run(ell::make_simple_apply(this, b, x));
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::apply_impl(const MultiVector<ValueType>* alpha,
                                           const MultiVector<ValueType>* b,
                                           const MultiVector<ValueType>* beta,
                                           MultiVector<ValueType>* x) const
{
    GKO_ASSERT_EQ(b->get_num_batch_items(), this->get_num_batch_items());
    GKO_ASSERT_EQ(this->get_num_batch_items(), x->get_num_batch_items());

    GKO_ASSERT_CONFORMANT(this->get_common_size(), b->get_common_size());
    GKO_ASSERT_EQUAL_ROWS(this->get_common_size(), x->get_common_size());
    GKO_ASSERT_EQUAL_COLS(b->get_common_size(), x->get_common_size());
    GKO_ASSERT_EQUAL_DIMENSIONS(alpha->get_common_size(), gko::dim<2>(1, 1));
    GKO_ASSERT_EQUAL_DIMENSIONS(beta->get_common_size(), gko::dim<2>(1, 1));
    this->get_executor()->run(
        ell::make_advanced_apply(alpha, this, b, beta, x));
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::convert_to(
    Ell<next_precision<ValueType, IndexType>>* result) const
{
    result->values_ = this->values_;
    result->col_idxs_ = this->col_idxs_;
    result->num_elems_per_row_ = this->num_elems_per_row_;
    result->set_size(this->get_size());
}


template <typename ValueType>
void Ell<ValueType, IndexType>::move_to(
    Ell<next_precision<ValueType, IndexType>>* result)
{
    this->convert_to(result);
}


#define GKO_DECLARE_BATCH_ELL_MATRIX(_type) class Ell<_vtype, _itype>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_BATCH_ELL_MATRIX);


}  // namespace matrix
}  // namespace batch
}  // namespace gko
