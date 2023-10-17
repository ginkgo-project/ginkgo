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


template <typename ValueType, typename IndexType>
std::unique_ptr<gko::matrix::Ell<ValueType, IndexType>>
Ell<ValueType, IndexType>::create_view_for_item(size_type item_id)
{
    auto exec = this->get_executor();
    auto num_rows = this->get_common_size()[0];
    auto stride = this->get_common_size()[0];
    auto mat = unbatch_type::create(
        exec, this->get_common_size(),
        make_array_view(exec, this->get_num_elements_per_item(),
                        this->get_values_for_item(item_id)),
        make_array_view(exec, this->get_num_elements_per_item(),
                        this->get_col_idxs()),
        this->get_num_stored_elements_per_row(), stride);
    return mat;
}


template <typename ValueType, typename IndexType>
std::unique_ptr<const gko::matrix::Ell<ValueType, IndexType>>
Ell<ValueType, IndexType>::create_const_view_for_item(size_type item_id) const
{
    auto exec = this->get_executor();
    auto num_rows = this->get_common_size()[0];
    auto stride = this->get_common_size()[0];
    auto mat = unbatch_type::create_const(
        exec, this->get_common_size(),
        make_const_array_view(exec, this->get_num_elements_per_item(),
                              this->get_const_values_for_item(item_id)),
        make_const_array_view(exec, this->get_num_elements_per_item(),
                              this->get_const_col_idxs()),
        this->get_num_stored_elements_per_row(), stride);
    return mat;
}


template <typename ValueType, typename IndexType>
std::unique_ptr<const Ell<ValueType, IndexType>>
Ell<ValueType, IndexType>::create_const(
    std::shared_ptr<const Executor> exec, const batch_dim<2>& sizes,
    const IndexType num_elems_per_row,
    gko::detail::const_array_view<ValueType>&& values,
    gko::detail::const_array_view<IndexType>&& col_idxs)
{
    // cast const-ness away, but return a const object afterwards,
    // so we can ensure that no modifications take place.
    return std::unique_ptr<const Ell>(
        new Ell{exec, sizes, num_elems_per_row,
                gko::detail::array_const_cast(std::move(values)),
                gko::detail::array_const_cast(std::move(col_idxs))});
}


template <typename ValueType, typename IndexType>
Ell<ValueType, IndexType>::Ell(std::shared_ptr<const Executor> exec,
                               const batch_dim<2>& size,
                               IndexType num_elems_per_row)
    : EnableBatchLinOp<Ell<ValueType, IndexType>>(exec, size),
      num_elems_per_row_(num_elems_per_row == 0 ? size.get_common_size()[1]
                                                : num_elems_per_row),
      values_(exec, compute_num_elems(size, num_elems_per_row_)),
      col_idxs_(exec, this->get_common_size()[0] * num_elems_per_row_)
{}


template <typename ValueType, typename IndexType>
Ell<ValueType, IndexType>* Ell<ValueType, IndexType>::apply(
    ptr_param<const MultiVector<ValueType>> b,
    ptr_param<MultiVector<ValueType>> x)
{
    this->validate_application_parameters(b.get(), x.get());
    auto exec = this->get_executor();
    this->apply_impl(make_temporary_clone(exec, b).get(),
                     make_temporary_clone(exec, x).get());
    return this;
}


template <typename ValueType, typename IndexType>
const Ell<ValueType, IndexType>* Ell<ValueType, IndexType>::apply(
    ptr_param<const MultiVector<ValueType>> b,
    ptr_param<MultiVector<ValueType>> x) const
{
    this->apply(b, x);
    return this;
}


template <typename ValueType, typename IndexType>
Ell<ValueType, IndexType>* Ell<ValueType, IndexType>::apply(
    ptr_param<const MultiVector<ValueType>> alpha,
    ptr_param<const MultiVector<ValueType>> b,
    ptr_param<const MultiVector<ValueType>> beta,
    ptr_param<MultiVector<ValueType>> x)
{
    this->validate_application_parameters(alpha.get(), b.get(), beta.get(),
                                          x.get());
    auto exec = this->get_executor();
    this->apply_impl(make_temporary_clone(exec, alpha).get(),
                     make_temporary_clone(exec, b).get(),
                     make_temporary_clone(exec, beta).get(),
                     make_temporary_clone(exec, x).get());
    return this;
}


template <typename ValueType, typename IndexType>
const Ell<ValueType, IndexType>* Ell<ValueType, IndexType>::apply(
    ptr_param<const MultiVector<ValueType>> alpha,
    ptr_param<const MultiVector<ValueType>> b,
    ptr_param<const MultiVector<ValueType>> beta,
    ptr_param<MultiVector<ValueType>> x) const
{
    this->apply(alpha, b, beta, x);
    return this;
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::apply_impl(const MultiVector<ValueType>* b,
                                           MultiVector<ValueType>* x) const
{
    this->get_executor()->run(ell::make_simple_apply(this, b, x));
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::apply_impl(const MultiVector<ValueType>* alpha,
                                           const MultiVector<ValueType>* b,
                                           const MultiVector<ValueType>* beta,
                                           MultiVector<ValueType>* x) const
{
    this->get_executor()->run(
        ell::make_advanced_apply(alpha, this, b, beta, x));
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::convert_to(
    Ell<next_precision<ValueType>, IndexType>* result) const
{
    result->values_ = this->values_;
    result->col_idxs_ = this->col_idxs_;
    result->num_elems_per_row_ = this->num_elems_per_row_;
    result->set_size(this->get_size());
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::move_to(
    Ell<next_precision<ValueType>, IndexType>* result)
{
    this->convert_to(result);
}


#define GKO_DECLARE_BATCH_ELL_MATRIX(ValueType) class Ell<ValueType, int32>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_ELL_MATRIX);


}  // namespace matrix
}  // namespace batch
}  // namespace gko
