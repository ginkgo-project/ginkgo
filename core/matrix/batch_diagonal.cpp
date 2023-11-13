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

#include <ginkgo/core/matrix/batch_diagonal.hpp>


#include <algorithm>
#include <type_traits>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/batch_dim.hpp>
#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>


#include "core/matrix/batch_diagonal_kernels.hpp"


namespace gko {
namespace batch {
namespace matrix {
namespace diagonal {
namespace {


GKO_REGISTER_OPERATION(simple_apply, batch_diagonal::simple_apply);


}  // namespace
}  // namespace diagonal


template <typename ValueType>
std::unique_ptr<gko::matrix::Diagonal<ValueType>>
Diagonal<ValueType>::create_view_for_item(size_type item_id)
{
    auto exec = this->get_executor();
    auto num_rows = this->get_common_size()[0];
    auto mat = unbatch_type::create(
        exec, num_rows,
        make_array_view(exec, this->get_num_elements_per_item(),
                        this->get_values_for_item(item_id)));
    return mat;
}


template <typename ValueType>
std::unique_ptr<const gko::matrix::Diagonal<ValueType>>
Diagonal<ValueType>::create_const_view_for_item(size_type item_id) const
{
    auto exec = this->get_executor();
    auto num_rows = this->get_common_size()[0];
    auto mat = unbatch_type::create_const(
        exec, num_rows,
        make_const_array_view(exec, this->get_num_elements_per_item(),
                              this->get_const_values_for_item(item_id)));
    return mat;
}


template <typename ValueType>
std::unique_ptr<const Diagonal<ValueType>> Diagonal<ValueType>::create_const(
    std::shared_ptr<const Executor> exec, const batch_dim<2>& sizes,
    gko::detail::const_array_view<ValueType>&& values)
{
    // cast const-ness away, but return a const object afterwards,
    // so we can ensure that no modifications take place.
    return std::unique_ptr<const Diagonal>(new Diagonal{
        exec, sizes, gko::detail::array_const_cast(std::move(values))});
}


template <typename ValueType>
Diagonal<ValueType>::Diagonal(std::shared_ptr<const Executor> exec,
                              const batch_dim<2>& size)
    : EnableBatchLinOp<Diagonal<ValueType>>(exec, size),
      values_(exec, size.get_num_batch_items() * size.get_common_size()[0])
{
    GKO_ASSERT_BATCH_HAS_SQUARE_DIMENSIONS(this->get_size());
}


template <typename ValueType>
Diagonal<ValueType>* Diagonal<ValueType>::apply(
    ptr_param<const MultiVector<ValueType>> b,
    ptr_param<MultiVector<ValueType>> x)
{
    this->validate_application_parameters(b.get(), x.get());
    auto exec = this->get_executor();
    this->apply_impl(make_temporary_clone(exec, b).get(),
                     make_temporary_clone(exec, x).get());
    return this;
}


template <typename ValueType>
const Diagonal<ValueType>* Diagonal<ValueType>::apply(
    ptr_param<const MultiVector<ValueType>> b,
    ptr_param<MultiVector<ValueType>> x) const
{
    this->validate_application_parameters(b.get(), x.get());
    auto exec = this->get_executor();
    this->apply_impl(make_temporary_clone(exec, b).get(),
                     make_temporary_clone(exec, x).get());
    return this;
}


template <typename ValueType>
Diagonal<ValueType>* Diagonal<ValueType>::apply(
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


template <typename ValueType>
const Diagonal<ValueType>* Diagonal<ValueType>::apply(
    ptr_param<const MultiVector<ValueType>> alpha,
    ptr_param<const MultiVector<ValueType>> b,
    ptr_param<const MultiVector<ValueType>> beta,
    ptr_param<MultiVector<ValueType>> x) const
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


template <typename ValueType>
void Diagonal<ValueType>::apply_impl(const MultiVector<ValueType>* b,
                                     MultiVector<ValueType>* x) const
{
    this->get_executor()->run(diagonal::make_simple_apply(this, b, x));
}


template <typename ValueType>
void Diagonal<ValueType>::apply_impl(const MultiVector<ValueType>* alpha,
                                     const MultiVector<ValueType>* b,
                                     const MultiVector<ValueType>* beta,
                                     MultiVector<ValueType>* x) const
{
    auto x_clone = x->clone();
    this->apply(b, x_clone.get());
    x->scale(beta);
    x->add_scaled(alpha, x_clone.get());
}


template <typename ValueType>
void Diagonal<ValueType>::convert_to(
    Diagonal<next_precision<ValueType>>* result) const
{
    result->values_ = this->values_;
    result->set_size(this->get_size());
}


template <typename ValueType>
void Diagonal<ValueType>::move_to(Diagonal<next_precision<ValueType>>* result)
{
    this->convert_to(result);
}


#define GKO_DECLARE_BATCH_DIAGONAL_MATRIX(ValueType) class Diagonal<ValueType>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DIAGONAL_MATRIX);


}  // namespace matrix
}  // namespace batch
}  // namespace gko
