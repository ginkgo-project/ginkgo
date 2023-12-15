// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/batch_csr.hpp>


#include <algorithm>
#include <type_traits>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/matrix/batch_csr_kernels.hpp"


namespace gko {
namespace batch {
namespace matrix {
namespace csr {
namespace {


GKO_REGISTER_OPERATION(simple_apply, batch_csr::simple_apply);
GKO_REGISTER_OPERATION(advanced_apply, batch_csr::advanced_apply);
GKO_REGISTER_OPERATION(scale, batch_csr::scale);
GKO_REGISTER_OPERATION(scale_add, batch_csr::scale_add);


}  // namespace
}  // namespace csr


template <typename ValueType, typename IndexType>
std::unique_ptr<gko::matrix::Csr<ValueType, IndexType>>
Csr<ValueType, IndexType>::create_view_for_item(size_type item_id)
{
    auto exec = this->get_executor();
    auto num_rows = this->get_common_size()[0];
    auto mat = unbatch_type::create(
        exec, this->get_common_size(),
        make_array_view(exec, this->get_num_elements_per_item(),
                        this->get_values_for_item(item_id)),
        make_array_view(exec, this->get_num_elements_per_item(),
                        this->get_col_idxs()),
        make_array_view(exec, num_rows + 1, this->get_row_ptrs()));
    return mat;
}


template <typename ValueType, typename IndexType>
std::unique_ptr<const gko::matrix::Csr<ValueType, IndexType>>
Csr<ValueType, IndexType>::create_const_view_for_item(size_type item_id) const
{
    auto exec = this->get_executor();
    auto num_rows = this->get_common_size()[0];
    auto mat = unbatch_type::create_const(
        exec, this->get_common_size(),
        make_const_array_view(exec, this->get_num_elements_per_item(),
                              this->get_const_values_for_item(item_id)),
        make_const_array_view(exec, this->get_num_elements_per_item(),
                              this->get_const_col_idxs()),
        make_const_array_view(exec, num_rows + 1, this->get_const_row_ptrs()));
    return mat;
}


template <typename ValueType, typename IndexType>
std::unique_ptr<const Csr<ValueType, IndexType>>
Csr<ValueType, IndexType>::create_const(
    std::shared_ptr<const Executor> exec, const batch_dim<2>& sizes,
    gko::detail::const_array_view<ValueType>&& values,
    gko::detail::const_array_view<IndexType>&& col_idxs,
    gko::detail::const_array_view<IndexType>&& row_ptrs)
{
    // cast const-ness away, but return a const object afterwards,
    // so we can ensure that no modifications take place.
    return std::unique_ptr<const Csr>(
        new Csr{exec, sizes, gko::detail::array_const_cast(std::move(values)),
                gko::detail::array_const_cast(std::move(col_idxs)),
                gko::detail::array_const_cast(std::move(row_ptrs))});
}


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>::Csr(std::shared_ptr<const Executor> exec,
                               const batch_dim<2>& size,
                               size_type num_nnz_per_item)
    : EnableBatchLinOp<Csr<ValueType, IndexType>>(exec, size),
      values_(exec, num_nnz_per_item * size.get_num_batch_items()),
      col_idxs_(exec, num_nnz_per_item),
      row_ptrs_(exec, size.get_common_size()[0] + 1)
{
    row_ptrs_.fill(0);
}


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>* Csr<ValueType, IndexType>::apply(
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
const Csr<ValueType, IndexType>* Csr<ValueType, IndexType>::apply(
    ptr_param<const MultiVector<ValueType>> b,
    ptr_param<MultiVector<ValueType>> x) const
{
    this->validate_application_parameters(b.get(), x.get());
    auto exec = this->get_executor();
    this->apply_impl(make_temporary_clone(exec, b).get(),
                     make_temporary_clone(exec, x).get());
    return this;
}


template <typename ValueType, typename IndexType>
Csr<ValueType, IndexType>* Csr<ValueType, IndexType>::apply(
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
const Csr<ValueType, IndexType>* Csr<ValueType, IndexType>::apply(
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


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::apply_impl(const MultiVector<ValueType>* b,
                                           MultiVector<ValueType>* x) const
{
    this->get_executor()->run(csr::make_simple_apply(this, b, x));
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::apply_impl(const MultiVector<ValueType>* alpha,
                                           const MultiVector<ValueType>* b,
                                           const MultiVector<ValueType>* beta,
                                           MultiVector<ValueType>* x) const
{
    this->get_executor()->run(
        csr::make_advanced_apply(alpha, this, b, beta, x));
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::scale_add(
    ptr_param<const MultiVector<ValueType>> alpha,
    ptr_param<const batch::matrix::Csr<ValueType, IndexType>> b)
{
    GKO_ASSERT_BATCH_EQUAL_NUM_ITEMS(alpha, b);
    GKO_ASSERT_BATCH_EQUAL_NUM_ITEMS(this, b);
    GKO_ASSERT_BATCH_EQUAL_DIMENSIONS(this, b);
    auto exec = this->get_executor();
    exec->run(csr::make_scale_add(make_temporary_clone(exec, alpha).get(),
                                  make_temporary_clone(exec, b).get(), this));
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::convert_to(
    Csr<next_precision<ValueType>, IndexType>* result) const
{
    result->values_ = this->values_;
    result->col_idxs_ = this->col_idxs_;
    result->row_ptrs_ = this->row_ptrs_;
    result->set_size(this->get_size());
}


template <typename ValueType, typename IndexType>
void Csr<ValueType, IndexType>::move_to(
    Csr<next_precision<ValueType>, IndexType>* result)
{
    this->convert_to(result);
}


#define GKO_DECLARE_BATCH_CSR_MATRIX(ValueType) class Csr<ValueType, int32>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_CSR_MATRIX);


template <typename ValueType, typename IndexType>
void two_sided_scale(
    const array<ValueType>& col_scale, const array<ValueType>& row_scale,
    std::shared_ptr<batch::matrix::Csr<ValueType, IndexType>>& in_out)
{
    GKO_ASSERT_EQ(col_scale.get_size(), (in_out->get_common_size()[1] *
                                         in_out->get_num_batch_items()));
    GKO_ASSERT_EQ(row_scale.get_size(), (in_out->get_common_size()[0] *
                                         in_out->get_num_batch_items()));
    in_out->get_executor()->run(
        csr::make_scale(&col_scale, &row_scale, in_out.get()));
}


#define GKO_DECLARE_TWO_SIDED_BATCH_SCALE(_vtype, _itype)               \
    void two_sided_scale(                                               \
        const array<_vtype>& col_scale, const array<_vtype>& row_scale, \
        std::shared_ptr<batch::matrix::Csr<_vtype, _itype>>& in_out)

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INT32_TYPE(
    GKO_DECLARE_TWO_SIDED_BATCH_SCALE);


}  // namespace matrix
}  // namespace batch
}  // namespace gko
