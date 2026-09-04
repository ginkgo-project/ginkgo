// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/matrix/scaled_permutation.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>

#include "core/base/dispatch_helper.hpp"
#include "core/matrix/scaled_permutation_kernels.hpp"


namespace gko {
namespace matrix {
namespace scaled_permutation {
namespace {


GKO_REGISTER_OPERATION(invert, scaled_permutation::invert);
GKO_REGISTER_OPERATION(compose, scaled_permutation::compose);


}  // namespace
}  // namespace scaled_permutation


template <typename ValueType, typename IndexType>
ScaledPermutation<ValueType, IndexType>::ScaledPermutation(
    std::shared_ptr<const Executor> exec, size_type size)
    : ScaledPermutation{exec, array<ValueType>{exec, size},
                        array<IndexType>{exec, size}}
{}


template <typename ValueType, typename IndexType>
ScaledPermutation<ValueType, IndexType>::ScaledPermutation(
    std::shared_ptr<const Executor> exec, array<value_type> scaling_factors,
    array<index_type> permutation_indices)
    : LinOp(exec,
            dim<2>{scaling_factors.get_size(), scaling_factors.get_size()}),
      scale_{exec, std::move(scaling_factors)},
      permutation_{exec, std::move(permutation_indices)}
{
    GKO_ASSERT_EQ(scale_.get_size(), permutation_.get_size());
}


template <typename ValueType, typename IndexType>
std::unique_ptr<ScaledPermutation<ValueType, IndexType>>
ScaledPermutation<ValueType, IndexType>::create(
    std::shared_ptr<const Executor> exec, size_type size)
{
    return std::unique_ptr<ScaledPermutation>{
        new ScaledPermutation{exec, size}};
}


template <typename ValueType, typename IndexType>
std::unique_ptr<ScaledPermutation<ValueType, IndexType>>
ScaledPermutation<ValueType, IndexType>::create(
    ptr_param<const Permutation<IndexType>> permutation)
{
    const auto exec = permutation->get_executor();
    const auto size = permutation->get_size()[0];
    array<value_type> scale{exec, size};
    array<index_type> perm{exec, size};
    exec->copy(size, permutation->get_const_permutation(), perm.get_data());
    scale.fill(one<ValueType>());
    return create(exec, std::move(scale), std::move(perm));
}


template <typename ValueType, typename IndexType>
std::unique_ptr<ScaledPermutation<ValueType, IndexType>>
ScaledPermutation<ValueType, IndexType>::create(
    std::shared_ptr<const Executor> exec, array<value_type> scaling_factors,
    array<index_type> permutation_indices)
{
    return std::unique_ptr<ScaledPermutation>{new ScaledPermutation{
        exec, std::move(scaling_factors), std::move(permutation_indices)}};
}


template <typename ValueType, typename IndexType>
std::unique_ptr<const ScaledPermutation<ValueType, IndexType>>
ScaledPermutation<ValueType, IndexType>::create_const(
    std::shared_ptr<const Executor> exec,
    gko::detail::const_array_view<value_type>&& scale,
    gko::detail::const_array_view<index_type>&& perm_idxs)
{
    return create(exec, gko::detail::array_const_cast(std::move(scale)),
                  gko::detail::array_const_cast(std::move(perm_idxs)));
}


template <typename ValueType, typename IndexType>
std::unique_ptr<ScaledPermutation<ValueType, IndexType>>
ScaledPermutation<ValueType, IndexType>::compute_inverse() const
{
    const auto exec = this->get_executor();
    const auto size = this->get_size()[0];
    auto result = ScaledPermutation::create(exec, size);
    exec->run(scaled_permutation::make_invert(
        this->get_const_scaling_factors(), this->get_const_permutation(), size,
        result->get_scaling_factors(), result->get_permutation()));
    return result;
}


template <typename ValueType, typename IndexType>
std::unique_ptr<ScaledPermutation<ValueType, IndexType>>
ScaledPermutation<ValueType, IndexType>::compose(
    ptr_param<const ScaledPermutation> other) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(this, other);
    const auto exec = this->get_executor();
    const auto size = this->get_size()[0];
    const auto local_other = make_temporary_clone(exec, other);
    auto result = ScaledPermutation::create(exec, size);
    exec->run(scaled_permutation::make_compose(
        this->get_const_scaling_factors(), this->get_const_permutation(),
        local_other->get_const_scaling_factors(),
        local_other->get_const_permutation(), size,
        result->get_scaling_factors(), result->get_permutation()));
    return result;
}


template <typename ValueType, typename IndexType>
void ScaledPermutation<ValueType, IndexType>::apply_impl(
    const AbstractMultiVector* b, AbstractMultiVector* x) const
{
    using dense_type = MultiVector<ValueType>;
    as<dense_type>(b->as_precision(precision_v<ValueType>))
        ->scale_permute(
            this, as<dense_type>(x->as_precision(precision_v<ValueType>).get()),
            permute_mode::rows);
}


template <typename ValueType, typename IndexType>
void ScaledPermutation<ValueType, IndexType>::apply_impl(
    const AbstractMultiVector* alpha, const AbstractMultiVector* b,
    const AbstractMultiVector* beta, AbstractMultiVector* x) const
{
    using dense_type = MultiVector<ValueType>;
    auto tmp = as<dense_type>(b->as_precision(precision_v<ValueType>))
                   ->scale_permute(this, permute_mode::rows);
    auto converted_x = as<dense_type>(x->as_precision(precision_v<ValueType>));
    converted_x->scale(beta);
    converted_x->add_scaled(alpha, tmp);
}


template <typename ValueType, typename IndexType>
void ScaledPermutation<ValueType, IndexType>::write(
    gko::matrix_data<value_type, index_type>& data) const
{
    const auto host_this =
        make_temporary_clone(this->get_executor()->get_master(), this);
    data.size = this->get_size();
    data.nonzeros.clear();
    data.nonzeros.reserve(data.size[0]);
    for (IndexType row = 0; row < this->get_size()[0]; row++) {
        auto col = host_this->get_const_permutation()[row];
        data.nonzeros.emplace_back(row, col,
                                   host_this->get_const_scaling_factors()[col]);
    }
}


#define GKO_DECLARE_SCALED_PERMUTATION_MATRIX(ValueType, IndexType) \
    class ScaledPermutation<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SCALED_PERMUTATION_MATRIX);


}  // namespace matrix
}  // namespace gko
