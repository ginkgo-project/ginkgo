// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/matrix/row_gatherer.hpp"

#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/multivector.hpp>

#include "core/base/dispatch_helper.hpp"


namespace gko {
namespace matrix {


template <typename IndexType>
RowGatherer<IndexType>::RowGatherer(std::shared_ptr<const Executor> exec,
                                    const dim<2>& size)
    : LinOp(exec, size), row_idxs_(exec, size[0])
{}


template <typename IndexType>
RowGatherer<IndexType>::RowGatherer(std::shared_ptr<const Executor> exec,
                                    const dim<2>& size,
                                    array<index_type> row_idxs)
    : LinOp(exec, size), row_idxs_{exec, std::move(row_idxs)}
{
    GKO_ASSERT_EQ(size[0], row_idxs_.get_size());
}


template <typename IndexType>
std::unique_ptr<RowGatherer<IndexType>> RowGatherer<IndexType>::create(
    std::shared_ptr<const Executor> exec, const dim<2>& size)
{
    return std::unique_ptr<RowGatherer>{new RowGatherer{exec, size}};
}


template <typename IndexType>
std::unique_ptr<RowGatherer<IndexType>> RowGatherer<IndexType>::create(
    std::shared_ptr<const Executor> exec, const dim<2>& size,
    array<index_type> row_idxs)
{
    return std::unique_ptr<RowGatherer>{
        new RowGatherer{exec, size, std::move(row_idxs)}};
}


template <typename IndexType>
std::unique_ptr<const RowGatherer<IndexType>>
RowGatherer<IndexType>::create_const(
    std::shared_ptr<const Executor> exec, const dim<2>& size,
    gko::detail::const_array_view<IndexType>&& row_idxs)
{
    // cast const-ness away, but return a const object afterwards,
    // so we can ensure that no modifications take place.
    return std::unique_ptr<const RowGatherer>{new RowGatherer{
        exec, size, gko::detail::array_const_cast(std::move(row_idxs))}};
}


template <typename IndexType>
void RowGatherer<IndexType>::apply_impl(const AbstractMultiVector* in,
                                        AbstractMultiVector* out) const
{
    std::visit(
        [this, in, out](auto p) {
            using value_type = std::decay_t<decltype(p)>;
            as<MultiVector<value_type>>(in)->row_gather(&row_idxs_, out);
        },
        precision_to_variant(in->get_precision()));
}

template <typename IndexType>
void RowGatherer<IndexType>::apply_impl(const AbstractMultiVector* alpha,
                                        const AbstractMultiVector* in,
                                        const AbstractMultiVector* beta,
                                        AbstractMultiVector* out) const
{
    std::visit(
        [this, in, alpha, beta, out](auto p) {
            using value_type = std::decay_t<decltype(p)>;
            as<MultiVector<value_type>>(in)->row_gather(alpha, &row_idxs_, beta,
                                                        out);
        },
        precision_to_variant(in->get_precision()));
}


#define GKO_DECLARE_ROWGATHERER_MATRIX(ValueType) class RowGatherer<ValueType>
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_ROWGATHERER_MATRIX);


}  // namespace matrix
}  // namespace gko
