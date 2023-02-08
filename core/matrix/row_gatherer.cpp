// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/matrix/row_gatherer.hpp"

#include <ginkgo/core/base/half.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/base/dispatch_helper.hpp"


namespace gko {
namespace matrix {


template <typename IndexType>
RowGatherer<IndexType>::RowGatherer(std::shared_ptr<const Executor> exec,
                                    const dim<2>& size)
    : EnableLinOp<RowGatherer>(exec, size), row_idxs_(exec, size[0])
{}


template <typename IndexType>
RowGatherer<IndexType>::RowGatherer(std::shared_ptr<const Executor> exec,
                                    const dim<2>& size,
                                    array<index_type> row_idxs)
    : EnableLinOp<RowGatherer>(exec, size), row_idxs_{exec, std::move(row_idxs)}
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
void RowGatherer<IndexType>::apply_impl(const LinOp* in, LinOp* out) const
{
    run<Dense,
#if GINKGO_ENABLE_HALF
        gko::half, std::complex<gko::half>,
#endif
        float, double, std::complex<float>, std::complex<double>>(
        in, [&](auto gather) { gather->row_gather(&row_idxs_, out); });
}

template <typename IndexType>
void RowGatherer<IndexType>::apply_impl(const LinOp* alpha, const LinOp* in,
                                        const LinOp* beta, LinOp* out) const
{
    run<Dense,
#if GINKGO_ENABLE_HALF
        gko::half, std::complex<gko::half>,
#endif
        float, double, std::complex<float>, std::complex<double>>(
        in,
        [&](auto gather) { gather->row_gather(alpha, &row_idxs_, beta, out); });
}


#define GKO_DECLARE_ROWGATHERER_MATRIX(_type) class RowGatherer<_type>
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_ROWGATHERER_MATRIX);


}  // namespace matrix
}  // namespace gko
