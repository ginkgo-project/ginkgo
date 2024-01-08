// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/row_gatherer.hpp>


#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/dispatch_helper.hpp"


namespace gko {
namespace matrix {


template <typename IndexType>
void RowGatherer<IndexType>::apply_impl(const LinOp* in, LinOp* out) const
{
    run<const Dense<float>*, const Dense<double>*,
        const Dense<std::complex<float>>*, const Dense<std::complex<double>>*>(
        in, [&](auto gather) { gather->row_gather(&row_idxs_, out); });
}

template <typename IndexType>
void RowGatherer<IndexType>::apply_impl(const LinOp* alpha, const LinOp* in,
                                        const LinOp* beta, LinOp* out) const
{
    run<const Dense<float>*, const Dense<double>*,
        const Dense<std::complex<float>>*, const Dense<std::complex<double>>*>(
        in,
        [&](auto gather) { gather->row_gather(alpha, &row_idxs_, beta, out); });
}


#define GKO_DECLARE_ROWGATHERER_MATRIX(_type) class RowGatherer<_type>
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_ROWGATHERER_MATRIX);


}  // namespace matrix
}  // namespace gko
