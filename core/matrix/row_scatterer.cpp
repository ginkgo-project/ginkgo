// SPDX-FileCopyrightText: 2024 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/matrix/row_scatterer.hpp"

namespace gko {
namespace matrix {


template <typename IndexType>
RowScatterer<IndexType>::RowScatterer(std::shared_ptr<const Executor> exec)
    : EnableLinOp<RowScatterer>(std::move(exec))
{}


template <typename IndexType>
RowScatterer<IndexType>::RowScatterer(std::shared_ptr<const Executor> exec,
                                      array<IndexType> idxs, size_type to_size)
    : EnableLinOp<RowScatterer>(exec, {to_size, idxs.get_size()}),
      idxs_(exec, std::move(idxs))
{}


template <typename IndexType>
void RowScatterer<IndexType>::apply_impl(const LinOp* b, LinOp* x) const
{}


template <typename IndexType>
void RowScatterer<IndexType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                         const LinOp* beta, LinOp* x) const
{}


}  // namespace matrix
}  // namespace gko
