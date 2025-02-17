// SPDX-FileCopyrightText: 2024 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ginkgo/core/base/lin_op.hpp>

namespace gko {
namespace matrix {


template <typename IndexType = int32>
class RowScatterer : public EnableLinOp<RowScatterer<IndexType>> {
    friend class EnablePolymorphicObject<RowScatterer<IndexType>, LinOp>;

public:
    static std::unique_ptr<RowScatterer> create(
        std::shared_ptr<const Executor> exec, array<IndexType> idxs,
        size_type to_size);

protected:
    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

private:
    explicit RowScatterer(std::shared_ptr<const Executor> exec);

    explicit RowScatterer(std::shared_ptr<const Executor> exec,
                          array<IndexType> idxs, size_type to_size);

    array<IndexType> idxs_;
};


}  // namespace matrix
}  // namespace gko
