// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ginkgo/core/base/lin_op.hpp>

namespace gko {
namespace matrix {
class MultiVector;
}


class MixedPrecisionOperator final
    : public EnableLinOp<MixedPrecisionOperator> {
    friend class EnablePolymorphicObject;

public:
    static std::unique_ptr<MixedPrecisionOperator> create(
        ptr_param<const LinOp> op);

protected:
    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const matrix::MultiVector* b, matrix::MultiVector* x) const;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

    void apply_impl(const matrix::MultiVector* alpha,
                    const matrix::MultiVector* b,
                    const matrix::MultiVector* beta,
                    matrix::MultiVector* x) const;

private:
    MixedPrecisionOperator(std::shared_ptr<const Executor> exec);

    MixedPrecisionOperator(const LinOp* op);

    const LinOp* op_;
};

auto mixed_precision(ptr_param<const LinOp> op)
    -> std::unique_ptr<MixedPrecisionOperator>;

}  // namespace gko
