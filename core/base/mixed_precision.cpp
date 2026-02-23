// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/mixed_precision.hpp>

#include "ginkgo/core/matrix/multivector.hpp"

namespace gko {


std::unique_ptr<MixedPrecisionOperator> MixedPrecisionOperator::create(
    ptr_param<const LinOp> op)
{
    return std::unique_ptr<MixedPrecisionOperator>(
        new MixedPrecisionOperator(op.get()));
}


void MixedPrecisionOperator::apply_impl(const LinOp* b, LinOp* x) const
{
    this->apply_impl(dynamic_cast<const matrix::MultiVector*>(b),
                     dynamic_cast<matrix::MultiVector*>(x));
}


template <typename MultiVector>
TemporaryPtr<MultiVector> as_precision(const LinOp* op, MultiVector* x)
{
    auto required_precision = op->supports_mixed_precision()
                                  ? x->get_precision()
                                  : op->get_precision();
    return x->as_precision(required_precision);
}


void MixedPrecisionOperator::apply_impl(const matrix::MultiVector* b,
                                        matrix::MultiVector* x) const
{
    auto b_tmp = as_precision(op_, b);
    auto x_tmp = as_precision(op_, x);
    (void)op_->apply(b_tmp.get(), x_tmp.get());
    x_tmp.copy_back();
}


void MixedPrecisionOperator::apply_impl(const LinOp* alpha, const LinOp* b,
                                        const LinOp* beta, LinOp* x) const
{
    this->apply_impl(dynamic_cast<const matrix::MultiVector*>(alpha),
                     dynamic_cast<const matrix::MultiVector*>(b),
                     dynamic_cast<const matrix::MultiVector*>(beta),
                     dynamic_cast<matrix::MultiVector*>(x));
}


void MixedPrecisionOperator::apply_impl(const matrix::MultiVector* alpha,
                                        const matrix::MultiVector* b,
                                        const matrix::MultiVector* beta,
                                        matrix::MultiVector* x) const
{
    auto b_tmp = as_precision(op_, b);
    auto x_tmp = as_precision(op_, x);
    (void)op_->apply(alpha, b_tmp.get(), beta, x_tmp.get());
    x_tmp.copy_back();
}


MixedPrecisionOperator::MixedPrecisionOperator(
    std::shared_ptr<const Executor> exec)
    : EnableLinOp(std::move(exec)), op_(nullptr)
{}


MixedPrecisionOperator::MixedPrecisionOperator(const LinOp* op)
    : EnableLinOp(op->get_executor(), op->get_size()), op_(op)
{}


auto mixed_precision(ptr_param<const LinOp> op)
    -> std::unique_ptr<MixedPrecisionOperator>
{
    return MixedPrecisionOperator::create(op);
}


}  // namespace gko
