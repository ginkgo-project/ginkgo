// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/matrix/conv.hpp"

#include <memory>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/matrix/conv_kernels.hpp"


namespace gko {
namespace matrix {
namespace conv {
namespace {

GKO_REGISTER_OPERATION(conv, conv::conv);

}  // namespace
}  // namespace conv

namespace conv2d {
namespace {

GKO_REGISTER_OPERATION(conv2d, conv2d::conv2d);

}  // namespace
}  // namespace conv2d

namespace conv2dsparse {
namespace {

GKO_REGISTER_OPERATION(conv2dsparse, conv2dsparse::conv2dsparse);

}  // namespace
}  // namespace conv2dsparse


// ------------------------------------------------------
// 1D convolution
// ------------------------------------------------------

template <typename ValueType>
void Conv<ValueType>::apply_impl(const LinOp* b, LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->get_executor()->run(
                conv::make_conv(kernel_, dense_b, dense_x));
        },
        b, x);
}


// ------------------------------------------------------
// 2D dense convolution
// ------------------------------------------------------

template <typename ValueType>
void Conv2d<ValueType>::apply_impl(const LinOp* b, LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            const auto input = dense_b;
            std::vector<decltype(dense_x)> outputs = {dense_x};

            // FIXED: remove the redundant 'const' here
            std::vector<decltype(kernels_[0].get())> kernel_ptrs;
            kernel_ptrs.reserve(kernels_.size());
            for (auto& k : kernels_) {
                kernel_ptrs.push_back(k.get());
            }

            this->get_executor()->run(
                conv2d::make_conv2d(kernel_ptrs, input, outputs));
        },
        b, x);
}


template <typename ValueType>
void Conv2d<ValueType>::apply_impl(const LinOp* b,
                                   const std::vector<LinOp*>& xs) const
{
    precision_dispatch_real_complex<ValueType>(
        [this, &xs](auto dense_b, auto /*ignored*/) {
            const auto input = dense_b;

            // Collect kernel pointers
            std::vector<decltype(kernels_[0].get())> kernel_ptrs;
            for (auto& k : kernels_) {
                kernel_ptrs.push_back(k.get());
            }

            // Convert all outputs to Dense
            std::vector<decltype(as<gko::matrix::Dense<ValueType>>(xs[0]))>
                outputs;
            outputs.reserve(xs.size());
            for (auto x : xs) {
                outputs.push_back(as<gko::matrix::Dense<ValueType>>(x));
            }

            // Run multi-filter convolution
            this->get_executor()->run(
                conv2d::make_conv2d(kernel_ptrs, input, outputs));
        },
        b,
        xs.front());  // dispatch uses only first output to determine precision
}


// ------------------------------------------------------
// 2D sparse convolution
// ------------------------------------------------------

template <typename ValueType, typename IndexType>
void Conv2dsparse<ValueType, IndexType>::apply_impl(const LinOp* b,
                                                    LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->get_executor()->run(conv2dsparse::make_conv2dsparse(
                kernel_.get(), dense_b, dense_x));
        },
        b, x);
}


// ------------------------------------------------------
// Alpha-beta apply overloads (placeholders)
// ------------------------------------------------------

template <typename ValueType>
void Conv<ValueType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                 const LinOp* beta, LinOp* x) const
{
    // not yet implemented
}

template <typename ValueType>
void Conv2d<ValueType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                   const LinOp* beta, LinOp* x) const
{
    // not yet implemented
}

template <typename ValueType, typename IndexType>
void Conv2dsparse<ValueType, IndexType>::apply_impl(const LinOp* alpha,
                                                    const LinOp* b,
                                                    const LinOp* beta,
                                                    LinOp* x) const
{
    // not yet implemented
}


// ------------------------------------------------------
// Validation
// ------------------------------------------------------

template <typename ValueType>
void Conv<ValueType>::validate_application_parameters(const LinOp* b,
                                                      const LinOp* x) const
{
    using gko::detail::get_size;
    const auto b_rows = get_size(b)[0];
    const auto x_rows = get_size(x)[0];
    const auto kernel_len = kernel_.get_size();

    if (x_rows != (b_rows + 2 * 0 - kernel_len) / 1 + 1) {
        throw DimensionMismatch(
            __FILE__, __LINE__, __func__, "x", x_rows, 1,
            "(b + 2*padding - kernel)/stride + 1",
            (b_rows + 2 * 0 - kernel_len) / 1 + 1, 1,
            "x must have size = (b + 2*padding - kernel)/stride + 1");
    }

    GKO_ASSERT_EQUAL_COLS(b, x);
    GKO_ASSERT_EQUAL_COLS(b, (gko::dim<2>{1, 1}));
}

template <typename ValueType>
void Conv2d<ValueType>::validate_application_parameters(const LinOp* b,
                                                        const LinOp* x) const
{
    using gko::detail::get_size;
    const auto b_rows = get_size(b)[0];
    const auto b_cols = get_size(b)[1];
    const auto x_rows = get_size(x)[0];
    const auto x_cols = get_size(x)[1];
    const auto kernel_rows = kernels_[0]->get_size()[0];
    const auto kernel_cols = kernels_[0]->get_size()[1];

    if (x_rows != (b_rows + 2 * 0 - kernel_rows) / 1 + 1) {
        throw DimensionMismatch(
            __FILE__, __LINE__, __func__, "x", x_rows, 1,
            "(b + 2*padding - kernel)/stride + 1",
            (b_rows + 2 * 0 - kernel_rows) / 1 + 1, 1,
            "x must have size = (b + 2*padding - kernel)/stride + 1");
    }
    if (x_cols != (b_cols + 2 * 0 - kernel_cols) / 1 + 1) {
        throw DimensionMismatch(
            __FILE__, __LINE__, __func__, "x", x_cols, 1,
            "(b + 2*padding - kernel)/stride + 1",
            (b_cols + 2 * 0 - kernel_cols) / 1 + 1, 1,
            "x must have size = (b + 2*padding - kernel)/stride + 1");
    }
}

template <typename ValueType, typename IndexType>
void Conv2dsparse<ValueType, IndexType>::validate_application_parameters(
    const LinOp* b, const LinOp* x) const
{
    // not yet implemented
}


// ------------------------------------------------------
// Constructors
// ------------------------------------------------------

template <typename ValueType>
Conv<ValueType>::Conv(std::shared_ptr<const Executor> exec)
    : EnableLinOp<Conv>(exec), kernel_{exec}
{}

template <typename ValueType>
Conv<ValueType>::Conv(std::shared_ptr<const Executor> exec,
                      const array<ValueType>& array)
    : EnableLinOp<Conv>(exec), kernel_{array}
{
    kernel_.set_executor(exec);
}

template <typename ValueType>
Conv2d<ValueType>::Conv2d(std::shared_ptr<const Executor> exec)
    : EnableLinOp<Conv2d>(exec),
      kernels_{{Dense<ValueType>::create(exec, dim<2>{}, 0)}}  // empty kernel
{}

template <typename ValueType>
Conv2d<ValueType>::Conv2d(
    std::shared_ptr<const Executor> exec,
    const std::vector<std::shared_ptr<const Dense<ValueType>>>& kernels)
    : EnableLinOp<Conv2d>(exec), kernels_{kernels}
{}

template <typename ValueType, typename IndexType>
Conv2dsparse<ValueType, IndexType>::Conv2dsparse(
    std::shared_ptr<const Executor> exec)
    : EnableLinOp<Conv2dsparse>(exec),
      kernel_{Csr<ValueType, IndexType>::create(exec, dim<2>{}, 0)}
{}

template <typename ValueType, typename IndexType>
Conv2dsparse<ValueType, IndexType>::Conv2dsparse(
    std::shared_ptr<const Executor> exec,
    std::shared_ptr<const Csr<ValueType, IndexType>> kernel)
    : EnableLinOp<Conv2dsparse>(exec), kernel_{std::move(kernel)}
{}


// ------------------------------------------------------
// Factory functions
// ------------------------------------------------------

template <typename ValueType>
std::unique_ptr<Conv<ValueType>> Conv<ValueType>::create(
    std::shared_ptr<const Executor> exec)
{
    return std::unique_ptr<Conv>{new Conv{exec}};
}

template <typename ValueType>
std::unique_ptr<Conv2d<ValueType>> Conv2d<ValueType>::create(
    std::shared_ptr<const Executor> exec)
{
    return std::unique_ptr<Conv2d>{new Conv2d{exec}};
}

template <typename ValueType>
std::unique_ptr<Conv2d<ValueType>> Conv2d<ValueType>::create(
    std::shared_ptr<const Executor> exec,
    const std::vector<std::shared_ptr<const Dense<ValueType>>>& kernels)
{
    return std::unique_ptr<Conv2d>{new Conv2d{exec, kernels}};
}

// backward-compatible single-kernel overload
template <typename ValueType>
std::unique_ptr<Conv2d<ValueType>> Conv2d<ValueType>::create(
    std::shared_ptr<const Executor> exec,
    std::shared_ptr<const Dense<ValueType>> kernel)
{
    return std::unique_ptr<Conv2d>{new Conv2d{exec, std::vector{kernel}}};
}

template <typename ValueType, typename IndexType>
std::unique_ptr<Conv2dsparse<ValueType, IndexType>>
Conv2dsparse<ValueType, IndexType>::create(std::shared_ptr<const Executor> exec)
{
    return std::unique_ptr<Conv2dsparse>{new Conv2dsparse{exec}};
}

template <typename ValueType>
std::unique_ptr<Conv<ValueType>> Conv<ValueType>::create(
    std::shared_ptr<const Executor> exec, const array<ValueType>& array)
{
    return std::unique_ptr<Conv>{new Conv{exec, array}};
}

template <typename ValueType, typename IndexType>
std::unique_ptr<Conv2dsparse<ValueType, IndexType>>
Conv2dsparse<ValueType, IndexType>::create(
    std::shared_ptr<const Executor> exec,
    std::shared_ptr<const Csr<ValueType, IndexType>> kernel)
{
    return std::unique_ptr<Conv2dsparse>{new Conv2dsparse{exec, kernel}};
}


// ------------------------------------------------------
// Explicit instantiation
// ------------------------------------------------------

#define GKO_DECLARE_CONV(ValueType) class Conv<ValueType>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CONV);

#define GKO_DECLARE_CONV2D(ValueType) class Conv2d<ValueType>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CONV2D);

#define GKO_DECLARE_CONV2DSPARSE(ValueType, IndexType) \
    class Conv2dsparse<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CONV2DSPARSE);

}  // namespace matrix
}  // namespace gko
