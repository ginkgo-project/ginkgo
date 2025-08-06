// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/matrix/conv.hpp"

#include <memory>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/matrix/conv_kernels.hpp"


namespace gko {
namespace matrix {
namespace conv {
namespace {


GKO_REGISTER_OPERATION(conv, conv::conv);


}  // namespace
}  // namespace conv


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

template <typename ValueType>
void Conv<ValueType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                 const LinOp* beta, LinOp* x) const
{
    // implmement
}


template <typename ValueType>
void Conv<ValueType>::validate_application_parameters(const LinOp* b,
                                                      const LinOp* x) const
{
    using gko::detail::get_size;
    const auto b_rows = get_size(b)[0];
    const auto x_rows = get_size(x)[0];
    const auto kernel_len = kernel_.get_size();

    if (x_rows != b_rows + kernel_len - 1) {
        throw DimensionMismatch(__FILE__, __LINE__, __func__, "x", x_rows, 1,
                                "b + kernel - 1", b_rows + kernel_len - 1, 1,
                                "x must have size = b + kernel - 1");
    }


    GKO_ASSERT_EQUAL_COLS(b, x);
}


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
std::unique_ptr<Conv<ValueType>> Conv<ValueType>::create(
    std::shared_ptr<const Executor> exec)
{
    return std::unique_ptr<Conv>{new Conv{exec}};
}

template <typename ValueType>
std::unique_ptr<Conv<ValueType>> Conv<ValueType>::create(
    std::shared_ptr<const Executor> exec, const array<ValueType>& array)
{
    return std::unique_ptr<Conv>{new Conv{exec, array}};
}


#define GKO_DECLARE_CONV(ValueType) class Conv<ValueType>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CONV);


}  // namespace matrix
}  // namespace gko
