// SPDX-FileCopyrightText: 2024 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/matrix/row_scatterer.hpp"

#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/base/dispatch_helper.hpp"
#include "core/components/bit_packed_storage.hpp"
#include "core/matrix/row_scatterer_kernels.hpp"

namespace gko {
namespace matrix {
namespace {


GKO_REGISTER_OPERATION(row_scatter, row_scatter::row_scatter);
GKO_REGISTER_OPERATION(advanced_row_scatter, row_scatter::advanced_row_scatter);


}  // namespace


template <typename IndexType>
std::unique_ptr<RowScatterer<IndexType>> RowScatterer<IndexType>::create(
    std::shared_ptr<const Executor> exec, array<IndexType> idxs,
    size_type to_size)
{
    return std::unique_ptr<RowScatterer>(
        new RowScatterer(std::move(exec), std::move(idxs), to_size));
}


template <typename IndexType>
RowScatterer<IndexType>::RowScatterer(std::shared_ptr<const Executor> exec)
    : EnableLinOp<RowScatterer<IndexType>>(std::move(exec))
{}


template <typename IndexType>
RowScatterer<IndexType>::RowScatterer(std::shared_ptr<const Executor> exec,
                                      array<IndexType> idxs, size_type to_size)
    : EnableLinOp<RowScatterer<IndexType>>(exec, {to_size, idxs.get_size()}),
      idxs_(exec, std::move(idxs)),
      mask_(exec,
            bit_packed_span<bool, IndexType, uint32>::storage_size(to_size, 1))
{}


template <typename IndexType>
void RowScatterer<IndexType>::apply_impl(const LinOp* b, LinOp* x) const
{
    auto impl = [&](const auto* orig, auto* target) {
        auto exec = this->get_executor();
        bool invalid_access = false;

        exec->run(make_row_scatter(&idxs_, orig, target, invalid_access));

        // TODO: find a uniform way to handle device-side errors
        if (invalid_access) {
            GKO_INVALID_STATE("Out-of-bounds scatter index detected.");
        }
    };

    run<Dense,
#if GINKGO_ENABLE_HALF
        gko::half, std::complex<gko::half>,
#endif
        float, double, std::complex<float>, std::complex<double>>(
        b, [&](auto* orig) {
            using value_type =
                typename std::decay_t<decltype(*orig)>::value_type;
            mixed_precision_dispatch_real_complex<value_type>(impl, orig, x);
        });
}


template <typename IndexType>
void RowScatterer<IndexType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                         const LinOp* beta, LinOp* x) const
{
    auto impl = [&](const auto* orig, auto* target) {
        auto exec = this->get_executor();
        bool invalid_access = false;

        auto dense_alpha = make_temporary_conversion<
            typename std::decay_t<decltype(*orig)>::value_type>(alpha);
        auto dense_beta = make_temporary_conversion<
            typename std::decay_t<decltype(*target)>::value_type>(beta);

        exec->run(make_advanced_row_scatter(
            &idxs_, dense_alpha.get(), orig, dense_beta.get(), target,
            bit_packed_span<bool, IndexType, uint32>(mask_.get_data(), 1,
                                                     this->get_size()[0]),
            invalid_access));

        if (invalid_access) {
            GKO_INVALID_STATE("Out-of-bounds scatter index detected.");
        }
    };

    mask_.fill(uint32{});

    run<Dense,
#if GINKGO_ENABLE_HALF
        gko::half, std::complex<gko::half>,
#endif
        float, double, std::complex<float>, std::complex<double>>(
        b, [&](auto* orig) {
            using value_type =
                typename std::decay_t<decltype(*orig)>::value_type;
            mixed_precision_dispatch_real_complex<value_type>(impl, orig, x);
        });
}


#define GKO_DECLARE_ROW_SCATTER(_type) class RowScatterer<_type>
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_ROW_SCATTER);


}  // namespace matrix
}  // namespace gko
