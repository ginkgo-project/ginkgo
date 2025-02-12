// SPDX-FileCopyrightText: 2024 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/matrix/row_scatterer.hpp"

#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/base/dispatch_helper.hpp"
#include "core/matrix/row_scatterer_kernels.hpp"

namespace gko {
namespace matrix {
namespace {


GKO_REGISTER_OPERATION(row_scatter, row_scatter::row_scatter);


}


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
      idxs_(exec, std::move(idxs))
{}


template <typename IndexType>
void RowScatterer<IndexType>::apply_impl(const LinOp* b, LinOp* x) const
{
    auto impl = [&](const auto* orig, auto* target) {
        auto exec = orig->get_executor();
        bool invalid_access = false;

        exec->run(make_row_scatter(
            make_temporary_clone(exec, &idxs_).get(), orig,
            make_temporary_clone(exec, target).get(), invalid_access));

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
    auto x_copy = gko::clone(x);
    run<Dense,
#if GINKGO_ENABLE_HALF
        gko::half, std::complex<gko::half>,
#endif
        float, double, std::complex<float>, std::complex<double>>(
        x, [&](auto* target) {
            using dense_type = std::decay_t<decltype(*target)>;
            as<dense_type>(x_copy)->fill(
                gko::zero<typename dense_type::value_type>());
            this->apply_impl(b, x_copy);
            target->scale(beta);
            target->add_scaled(alpha, x_copy);
        });
}


#define GKO_DECLARE_ROW_SCATTER(_type) class RowScatterer<_type>
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_ROW_SCATTER);


}  // namespace matrix
}  // namespace gko
