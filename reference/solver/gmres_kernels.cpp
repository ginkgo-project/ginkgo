// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/gmres_kernels.hpp"

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The GMRES solver namespace.
 *
 * @ingroup gmres
 */
namespace gmres {


template <typename ValueType>
void restart(std::shared_ptr<const ReferenceExecutor> exec,
             matrix::view::dense<const ValueType> residual,
             matrix::view::dense<const remove_complex<ValueType>> residual_norm,
             matrix::view::dense<ValueType> residual_norm_collection,
             matrix::view::dense<ValueType> krylov_bases,
             size_type* final_iter_nums)
{
    for (size_type j = 0; j < residual.size[1]; ++j) {
        residual_norm_collection(0, j) = residual_norm(0, j);
        for (size_type i = 0; i < residual.size[0]; ++i) {
            krylov_bases(i, j) = residual(i, j) / residual_norm(0, j);
        }
        final_iter_nums[j] = 0;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_RESTART_KERNEL);


template <typename ValueType>
void multi_axpy(std::shared_ptr<const ReferenceExecutor> exec,
                matrix::view::dense<const ValueType> krylov_bases,
                matrix::view::dense<const ValueType> y,
                matrix::view::dense<ValueType> before_preconditioner,
                const size_type* final_iter_nums, stopping_status* stop_status)
{
    const auto krylov_bases_rowoffset = before_preconditioner.size[0];
    for (size_type k = 0; k < before_preconditioner.size[1]; ++k) {
        if (stop_status[k].is_finalized()) {
            continue;
        }
        for (size_type i = 0; i < before_preconditioner.size[0]; ++i) {
            before_preconditioner(i, k) = zero<ValueType>();
            for (size_type j = 0; j < final_iter_nums[k]; ++j) {
                before_preconditioner(i, k) +=
                    krylov_bases(i + j * krylov_bases_rowoffset, k) * y(j, k);
            }
        }
        if (stop_status[k].has_stopped()) {
            stop_status[k].finalize();
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_MULTI_AXPY_KERNEL);

template <typename ValueType>
void multi_dot(std::shared_ptr<const ReferenceExecutor> exec,
               matrix::view::dense<const ValueType> krylov_bases,
               matrix::view::dense<const ValueType> next_krylov,
               matrix::view::dense<ValueType> hessenberg_col)
{
    auto num_rhs = next_krylov.size[1];
    auto krylov_bases_rowoffset = next_krylov.size[0];
    for (size_type i = 0; i < hessenberg_col.size[0] - 1; ++i) {
        for (size_type k = 0; k < num_rhs; ++k) {
            hessenberg_col(i, k) = zero<ValueType>();
            for (size_type j = 0; j < krylov_bases_rowoffset; ++j) {
                hessenberg_col(i, k) +=
                    conj(krylov_bases(i * krylov_bases_rowoffset + j, k)) *
                    next_krylov(j, k);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_MULTI_DOT_KERNEL);

}  // namespace gmres
}  // namespace reference
}  // namespace kernels
}  // namespace gko
