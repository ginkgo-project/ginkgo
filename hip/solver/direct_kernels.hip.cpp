// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/direct_kernels.hpp"


#if GKO_HAVE_CUDSS


#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {
namespace kernels {
namespace hip {
namespace direct {


template <typename ValueType, typename IndexType>
void generate(std::shared_ptr<const HipExecutor> exec,
              const matrix::Csr<ValueType, IndexType>*,
              std::shared_ptr<experimental::solver::direct_vendor_state>&,
              const experimental::solver::vendor_parameters&)
{
    GKO_NOT_SUPPORTED(exec);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DIRECT_GENERATE_KERNEL);


template <typename ValueType>
void solve(std::shared_ptr<const HipExecutor> exec,
           const experimental::solver::direct_vendor_state*,
           const matrix::Dense<ValueType>*, matrix::Dense<ValueType>*)
{
    GKO_NOT_SUPPORTED(exec);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DIRECT_SOLVE_KERNEL);


}  // namespace direct
}  // namespace hip
}  // namespace kernels
}  // namespace gko


#endif  // GKO_HAVE_CUDSS
