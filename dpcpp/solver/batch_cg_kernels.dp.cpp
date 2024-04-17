// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/batch_cg_kernels.hpp"


#include <CL/sycl.hpp>


#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/matrix/batch_ell.hpp>
#include <ginkgo/core/solver/batch_cg.hpp>


#include "core/base/batch_struct.hpp"
#include "core/matrix/batch_struct.hpp"
#include "core/solver/batch_dispatch.hpp"
#include "dpcpp/base/batch_struct.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/dpct.hpp"
#include "dpcpp/base/helper.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"
#include "dpcpp/components/intrinsics.dp.hpp"
#include "dpcpp/components/reduction.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"
#include "dpcpp/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The batch Cg solver namespace.
 *
 * @ingroup batch_cg
 */
namespace batch_cg {


#include "dpcpp/base/batch_multi_vector_kernels.hpp.inc"
#include "dpcpp/matrix/batch_csr_kernels.hpp.inc"
#include "dpcpp/matrix/batch_dense_kernels.hpp.inc"
#include "dpcpp/matrix/batch_ell_kernels.hpp.inc"


template <typename T>
using settings = gko::kernels::batch_cg::settings<T>;


template <typename ValueType>
class KernelCaller {
public:
    KernelCaller(std::shared_ptr<const DefaultExecutor> exec,
                 const settings<remove_complex<ValueType>> settings)
        : exec_{std::move(exec)}, settings_{settings}
    {}

    template <typename StopType, const int subgroup_size,
              const int n_shared_total, typename PrecType, typename LogType,
              typename BatchMatrixType>
    __dpct_inline__ void launch_apply_kernel(
        const gko::kernels::batch_cg::storage_config& sconf, LogType& logger,
        PrecType& prec, const BatchMatrixType mat,
        const ValueType* const __restrict__ b_values,
        ValueType* const __restrict__ x_values,
        ValueType* const __restrict__ workspace, const int& group_size,
        const int& shared_size) const
    {
        GKO_NOT_IMPLEMENTED;
    }

    template <typename BatchMatrixType, typename PrecType, typename StopType,
              typename LogType>
    void call_kernel(
        LogType logger, const BatchMatrixType& mat, PrecType prec,
        const gko::batch::multi_vector::uniform_batch<const ValueType>& b,
        const gko::batch::multi_vector::uniform_batch<ValueType>& x) const
    {
        GKO_NOT_IMPLEMENTED;
    }

private:
    std::shared_ptr<const DefaultExecutor> exec_;
    const settings<remove_complex<ValueType>> settings_;
};


template <typename ValueType>
void apply(std::shared_ptr<const DefaultExecutor> exec,
           const settings<remove_complex<ValueType>>& settings,
           const batch::BatchLinOp* const mat,
           const batch::BatchLinOp* const precond,
           const batch::MultiVector<ValueType>* const b,
           batch::MultiVector<ValueType>* const x,
           batch::log::detail::log_data<remove_complex<ValueType>>& logdata)
{
    GKO_NOT_IMPLEMENTED;
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_CG_APPLY_KERNEL);


}  // namespace batch_cg
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
