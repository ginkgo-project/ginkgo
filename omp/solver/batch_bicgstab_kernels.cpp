/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include "core/solver/batch_bicgstab_kernels.hpp"


#include "core/solver/batch_dispatch.hpp"


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The batch Bicgstab solver namespace.
 *
 * @ingroup batch_bicgstab
 */
namespace batch_bicgstab {


namespace {


constexpr int max_num_rhs = 1;


#include "reference/base/batch_multi_vector_kernels.hpp.inc"
#include "reference/matrix/batch_dense_kernels.hpp.inc"
#include "reference/matrix/batch_ell_kernels.hpp.inc"
#include "reference/solver/batch_bicgstab_kernels.hpp.inc"


}  // unnamed namespace


template <typename T>
using settings = gko::kernels::batch_bicgstab::settings<T>;


template <typename ValueType>
class kernel_caller {
public:
    kernel_caller(std::shared_ptr<const DefaultExecutor> exec,
                  const settings<remove_complex<ValueType>> settings)
        : exec_{std::move(exec)}, settings_{settings}
    {}

    template <typename BatchMatrixType, typename PrecondType, typename StopType,
              typename LogType>
    void call_kernel(
        const LogType& logger, const BatchMatrixType& mat, PrecondType precond,
        const gko::batch::multi_vector::uniform_batch<const ValueType>& b,
        const gko::batch::multi_vector::uniform_batch<ValueType>& x) const
    {
        using real_type = typename gko::remove_complex<ValueType>;
        const size_type num_batch_items = mat.num_batch_items;
        const auto num_rows = mat.num_rows;
        const auto num_rhs = b.num_rhs;
        if (num_rhs > 1) {
            GKO_NOT_IMPLEMENTED;
        }

        const int local_size_bytes =
            gko::kernels::batch_bicgstab::local_memory_requirement<ValueType>(
                num_rows, num_rhs) +
            PrecondType::dynamic_work_size(num_rows,
                                           mat.get_single_item_num_nnz()) *
                sizeof(ValueType);

#pragma omp parallel for
        for (size_type batch_id = 0; batch_id < num_batch_items; batch_id++) {
            // TODO: Align to cache line boundary
            // TODO: Allocate and free once per thread rather than once per
            // work-item.
            auto local_space = array<unsigned char>(exec_, local_size_bytes);
            batch_entry_bicgstab_impl<StopType, PrecondType, LogType,
                                      BatchMatrixType, ValueType>(
                settings_, logger, precond, mat, b, x, batch_id,
                local_space.get_data());
        }
    }

private:
    const std::shared_ptr<const DefaultExecutor> exec_;
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
    auto dispatcher = batch::solver::create_dispatcher<ValueType>(
        kernel_caller<ValueType>(exec, settings), settings, mat, precond);
    dispatcher.apply(b, x, logdata);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_BICGSTAB_APPLY_KERNEL);


}  // namespace batch_bicgstab
}  // namespace omp
}  // namespace kernels
}  // namespace gko
