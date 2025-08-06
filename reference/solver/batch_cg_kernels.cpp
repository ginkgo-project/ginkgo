// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/batch_cg_kernels.hpp"

#include "core/base/batch_instantiation.hpp"
#include "core/solver/batch_dispatch.hpp"
#include "reference/base/batch_multi_vector_kernels.hpp"
#include "reference/matrix/batch_csr_kernels.hpp"
#include "reference/matrix/batch_dense_kernels.hpp"
#include "reference/matrix/batch_ell_kernels.hpp"
#include "reference/solver/batch_cg_kernels.hpp"

namespace gko {
namespace kernels {
namespace reference {
namespace batch_cg {
namespace {


constexpr int max_num_rhs = 1;


}  // unnamed namespace


template <typename T>
using settings = gko::kernels::batch_cg::settings<T>;


template <typename ValueType>
class kernel_caller {
public:
    kernel_caller(std::shared_ptr<const DefaultExecutor> exec,
                  const settings<remove_complex<ValueType>> settings)
        : exec_{std::move(exec)}, settings_{settings}
    {}

    template <typename BatchMatrixEntry, typename PrecEntry, typename StopType,
              typename LogType>
    void call_kernel(
        const LogType& logger, const BatchMatrixEntry& mat, PrecEntry prec,
        const gko::batch::multi_vector::uniform_batch<const ValueType>& b,
        const gko::batch::multi_vector::uniform_batch<ValueType>& x) const
    {
        using real_type = typename gko::remove_complex<ValueType>;
        const size_type num_batch_items = mat.num_batch_items;
        const auto num_rows = mat.num_rows;
        const auto num_rhs = b.num_rhs;
        if (num_rhs > max_num_rhs) {
            GKO_NOT_IMPLEMENTED;
        }

        const size_type local_size_bytes =
            gko::kernels::batch_cg::local_memory_requirement<ValueType>(
                num_rows, num_rhs) +
            PrecEntry::dynamic_work_size(num_rows,
                                         mat.get_single_item_num_nnz());
        array<unsigned char> local_space(exec_, local_size_bytes);

        for (size_type batch_id = 0; batch_id < num_batch_items; batch_id++) {
            batch_single_kernels::batch_entry_cg_impl<
                StopType, PrecEntry, LogType, BatchMatrixEntry, ValueType>(
                settings_, logger, prec, mat, b, x, batch_id,
                local_space.get_data());
        }
    }

private:
    const std::shared_ptr<const DefaultExecutor> exec_;
    const settings<remove_complex<ValueType>> settings_;
};


template <typename ValueType, typename BatchMatrixType, typename PrecType>
void apply(std::shared_ptr<const DefaultExecutor> exec,
           const settings<remove_complex<ValueType>>& settings,
           const BatchMatrixType* mat, const PrecType* precond,
           const batch::MultiVector<ValueType>* b,
           batch::MultiVector<ValueType>* x,
           batch::log::detail::log_data<remove_complex<ValueType>>& logdata)
{
    auto dispatcher = batch::solver::create_dispatcher<ValueType>(
        kernel_caller<ValueType>(exec, settings), settings, mat, precond);
    dispatcher.apply(b, x, logdata);
}

GKO_INSTANTIATE_FOR_BATCH_VALUE_MATRIX_PRECONDITIONER_BASE(
    GKO_DECLARE_BATCH_CG_APPLY_KERNEL_WRAPPER);


}  // namespace batch_cg
}  // namespace reference
}  // namespace kernels
}  // namespace gko
