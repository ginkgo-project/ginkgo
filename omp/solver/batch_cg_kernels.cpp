// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/batch_cg_kernels.hpp"

#include <omp.h>

#include <ginkgo/core/base/array.hpp>

#include "core/base/batch_instantiation.hpp"
#include "core/solver/batch_dispatch.hpp"
#include "reference/base/batch_multi_vector_kernels.hpp"
#include "reference/matrix/batch_csr_kernels.hpp"
#include "reference/matrix/batch_dense_kernels.hpp"
#include "reference/matrix/batch_ell_kernels.hpp"
#include "reference/solver/batch_cg_kernels.hpp"


namespace gko {
namespace kernels {
namespace omp {
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
            gko::kernels::batch_cg::local_memory_requirement<ValueType>(
                num_rows, num_rhs) +
            PrecondType::dynamic_work_size(num_rows,
                                           mat.get_single_item_num_nnz());
        int max_threads = omp_get_max_threads();
        auto local_space =
            array<unsigned char>(exec_, local_size_bytes * max_threads);
#pragma omp parallel for
        for (size_type batch_id = 0; batch_id < num_batch_items; batch_id++) {
            auto thread_local_space = gko::make_array_view(
                exec_, local_size_bytes,
                local_space.get_data() +
                    omp_get_thread_num() * local_size_bytes);
            batch_single_kernels::batch_entry_cg_impl<
                StopType, PrecondType, LogType, BatchMatrixType, ValueType>(
                settings_, logger, precond, mat, b, x, batch_id,
                thread_local_space.get_data());
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

GKO_INSTANTIATE_FOR_BATCH_VALUE_MATRIX_PRECONDITIONER(
    GKO_DECLARE_BATCH_CG_APPLY_KERNEL_WRAPPER);


}  // namespace batch_cg
}  // namespace omp
}  // namespace kernels
}  // namespace gko
