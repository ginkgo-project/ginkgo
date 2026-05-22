// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_DPCPP_SOLVER_COMMON_TRS_KERNELS_HPP_
#define GKO_DPCPP_SOLVER_COMMON_TRS_KERNELS_HPP_


#include <sycl/sycl.hpp>

#include <ginkgo/core/base/math.hpp>

#include "dpcpp/base/onemkl_bindings.hpp"
#include "dpcpp/base/types.hpp"


namespace gko {
namespace solver {


struct SolveStruct {
    virtual ~SolveStruct() = default;
};


}  // namespace solver


namespace kernels {
namespace dpcpp {
namespace {


template <typename ValueType, typename IndexType>
struct OneMklSolveStruct : gko::solver::SolveStruct {
    std::shared_ptr<const gko::DpcppExecutor> exec;
    oneapi::mkl::sparse::matrix_handle_t mat_handle;
    size_type num_rhs;
    oneapi::mkl::uplo uplo;
    oneapi::mkl::diag diag;

    OneMklSolveStruct(std::shared_ptr<const gko::DpcppExecutor> exec,
                      const matrix::Csr<ValueType, IndexType>* matrix,
                      size_type num_rhs, bool is_upper, bool unit_diag)
        : exec{exec}, num_rhs{num_rhs}
    {
        if (num_rhs == 0) {
            return;
        }
        uplo = is_upper ? oneapi::mkl::uplo::upper : oneapi::mkl::uplo::lower;
        diag = unit_diag ? oneapi::mkl::diag::unit : oneapi::mkl::diag::nonunit;
        oneapi::mkl::sparse::init_matrix_handle(&mat_handle);
        oneapi::mkl::sparse::set_csr_data(
            *exec->get_queue(), mat_handle, IndexType(matrix->get_size()[0]),
            IndexType(matrix->get_size()[1]), oneapi::mkl::index_base::zero,
            const_cast<IndexType*>(matrix->get_const_row_ptrs()),
            const_cast<IndexType*>(matrix->get_const_col_idxs()),
            const_cast<ValueType*>(matrix->get_const_values()));

        oneapi::mkl::sparse::optimize_trsm(
            *exec->get_queue(), oneapi::mkl::layout::row_major, uplo,
            oneapi::mkl::transpose::nontrans, diag, mat_handle,
            static_cast<int64>(num_rhs));
    }

    void solve(const matrix::Csr<ValueType, IndexType>* matrix,
               matrix::view::dense<const ValueType> input,
               matrix::view::dense<ValueType> output) const
    {
        if (input.size[1] != num_rhs) {
            throw gko::ValueMismatch{
                __FILE__,
                __LINE__,
                __FUNCTION__,
                input.size[1],
                num_rhs,
                "the dimensions of the multivector do not match the value "
                "provided at generation time. Check the value specified in "
                ".with_num_rhs(...)."};
        }
        oneapi::mkl::sparse::trsm(
            *exec->get_queue(), oneapi::mkl::layout::row_major,
            oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
            uplo, diag, one<ValueType>(), mat_handle, input.values,
            static_cast<int64>(num_rhs), input.stride, output.values,
            output.stride);
    }

    ~OneMklSolveStruct()
    {
        if (mat_handle) {
            oneapi::mkl::sparse::release_matrix_handle(*exec->get_queue(),
                                                       &mat_handle);
        }
    }

    OneMklSolveStruct(const OneMklSolveStruct&) = delete;

    OneMklSolveStruct(OneMklSolveStruct&&) = delete;

    OneMklSolveStruct& operator=(const OneMklSolveStruct&) = delete;

    OneMklSolveStruct& operator=(OneMklSolveStruct&&) = delete;
};


template <typename ValueType, typename IndexType>
void generate_kernel(std::shared_ptr<const DpcppExecutor> exec,
                     const matrix::Csr<ValueType, IndexType>* matrix,
                     std::shared_ptr<solver::SolveStruct>& solve_struct,
                     const gko::size_type num_rhs, bool is_upper,
                     bool unit_diag)
{
    if (matrix->get_size()[0] == 0) {
        return;
    }
    if constexpr (onemkl::is_supported<ValueType>::value) {
        solve_struct =
            std::make_shared<OneMklSolveStruct<ValueType, IndexType>>(
                exec, matrix, num_rhs, is_upper, unit_diag);
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}


template <typename ValueType, typename IndexType>
void solve_kernel(std::shared_ptr<const DpcppExecutor> exec,
                  const matrix::Csr<ValueType, IndexType>* matrix,
                  const solver::SolveStruct* solve_struct,
                  matrix::view::dense<const ValueType> b,
                  matrix::view::dense<ValueType> x)
{
    if (matrix->get_size()[0] == 0 || b.size[1] == 0) {
        return;
    }
    using vec = matrix::Dense<ValueType>;

    if constexpr (onemkl::is_supported<ValueType>::value) {
        if (auto onemkl_solve_struct =
                dynamic_cast<const OneMklSolveStruct<ValueType, IndexType>*>(
                    solve_struct)) {
            onemkl_solve_struct->solve(matrix, b, x);
        } else {
            GKO_NOT_SUPPORTED(solve_struct);
        }
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}


}  // namespace
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko


#endif  // GKO_DPCPP_SOLVER_COMMON_TRS_KERNELS_HPP_
