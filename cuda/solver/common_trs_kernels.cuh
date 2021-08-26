/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#ifndef GKO_CUDA_SOLVER_COMMON_TRS_KERNELS_CUH_
#define GKO_CUDA_SOLVER_COMMON_TRS_KERNELS_CUH_


#include <functional>
#include <iostream>
#include <memory>


#include <cuda.h>
#include <cusparse.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>


#include "core/matrix/dense_kernels.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "cuda/base/cusparse_bindings.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/pointer_mode_guard.hpp"
#include "cuda/base/types.hpp"


namespace gko {
namespace solver {


struct SolveStruct {
    virtual ~SolveStruct() = default;
};


}  // namespace solver


namespace kernels {
namespace cuda {
namespace {


#if (defined(CUDA_VERSION) && (CUDA_VERSION >= 11030))


template <typename ValueType, typename IndexType>
struct CudaSolveStruct : gko::solver::SolveStruct {
    cusparseHandle_t handle;
    cusparseSpSMDescr_t spsm_descr;
    cusparseSpMatDescr_t descr_a;
    Array<char> work;

    CudaSolveStruct(std::shared_ptr<const gko::CudaExecutor> exec,
                    const matrix::Csr<ValueType, IndexType>* matrix,
                    size_type num_rhs, bool is_upper)
        : handle{exec->get_cusparse_handle()},
          spsm_descr{},
          descr_a{},
          work{exec}
    {
        cusparse::pointer_mode_guard pm_guard(handle);
        spsm_descr = cusparse::create_spsm_descr();
        descr_a = cusparse::create_csr(
            matrix->get_size()[0], matrix->get_size()[1],
            matrix->get_num_stored_elements(),
            const_cast<IndexType*>(matrix->get_const_row_ptrs()),
            const_cast<IndexType*>(matrix->get_const_col_idxs()),
            const_cast<ValueType*>(matrix->get_const_values()));
        cusparse::set_attribute<cusparseFillMode_t>(
            descr_a, CUSPARSE_SPMAT_FILL_MODE,
            is_upper ? CUSPARSE_FILL_MODE_UPPER : CUSPARSE_FILL_MODE_LOWER);
        cusparse::set_attribute<cusparseDiagType_t>(
            descr_a, CUSPARSE_SPMAT_DIAG_TYPE, CUSPARSE_DIAG_TYPE_NON_UNIT);

        const auto rows = matrix->get_size()[0];
        // workaround suggested by NVIDIA engineers: for some reason
        // cusparse needs non-nullptr input vectors even for analysis
        auto descr_b = cusparse::create_dnmat(
            matrix->get_size()[0], num_rhs, matrix->get_size()[1],
            reinterpret_cast<ValueType*>(0xFF));
        auto descr_c = cusparse::create_dnmat(
            matrix->get_size()[0], num_rhs, matrix->get_size()[1],
            reinterpret_cast<ValueType*>(0xFF));

        auto work_size = cusparse::spsm_buffer_size(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE, one<ValueType>(), descr_a,
            descr_b, descr_c, CUSPARSE_SPSM_ALG_DEFAULT, spsm_descr);

        work.resize_and_reset(work_size);

        cusparse::spsm_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                one<ValueType>(), descr_a, descr_b, descr_c,
                                CUSPARSE_SPSM_ALG_DEFAULT, spsm_descr,
                                work.get_data());

        cusparse::destroy(descr_b);
        cusparse::destroy(descr_c);
    }

    void solve(const matrix::Csr<ValueType, IndexType>*,
               const matrix::Dense<ValueType>* input,
               matrix::Dense<ValueType>* output, matrix::Dense<ValueType>*,
               matrix::Dense<ValueType>*) const
    {
        cusparse::pointer_mode_guard pm_guard(handle);
        auto descr_b = cusparse::create_dnmat(
            input->get_size()[0], input->get_size()[1], input->get_stride(),
            const_cast<ValueType*>(input->get_const_values()));
        auto descr_c =
            cusparse::create_dnmat(output->get_size()[0], output->get_size()[1],
                                   output->get_stride(), output->get_values());

        cusparse::spsm_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                             CUSPARSE_OPERATION_NON_TRANSPOSE, one<ValueType>(),
                             descr_a, descr_b, descr_c,
                             CUSPARSE_SPSM_ALG_DEFAULT, spsm_descr);

        cusparse::destroy(descr_b);
        cusparse::destroy(descr_c);
    }

    ~CudaSolveStruct()
    {
        if (descr_a) {
            cusparse::destroy(descr_a);
            descr_a = nullptr;
        }
        if (spsm_descr) {
            cusparse::destroy(spsm_descr);
            spsm_descr = nullptr;
        }
    }

    CudaSolveStruct(const SolveStruct&) = delete;

    CudaSolveStruct(SolveStruct&&) = delete;

    CudaSolveStruct& operator=(const SolveStruct&) = delete;

    CudaSolveStruct& operator=(SolveStruct&&) = delete;
};


#elif (defined(CUDA_VERSION) && (CUDA_VERSION >= 9020))

template <typename ValueType, typename IndexType>
struct CudaSolveStruct : gko::solver::SolveStruct {
    std::shared_ptr<const gko::CudaExecutor> exec;
    cusparseHandle_t handle;
    int algorithm;
    csrsm2Info_t solve_info;
    cusparseSolvePolicy_t policy;
    cusparseMatDescr_t factor_descr;
    mutable Array<char> work;

    CudaSolveStruct(std::shared_ptr<const gko::CudaExecutor> exec,
                    const matrix::Csr<ValueType, IndexType>* matrix,
                    size_type num_rhs, bool is_upper)
        : exec{exec},
          handle{exec->get_cusparse_handle()},
          algorithm{},
          solve_info{},
          policy{},
          factor_descr{},
          work{exec}
    {
        cusparse::pointer_mode_guard pm_guard(handle);
        factor_descr = cusparse::create_mat_descr();
        solve_info = cusparse::create_solve_info();
        cusparse::set_mat_fill_mode(
            factor_descr,
            is_upper ? CUSPARSE_FILL_MODE_UPPER : CUSPARSE_FILL_MODE_LOWER);
        algorithm = 0;
        policy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;

        size_type work_size{};

        cusparse::buffer_size_ext(
            handle, algorithm, CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_TRANSPOSE, matrix->get_size()[0], num_rhs,
            matrix->get_num_stored_elements(), one<ValueType>(), factor_descr,
            matrix->get_const_values(), matrix->get_const_row_ptrs(),
            matrix->get_const_col_idxs(), nullptr, num_rhs, solve_info, policy,
            &work_size);

        // allocate workspace
        work.resize_and_reset(work_size);

        cusparse::csrsm2_analysis(
            handle, algorithm, CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_TRANSPOSE, matrix->get_size()[0], num_rhs,
            matrix->get_num_stored_elements(), one<ValueType>(), factor_descr,
            matrix->get_const_values(), matrix->get_const_row_ptrs(),
            matrix->get_const_col_idxs(), nullptr, num_rhs, solve_info, policy,
            work.get_data());
    }

    void solve(const matrix::Csr<ValueType, IndexType>* matrix,
               const matrix::Dense<ValueType>* input,
               matrix::Dense<ValueType>* output, matrix::Dense<ValueType>*,
               matrix::Dense<ValueType>*) const
    {
        cusparse::pointer_mode_guard pm_guard(handle);
        dense::copy(exec, input, output);
        cusparse::csrsm2_solve(
            handle, algorithm, CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_TRANSPOSE, matrix->get_size()[0],
            output->get_stride(), matrix->get_num_stored_elements(),
            one<ValueType>(), factor_descr, matrix->get_const_values(),
            matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
            output->get_values(), output->get_stride(), solve_info, policy,
            work.get_data());
    }

    ~CudaSolveStruct()
    {
        if (factor_descr) {
            cusparse::destroy(factor_descr);
            factor_descr = nullptr;
        }
        if (solve_info) {
            cusparse::destroy(solve_info);
            solve_info = nullptr;
        }
    }

    CudaSolveStruct(const CudaSolveStruct&) = delete;

    CudaSolveStruct(CudaSolveStruct&&) = delete;

    CudaSolveStruct& operator=(const CudaSolveStruct&) = delete;

    CudaSolveStruct& operator=(CudaSolveStruct&&) = delete;
};


#endif


void should_perform_transpose_kernel(std::shared_ptr<const CudaExecutor> exec,
                                     bool& do_transpose)
{
    do_transpose = false;
}


template <typename ValueType, typename IndexType>
void generate_kernel(std::shared_ptr<const CudaExecutor> exec,
                     const matrix::Csr<ValueType, IndexType>* matrix,
                     std::shared_ptr<solver::SolveStruct>& solve_struct,
                     const gko::size_type num_rhs, bool is_upper)
{
    if (cusparse::is_supported<ValueType, IndexType>::value) {
        solve_struct = std::make_shared<CudaSolveStruct<ValueType, IndexType>>(
            exec, matrix, num_rhs, is_upper);
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}


template <typename ValueType, typename IndexType>
void solve_kernel(std::shared_ptr<const CudaExecutor> exec,
                  const matrix::Csr<ValueType, IndexType>* matrix,
                  const solver::SolveStruct* solve_struct,
                  matrix::Dense<ValueType>* trans_b,
                  matrix::Dense<ValueType>* trans_x,
                  const matrix::Dense<ValueType>* b,
                  matrix::Dense<ValueType>* x)
{
    using vec = matrix::Dense<ValueType>;

    if (cusparse::is_supported<ValueType, IndexType>::value) {
        if (auto cuda_solve_struct =
                dynamic_cast<const CudaSolveStruct<ValueType, IndexType>*>(
                    solve_struct)) {
            cuda_solve_struct->solve(matrix, b, x, trans_b, trans_x);
        } else {
            GKO_NOT_SUPPORTED(solve_struct);
        }
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}


}  // namespace
}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CUDA_SOLVER_COMMON_TRS_KERNELS_CUH_
