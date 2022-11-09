/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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


#include <functional>
#include <iostream>
#include <memory>


#include <cuda.h>
#include <cusparse.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/solver/triangular.hpp>


#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "cuda/base/cusparse_bindings.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/pointer_mode_guard.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/atomic.cuh"
#include "cuda/components/thread_ids.cuh"
#include "cuda/components/uninitialized_array.hpp"
#include "cuda/components/volatile.cuh"


namespace gko {
namespace solver {


// struct SolveStruct {
//     virtual ~SolveStruct() = default;
// };


}  // namespace solver


namespace kernels {
namespace cuda {
namespace {


#if (defined(CUDA_VERSION) && (CUDA_VERSION >= 11031))


template <typename ValueType, typename IndexType>
struct CudaSolveStruct : gko::solver::SolveStruct {
    cusparseHandle_t handle;
    cusparseSpSMDescr_t spsm_descr;
    cusparseSpMatDescr_t descr_a;
    size_type num_rhs;

    // Implicit parameter in spsm_solve, therefore stored here.
    array<char> work;

    CudaSolveStruct(std::shared_ptr<const gko::CudaExecutor> exec,
                    const matrix::Csr<ValueType, IndexType>* matrix,
                    size_type num_rhs, bool is_upper, bool unit_diag)
        : handle{exec->get_cusparse_handle()},
          spsm_descr{},
          descr_a{},
          num_rhs{num_rhs},
          work{exec}
    {
        if (num_rhs == 0) {
            return;
        }
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
            descr_a, CUSPARSE_SPMAT_DIAG_TYPE,
            unit_diag ? CUSPARSE_DIAG_TYPE_UNIT : CUSPARSE_DIAG_TYPE_NON_UNIT);

        const auto rows = matrix->get_size()[0];
        // workaround suggested by NVIDIA engineers: for some reason
        // cusparse needs non-nullptr input vectors even for analysis
        auto descr_b = cusparse::create_dnmat(
            dim<2>{matrix->get_size()[0], num_rhs}, matrix->get_size()[1],
            reinterpret_cast<ValueType*>(0xDEAD));
        auto descr_c = cusparse::create_dnmat(
            dim<2>{matrix->get_size()[0], num_rhs}, matrix->get_size()[1],
            reinterpret_cast<ValueType*>(0xDEAF));

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
        if (input->get_size()[1] != num_rhs) {
            throw gko::ValueMismatch{
                __FILE__,
                __LINE__,
                __FUNCTION__,
                input->get_size()[1],
                num_rhs,
                "the dimensions of the multivector do not match the value "
                "provided at generation time. Check the value specified in "
                ".with_num_rhs(...)."};
        }
        cusparse::pointer_mode_guard pm_guard(handle);
        auto descr_b = cusparse::create_dnmat(
            input->get_size(), input->get_stride(),
            const_cast<ValueType*>(input->get_const_values()));
        auto descr_c = cusparse::create_dnmat(
            output->get_size(), output->get_stride(), output->get_values());

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
    size_type num_rhs;
    mutable array<char> work;

    CudaSolveStruct(std::shared_ptr<const gko::CudaExecutor> exec,
                    const matrix::Csr<ValueType, IndexType>* matrix,
                    size_type num_rhs, bool is_upper, bool unit_diag)
        : exec{exec},
          handle{exec->get_cusparse_handle()},
          algorithm{},
          solve_info{},
          policy{},
          factor_descr{},
          num_rhs{num_rhs},
          work{exec}
    {
        if (num_rhs == 0) {
            return;
        }
        cusparse::pointer_mode_guard pm_guard(handle);
        factor_descr = cusparse::create_mat_descr();
        solve_info = cusparse::create_solve_info();
        cusparse::set_mat_fill_mode(
            factor_descr,
            is_upper ? CUSPARSE_FILL_MODE_UPPER : CUSPARSE_FILL_MODE_LOWER);
        cusparse::set_mat_diag_type(
            factor_descr,
            unit_diag ? CUSPARSE_DIAG_TYPE_UNIT : CUSPARSE_DIAG_TYPE_NON_UNIT);
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
        if (input->get_size()[1] != num_rhs) {
            throw gko::ValueMismatch{
                __FILE__,
                __LINE__,
                __FUNCTION__,
                input->get_size()[1],
                num_rhs,
                "the dimensions of the multivector do not match the value "
                "provided at generation time. Check the value specified in "
                ".with_num_rhs(...)."};
        }
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


constexpr int default_block_size = 512;
constexpr int fallback_block_size = 32;


template <bool is_upper, typename ValueType, typename IndexType>
__global__ void sptrsv_naive_caching_kernel(
    const IndexType* const rowptrs, const IndexType* const colidxs,
    const ValueType* const vals, const ValueType* const b, size_type b_stride,
    ValueType* const x, size_type x_stride, const size_type n,
    const size_type nrhs, bool unit_diag, bool* nan_produced,
    IndexType* atomic_counter)
{
    __shared__ uninitialized_array<ValueType, default_block_size> x_s_array;
    __shared__ IndexType block_base_idx;

    if (threadIdx.x == 0) {
        block_base_idx =
            atomic_add(atomic_counter, IndexType{1}) * default_block_size;
    }
    __syncthreads();
    const auto full_gid = static_cast<IndexType>(threadIdx.x) + block_base_idx;
    const auto rhs = full_gid % nrhs;
    const auto gid = full_gid / nrhs;
    const auto row = is_upper ? n - 1 - gid : gid;

    if (gid >= n) {
        return;
    }

    const auto self_shmem_id = full_gid / default_block_size;
    const auto self_shid = full_gid % default_block_size;

    ValueType* x_s = x_s_array;
    x_s[self_shid] = nan<ValueType>();

    __syncthreads();

    // lower tri matrix: start at beginning, run forward until last entry,
    // (row_end - 1) which is the diagonal entry
    // upper tri matrix: start at last entry (row_end - 1), run backward
    // until first entry, which is the diagonal entry
    const auto row_begin = is_upper ? rowptrs[row + 1] - 1 : rowptrs[row];
    const auto row_end = is_upper ? rowptrs[row] - 1 : rowptrs[row + 1];
    const int row_step = is_upper ? -1 : 1;

    auto sum = zero<ValueType>();
    auto i = row_begin;
    for (; i != row_end; i += row_step) {
        const auto dependency = colidxs[i];
        if (is_upper ? dependency <= row : dependency >= row) {
            break;
        }
        auto x_p = &x[dependency * x_stride + rhs];

        const auto dependency_gid = is_upper ? (n - 1 - dependency) * nrhs + rhs
                                             : dependency * nrhs + rhs;
        const bool shmem_possible =
            (dependency_gid / default_block_size) == self_shmem_id;
        if (shmem_possible) {
            const auto dependency_shid = dependency_gid % default_block_size;
            x_p = &x_s[dependency_shid];
        }

        ValueType x = *x_p;
        while (is_nan(x)) {
            x = load(x_p, 0);
        }

        sum += x * vals[i];
    }

    // The first entry past the triangular part will be the diagonal
    const auto diag = unit_diag ? one<ValueType>() : vals[i];
    const auto r = (b[row * b_stride + rhs] - sum) / diag;

    store(x_s, self_shid, r);
    x[row * x_stride + rhs] = r;

    // This check to ensure no infinte loops happen.
    if (is_nan(r)) {
        store(x_s, self_shid, zero<ValueType>());
        x[row * x_stride + rhs] = zero<ValueType>();
        *nan_produced = true;
    }
}


template <bool is_upper, typename ValueType, typename IndexType>
__global__ void sptrsv_naive_legacy_kernel(
    const IndexType* const rowptrs, const IndexType* const colidxs,
    const ValueType* const vals, const ValueType* const b, size_type b_stride,
    ValueType* const x, size_type x_stride, const size_type n,
    const size_type nrhs, bool unit_diag, bool* nan_produced,
    IndexType* atomic_counter)
{
    __shared__ IndexType block_base_idx;
    if (threadIdx.x == 0) {
        block_base_idx =
            atomic_add(atomic_counter, IndexType{1}) * fallback_block_size;
    }
    __syncthreads();
    const auto full_gid = static_cast<IndexType>(threadIdx.x) + block_base_idx;
    const auto rhs = full_gid % nrhs;
    const auto gid = full_gid / nrhs;
    const auto row = is_upper ? n - 1 - gid : gid;

    if (gid >= n) {
        return;
    }

    // lower tri matrix: start at beginning, run forward
    // upper tri matrix: start at last entry (row_end - 1), run backward
    const auto row_begin = is_upper ? rowptrs[row + 1] - 1 : rowptrs[row];
    const auto row_end = is_upper ? rowptrs[row] - 1 : rowptrs[row + 1];
    const int row_step = is_upper ? -1 : 1;

    auto sum = zero<ValueType>();
    auto j = row_begin;
    auto col = colidxs[j];
    while (j != row_end) {
        auto x_val = load(x, col * x_stride + rhs);
        while (!is_nan(x_val)) {
            sum += vals[j] * x_val;
            j += row_step;
            col = colidxs[j];
            x_val = load(x, col * x_stride + rhs);
        }
        // to avoid the kernel hanging on matrices without diagonal,
        // we bail out if we are past the triangle, even if it's not
        // the diagonal entry. This may lead to incorrect results,
        // but prevents an infinite loop.
        if (is_upper ? row >= col : row <= col) {
            // assert(row == col);
            auto diag = unit_diag ? one<ValueType>() : vals[j];
            const auto r = (b[row * b_stride + rhs] - sum) / diag;
            store(x, row * x_stride + rhs, r);
            // after we encountered the diagonal, we are done
            // this also skips entries outside the triangle
            j = row_end;
            if (is_nan(r)) {
                store(x, row * x_stride + rhs, zero<ValueType>());
                *nan_produced = true;
            }
        }
    }
}


template <typename IndexType>
__global__ void sptrsv_init_kernel(bool* const nan_produced,
                                   IndexType* const atomic_counter)
{
    *nan_produced = false;
    *atomic_counter = IndexType{};
}


template <typename ValueType, typename IndexType>
struct SptrsvebcrnSolveStruct : gko::solver::SolveStruct {
    bool is_upper;
    bool unit_diag;

    SptrsvebcrnSolveStruct(std::shared_ptr<const gko::CudaExecutor>,
                           const matrix::Csr<ValueType, IndexType>*, size_type,
                           bool is_upper, bool unit_diag)
        : is_upper{is_upper}, unit_diag{unit_diag}
    {}

    void solve(std::shared_ptr<const CudaExecutor> exec,
               const matrix::Csr<ValueType, IndexType>* matrix,
               const matrix::Dense<ValueType>* b,
               matrix::Dense<ValueType>* x) const
    {
        // Pre-Volta GPUs may deadlock due to missing independent thread
        // scheduling.
        const auto is_fallback_required = exec->get_major_version() < 7;

        const auto n = matrix->get_size()[0];
        const auto nrhs = b->get_size()[1];

        // Initialize x to all NaNs.
        dense::fill(exec, x, nan<ValueType>());

        array<bool> nan_produced(exec, 1);
        array<IndexType> atomic_counter(exec, 1);
        sptrsv_init_kernel<<<1, 1>>>(nan_produced.get_data(),
                                     atomic_counter.get_data());

        const dim3 block_size(
            is_fallback_required ? fallback_block_size : default_block_size, 1,
            1);
        const dim3 grid_size(ceildiv(n * nrhs, block_size.x), 1, 1);

        if (is_fallback_required) {
            if (is_upper) {
                sptrsv_naive_legacy_kernel<true><<<grid_size, block_size>>>(
                    matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                    as_cuda_type(matrix->get_const_values()),
                    as_cuda_type(b->get_const_values()), b->get_stride(),
                    as_cuda_type(x->get_values()), x->get_stride(), n, nrhs,
                    unit_diag, nan_produced.get_data(),
                    atomic_counter.get_data());
            } else {
                sptrsv_naive_legacy_kernel<false><<<grid_size, block_size>>>(
                    matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                    as_cuda_type(matrix->get_const_values()),
                    as_cuda_type(b->get_const_values()), b->get_stride(),
                    as_cuda_type(x->get_values()), x->get_stride(), n, nrhs,
                    unit_diag, nan_produced.get_data(),
                    atomic_counter.get_data());
            }
        } else {
            if (is_upper) {
                sptrsv_naive_caching_kernel<true><<<grid_size, block_size>>>(
                    matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                    as_cuda_type(matrix->get_const_values()),
                    as_cuda_type(b->get_const_values()), b->get_stride(),
                    as_cuda_type(x->get_values()), x->get_stride(), n, nrhs,
                    unit_diag, nan_produced.get_data(),
                    atomic_counter.get_data());
            } else {
                sptrsv_naive_caching_kernel<false><<<grid_size, block_size>>>(
                    matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                    as_cuda_type(matrix->get_const_values()),
                    as_cuda_type(b->get_const_values()), b->get_stride(),
                    as_cuda_type(x->get_values()), x->get_stride(), n, nrhs,
                    unit_diag, nan_produced.get_data(),
                    atomic_counter.get_data());
            }
        }
#if GKO_VERBOSE_LEVEL >= 1
        if (exec->copy_val_to_host(nan_produced.get_const_data())) {
            std::cerr << "Error: triangular solve produced NaN, either not all "
                         "diagonal "
                         "elements are nonzero, or the system is very "
                         "ill-conditioned. "
                         "The NaN will be replaced with a zero.\n";
        }
#endif  // GKO_VERBOSE_LEVEL >= 1
    }
};


template <bool is_upper, typename ValueType, typename IndexType>
__global__ void sptrsmebrcnm_kernel(
    const IndexType* const rowptrs, const IndexType* const colidxs,
    const ValueType* const vals, const ValueType* const b, size_type b_stride,
    ValueType* const x, size_type x_stride, const size_type n,
    const IndexType nrhs, bool* nan_produced, IndexType* atomic_counter,
    IndexType m, bool unit_diag)
{
    __shared__ IndexType block_base_idx;

    if (threadIdx.x == 0) {
        block_base_idx =
            atomic_add(atomic_counter, IndexType{1}) * default_block_size;
    }
    __syncthreads();
    const auto full_gid = static_cast<IndexType>(threadIdx.x) + block_base_idx;
    const auto rhs = (full_gid / m) % nrhs;
    const auto gid = full_gid / (m * nrhs);
    const auto row = is_upper ? n - 1 - gid : gid;

    if (gid >= n || rhs >= nrhs || full_gid % m != 0) {
        return;
    }

    // lower tri matrix: start at beginning, run forward until last entry,
    // (row_end - 1) which is the diagonal entry
    // upper tri matrix: start at last entry (row_end - 1), run backward
    // until first entry, which is the diagonal entry
    const auto row_begin = is_upper ? rowptrs[row + 1] - 1 : rowptrs[row];
    const auto row_diag = is_upper ? rowptrs[row] : rowptrs[row + 1] - 1;
    const int row_step = is_upper ? -1 : 1;

    auto sum = zero<ValueType>();
    auto i = row_begin;
    for (; i != row_diag; i += row_step) {
        const auto dependency = colidxs[i];
        if (is_upper ? dependency <= row : dependency >= row) {
            break;
        }

        auto x_p = &x[dependency * x_stride + rhs];


        ValueType x = *x_p;
        while (is_nan(x)) {
            x = load(x_p, 0);
        }

        sum += x * vals[i];
    }

    const auto diag = unit_diag ? one<ValueType>() : vals[i];
    const auto r = (b[row * b_stride + rhs] - sum) / diag;
    x[row * x_stride + rhs] = r;

    // This check to ensure no infinte loops happen.
    if (is_nan(r)) {
        x[row * x_stride + rhs] = zero<ValueType>();
        *nan_produced = true;
    }
}


template <typename ValueType, typename IndexType>
struct SptrsvebcrnmSolveStruct : gko::solver::SolveStruct {
    bool is_upper;
    IndexType m;
    bool unit_diag;

    SptrsvebcrnmSolveStruct(std::shared_ptr<const gko::CudaExecutor>,
                            const matrix::Csr<ValueType, IndexType>*, size_type,
                            bool is_upper, bool unit_diag, uint8 m)
        : is_upper{is_upper}, m{m}, unit_diag{unit_diag}
    {}

    void solve(std::shared_ptr<const CudaExecutor> exec,
               const matrix::Csr<ValueType, IndexType>* matrix,
               const matrix::Dense<ValueType>* b,
               matrix::Dense<ValueType>* x) const
    {
        // Pre-Volta GPUs may deadlock due to missing independent thread
        // scheduling.
        const auto is_fallback_required = exec->get_major_version() < 7;

        const auto n = matrix->get_size()[0];
        const IndexType nrhs = b->get_size()[1];

        // Initialize x to all NaNs.
        dense::fill(exec, x, nan<ValueType>());

        array<bool> nan_produced(exec, 1);
        array<IndexType> atomic_counter(exec, 1);
        sptrsv_init_kernel<<<1, 1>>>(nan_produced.get_data(),
                                     atomic_counter.get_data());

        const dim3 block_size(
            is_fallback_required ? fallback_block_size : default_block_size, 1,
            1);
        const dim3 grid_size(
            ceildiv(n * (is_fallback_required ? 1 : m) * nrhs, block_size.x), 1,
            1);

        if (is_fallback_required) {
            if (is_upper) {
                sptrsv_naive_legacy_kernel<true><<<grid_size, block_size>>>(
                    matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                    as_cuda_type(matrix->get_const_values()),
                    as_cuda_type(b->get_const_values()), b->get_stride(),
                    as_cuda_type(x->get_values()), x->get_stride(), n, nrhs,
                    unit_diag, nan_produced.get_data(),
                    atomic_counter.get_data());
            } else {
                sptrsv_naive_legacy_kernel<false><<<grid_size, block_size>>>(
                    matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                    as_cuda_type(matrix->get_const_values()),
                    as_cuda_type(b->get_const_values()), b->get_stride(),
                    as_cuda_type(x->get_values()), x->get_stride(), n, nrhs,
                    unit_diag, nan_produced.get_data(),
                    atomic_counter.get_data());
            }
        } else {
            if (is_upper) {
                sptrsmebrcnm_kernel<true><<<grid_size, block_size>>>(
                    matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                    as_cuda_type(matrix->get_const_values()),
                    as_cuda_type(b->get_const_values()), b->get_stride(),
                    as_cuda_type(x->get_values()), x->get_stride(), n, nrhs,
                    nan_produced.get_data(), atomic_counter.get_data(), m,
                    unit_diag);
            } else {
                sptrsmebrcnm_kernel<false><<<grid_size, block_size>>>(
                    matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                    as_cuda_type(matrix->get_const_values()),
                    as_cuda_type(b->get_const_values()), b->get_stride(),
                    as_cuda_type(x->get_values()), x->get_stride(), n, nrhs,
                    nan_produced.get_data(), atomic_counter.get_data(), m,
                    unit_diag);
            }
        }
#if GKO_VERBOSE_LEVEL >= 1
        if (exec->copy_val_to_host(nan_produced.get_const_data())) {
            std::cerr << "Error: triangular solve produced NaN, either not all "
                         "diagonal "
                         "elements are nonzero, or the system is very "
                         "ill-conditioned. "
                         "The NaN will be replaced with a zero.\n";
        }
#endif  // GKO_VERBOSE_LEVEL >= 1
    }
};


template <typename ValueType, typename IndexType, bool is_upper>
__global__ void sptrsvelcr_kernel(
    const IndexType* const rowptrs, const IndexType* const colidxs,
    const ValueType* const vals, const ValueType* const b,
    const size_type b_stride, ValueType* const x, const size_type x_stride,
    const IndexType* const levels, const IndexType sweep, const IndexType n,
    const IndexType nrhs, bool unit_diag)
{
    const auto gid = thread::get_thread_id_flat<IndexType>();
    const auto row = gid / nrhs;
    const auto rhs = gid % nrhs;

    if (row >= n) {
        return;
    }

    if (levels[row] != sweep) {
        return;
    }

    const auto row_start = is_upper ? rowptrs[row + 1] - 1 : rowptrs[row];
    const auto row_end = is_upper ? rowptrs[row] : rowptrs[row + 1] - 1;
    const auto row_step = is_upper ? -1 : 1;

    auto sum = zero<ValueType>();
    IndexType i = row_start;
    for (; i != row_end; i += row_step) {
        const auto dependency = colidxs[i];
        if (is_upper ? dependency <= row : dependency >= row) {
            break;
        }

        sum += x[dependency * x_stride + rhs] * vals[i];
    }

    const auto diag = unit_diag ? one<ValueType>() : vals[i];
    const auto r = (b[row * b_stride + rhs] - sum) / diag;
    x[row * x_stride + rhs] = r;
}


template <typename IndexType, bool is_upper>
__global__ void level_generation_kernel(const IndexType* const rowptrs,
                                        const IndexType* const colidxs,
                                        volatile IndexType* const levels,
                                        volatile IndexType* const height,
                                        const IndexType n,
                                        IndexType* const atomic_counter)
{
    __shared__ uninitialized_array<IndexType, default_block_size> level_s_array;
    __shared__ IndexType block_base_idx;

    if (threadIdx.x == 0) {
        block_base_idx =
            atomic_add(atomic_counter, IndexType{1}) * default_block_size;
    }
    __syncthreads();
    const auto full_gid = static_cast<IndexType>(threadIdx.x) + block_base_idx;
    const auto gid = full_gid;
    const auto row = is_upper ? n - 1 - gid : gid;

    if (gid >= n) {
        return;
    }

    const auto self_shmem_id = full_gid / default_block_size;
    const auto self_shid = full_gid % default_block_size;

    IndexType* level_s = level_s_array;
    level_s[self_shid] = -1;

    __syncthreads();

    // lower tri matrix: start at beginning, run forward until last entry,
    // (row_end - 1) which is the diagonal entry
    // upper tri matrix: start at last entry (row_end - 1), run backward
    // until first entry, which is the diagonal entry
    const auto row_begin = is_upper ? rowptrs[row + 1] - 1 : rowptrs[row];
    const auto row_end = is_upper ? rowptrs[row] : rowptrs[row + 1] - 1;
    const int row_step = is_upper ? -1 : 1;

    IndexType level = -one<IndexType>();
    for (auto i = row_begin; i != row_end; i += row_step) {
        const auto dependency = colidxs[i];
        if (is_upper ? dependency <= row : dependency >= row) {
            break;
        }

        auto l_p = &levels[dependency];

        const auto dependency_gid = is_upper ? n - 1 - dependency : dependency;
        const bool shmem_possible =
            (dependency_gid / default_block_size) == self_shmem_id;
        if (shmem_possible) {
            const auto dependency_shid = dependency_gid % default_block_size;
            l_p = &level_s[dependency_shid];
        }

        IndexType l = *l_p;
        while (l == -one<IndexType>()) {
            l = load(l_p, 0);
        }

        level = max(l, level);
    }

    store(level_s, self_shid, level + 1);
    levels[row] = level + 1;

    atomic_max((IndexType*)height, level + 1);
}


template <typename IndexType>
__global__ void sptrsv_level_counts_kernel(
    const IndexType* const levels, volatile IndexType* const level_counts,
    IndexType* const lperm, const IndexType n)
{
    const auto gid = blockIdx.x * blockDim.x + threadIdx.x;
    const auto row = gid;

    if (row >= n) {
        return;
    }

    auto level = levels[row];

    // TODO: Make this a parallel reduction from n -> #levels
    const auto i = atomic_add((IndexType*)(level_counts + level), (IndexType)1);

    lperm[row] = i;
}


template <typename IndexType>
__global__ void sptrsv_lperm_finalize_kernel(
    const IndexType* const levels, const IndexType* const level_counts,
    IndexType* const lperm, const IndexType n)
{
    const auto gid = blockIdx.x * blockDim.x + threadIdx.x;
    const auto row = gid;

    if (row >= n) {
        return;
    }

    lperm[row] += level_counts[levels[row]];
}


template <typename ValueType, typename IndexType>
struct SptrsvlrSolveStruct : solver::SolveStruct {
    bool is_upper;
    array<IndexType> levels;
    IndexType height;
    bool unit_diag;

    SptrsvlrSolveStruct(std::shared_ptr<const gko::CudaExecutor> exec,
                        const matrix::Csr<ValueType, IndexType>* matrix,
                        size_type, bool is_upper, bool unit_diag)
        : is_upper{is_upper}, unit_diag{unit_diag}
    {
        const IndexType n = matrix->get_size()[0];
        cudaMemset(levels.get_data(), 0xFF, n * sizeof(IndexType));

        array<uint8> changed(exec, 1);
        cudaMemset(changed.get_data(), 1, sizeof(uint8));

        array<IndexType> height_d(exec, 1);
        cudaMemset(height_d.get_data(), 0, sizeof(IndexType));

        array<IndexType> atomic_counter(exec, 1);
        cudaMemset(atomic_counter.get_data(), 0, sizeof(IndexType));

        const auto block_size = default_block_size;
        const auto block_count = (n + block_size - 1) / block_size;

        if (is_upper) {
            level_generation_kernel<IndexType, true>
                <<<block_count, block_size>>>(
                    matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                    levels.get_data(), height_d.get_data(), n,
                    atomic_counter.get_data());
        } else {
            level_generation_kernel<IndexType, false>
                <<<block_count, block_size>>>(
                    matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                    levels.get_data(), height_d.get_data(), n,
                    atomic_counter.get_data());
        }

        height = exec->copy_val_to_host(height_d.get_const_data()) + 1;
    }

    void solve(std::shared_ptr<const CudaExecutor>,
               const matrix::Csr<ValueType, IndexType>* matrix,
               const matrix::Dense<ValueType>* b,
               matrix::Dense<ValueType>* x) const
    {
        const IndexType n = matrix->get_size()[0];
        const IndexType nrhs = b->get_size()[1];

        for (IndexType done_for = 0; done_for < height; ++done_for) {
            const dim3 block_size(default_block_size, 1, 1);
            const dim3 grid_size(ceildiv(n * nrhs, block_size.x), 1, 1);

            if (is_upper) {
                sptrsvelcr_kernel<decltype(as_cuda_type(ValueType{})),
                                  IndexType, true><<<grid_size, block_size>>>(
                    matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                    as_cuda_type(matrix->get_const_values()),
                    as_cuda_type(b->get_const_values()), b->get_stride(),
                    as_cuda_type(x->get_values()), x->get_stride(),
                    levels.get_const_data(), done_for, n, nrhs, unit_diag);
            } else {
                sptrsvelcr_kernel<decltype(as_cuda_type(ValueType{})),
                                  IndexType, false><<<grid_size, block_size>>>(
                    matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                    as_cuda_type(matrix->get_const_values()),
                    as_cuda_type(b->get_const_values()), b->get_stride(),
                    as_cuda_type(x->get_values()), x->get_stride(),
                    levels.get_const_data(), done_for, n, nrhs, unit_diag);
            }
        }
    }
};


// Values other than 32 don't work.
constexpr int32 warp_inverse_size = 32;


template <typename IndexType, bool is_upper>
__global__ void sptrsvebcrwi_generate_prep_kernel(
    const IndexType* const rowptrs, const IndexType* const colidxs,
    IndexType* const row_skip_counts, const size_type n)
{
    const auto gid = thread::get_thread_id_flat<IndexType>();
    const auto row = gid;

    if (row >= n) {
        return;
    }

    const auto row_start = is_upper ? rowptrs[row + 1] - 1 : rowptrs[row];
    const auto row_end =
        is_upper ? rowptrs[row] - 1 : rowptrs[row + 1];  // Includes diagonal
    const auto row_step = is_upper ? -1 : 1;
    const auto local_inv_count =
        is_upper ? warp_inverse_size - (row % warp_inverse_size) - 1
                 : row % warp_inverse_size;

    // TODO: Evaluate end-to-start iteration with early break optimization
    // Note: This optimization is only sensible when a hint
    //       "does this use compact storage" is set to false.
    // FIXME: Document a requirement of sorted indices, then
    //        break on first hit in the diagonal box, calculating
    //        the number of not-visited entries. That is more
    //        efficient for compact storage schemes.
    IndexType row_skip_count = 0;
    for (IndexType i = row_start; i != row_end; i += row_step) {
        const auto dep = colidxs[i];

        if (is_upper) {
            // Includes diagonal, entries from the other factor evaluate to
            // negative
            if (dep - row <= local_inv_count) {
                ++row_skip_count;
            }
        } else {
            if (row - dep <= local_inv_count) {
                ++row_skip_count;
            }
        }
    }

    row_skip_counts[row] = row_skip_count;
}


template <typename ValueType, typename IndexType, bool is_upper>
__global__ void sptrsvebcrwi_generate_inv_kernel(
    const IndexType* const rowptrs, const IndexType* const colidxs,
    const ValueType* const vals, IndexType* const row_skip_counts,
    ValueType* const band_inv,  // zero initialized
    uint32* const masks, const size_type n, const bool unit_diag)
{
    const auto gid = thread::get_thread_id_flat<IndexType>();
    const auto inv_block = gid / warp_inverse_size;
    const auto rhs = gid % warp_inverse_size;

    const auto local_start_row = is_upper ? warp_inverse_size - 1 : 0;
    const auto local_end_row = is_upper ? -1 : warp_inverse_size;
    const auto local_step_row = is_upper ? -1 : 1;

#pragma unroll
    for (IndexType _i = local_start_row; _i != local_end_row;
         _i += local_step_row) {
        const auto row = (gid / warp_inverse_size) * warp_inverse_size + _i;

        // Skips entries beyond matrix size, in the last/first block
        if (row >= n) {
            continue;
        }

        // Go though all block-internal dependencies of the row

        const auto row_start = is_upper
                                   ? rowptrs[row] + row_skip_counts[row] - 1
                                   : rowptrs[row + 1] - row_skip_counts[row];
        const auto row_end = is_upper ? rowptrs[row] : rowptrs[row + 1] - 1;
        const auto row_step = is_upper ? -1 : 1;

        auto sum = zero<ValueType>();
        IndexType i = row_start;
        for (; i != row_end; i += row_step) {
            const auto dep = colidxs[i];

            // To skip out-of-triangle entries for compressed storage
            if (dep == row) {
                break;
            }

            sum +=
                band_inv[inv_block * (warp_inverse_size * warp_inverse_size) +
                         dep % warp_inverse_size + rhs * warp_inverse_size] *
                vals[i];
        }

        const auto diag = unit_diag ? one<ValueType>() : vals[i];
        const auto r =
            ((rhs == _i ? one<ValueType>() : zero<ValueType>()) - sum) / diag;

        band_inv[inv_block * (warp_inverse_size * warp_inverse_size) +
                 row % warp_inverse_size + rhs * warp_inverse_size] = r;
    }


    if (gid >= n) {
        return;
    }

    const auto local_row = rhs;
    const auto row = gid;

    const auto activemask = __activemask();

    // Discover connected components.

    // Abuse masks as intermediate storage for component descriptors
    store(masks, row, local_row);
    __syncwarp(activemask);

    for (IndexType _i = 0; _i < warp_inverse_size; ++_i) {
        uint32 current_min = local_row;

        const auto h_start = is_upper ? local_row + 1 : 0;
        const auto h_end = is_upper ? warp_inverse_size : local_row;
        const auto v_start = is_upper ? 0 : local_row + 1;
        const auto v_end = is_upper ? local_row : warp_inverse_size;

        for (IndexType i = h_start; i < h_end; ++i) {
            if (band_inv[inv_block * (warp_inverse_size * warp_inverse_size) +
                         local_row + i * warp_inverse_size] != 0.0) {
                const auto load1 = load(masks, row - local_row + i);
                if (current_min > load1) {
                    current_min = load1;
                }
            }
        }
        for (IndexType i = v_start; i < v_end; ++i) {
            if (band_inv[inv_block * (warp_inverse_size * warp_inverse_size) +
                         i + local_row * warp_inverse_size] != 0.0) {
                const auto load2 = load(masks, row - local_row + i);
                if (current_min > load2) {
                    current_min = load2;
                }
            }
        }

        // That was one round of fixed-point min iteration.
        store(masks, row, current_min);
        __syncwarp(activemask);
    }

    // Now translate that into masks.
    uint32 mask = 0b0;
    const auto component = load(masks, row);
    for (IndexType i = 0; i < warp_inverse_size; ++i) {
        if (load(masks, row - local_row + i) == component) {
            mask |= (0b1 << (is_upper ? warp_inverse_size - i - 1 : i));
        }
    }

    __syncwarp(activemask);

    masks[row] = mask;
}


template <typename ValueType, typename IndexType, bool is_upper>
__global__ void sptrsvebcrwi_kernel(
    const IndexType* const rowptrs, const IndexType* const colidxs,
    const IndexType* const row_skip_counts, const ValueType* const vals,
    const ValueType* const b, const size_type b_stride, ValueType* const x,
    const size_type x_stride, const ValueType* const band_inv,
    const uint32* const masks, const size_type n, const size_type nrhs,
    bool* nan_produced)
{
    const auto gid = thread::get_thread_id_flat<IndexType>();
    const auto row =
        is_upper ? ((IndexType)n + blockDim.x - 1) / blockDim.x * blockDim.x -
                       gid - 1
                 : gid;
    const auto rhs = blockDim.y * blockIdx.y + threadIdx.y;

    if (row >= n) {
        return;
    }
    if (rhs >= nrhs) {
        return;
    }

    const int self_shid = row % default_block_size;
    const auto skip_count = row_skip_counts[row];

    const auto row_start = is_upper ? rowptrs[row + 1] - 1 : rowptrs[row];
    const auto row_end =
        is_upper ? rowptrs[row] + skip_count - 1
                 : rowptrs[row + 1] - skip_count;  // no -1, as skip_count >= 1
    const auto row_step = is_upper ? -1 : 1;

    ValueType sum = 0.0;
    for (IndexType i = row_start; i != row_end; i += row_step) {
        const auto dependency = colidxs[i];
        auto x_p = &x[dependency * x_stride + rhs];

        ValueType x = *x_p;
        while (is_nan(x)) {
            x = load(x_p, 0);
        }

        sum += x * vals[i];
    }

    __shared__ uninitialized_array<ValueType, default_block_size> b_s_array;
    ValueType* b_s = b_s_array;
    store(b_s, self_shid, b[row * b_stride + rhs] - sum);

    // Now sync all necessary threads before going into the mult.
    // Inactive threads can not have a sync bit set.
    const auto syncmask = masks[row];
    __syncwarp(syncmask);

    const auto band_inv_block =
        band_inv +
        (warp_inverse_size * warp_inverse_size) * (row / warp_inverse_size) +
        row % warp_inverse_size;
    const auto local_offset = row % warp_inverse_size;

    ValueType inv_sum = zero<ValueType>();
    for (int i = 0; i < warp_inverse_size; ++i) {
        inv_sum += band_inv_block[i * warp_inverse_size] *
                   load(b_s, self_shid - local_offset + i);
    }

    const auto r = inv_sum;
    x[row * x_stride + rhs] = r;

    // This check to ensure no infinte loops happen.
    if (is_nan(r)) {
        x[row * x_stride + rhs] = zero<ValueType>();
        *nan_produced = true;
    }
}


template <typename ValueType, typename IndexType>
struct SptrsvebrwiSolveStruct : gko::solver::SolveStruct {
    bool is_upper;
    bool unit_diag;
    array<ValueType> band_inv;
    array<IndexType> row_skip_counts;
    array<uint32> masks;

    SptrsvebrwiSolveStruct(std::shared_ptr<const gko::CudaExecutor> exec,
                           const matrix::Csr<ValueType, IndexType>* matrix,
                           size_type, bool is_upper, bool unit_diag)
        : is_upper{is_upper},
          unit_diag{unit_diag},
          band_inv{exec, static_cast<uint64>(warp_inverse_size) *
                             static_cast<uint64>(warp_inverse_size) *
                             ceildiv(matrix->get_size()[0],
                                     static_cast<uint64>(warp_inverse_size))},
          row_skip_counts{exec, matrix->get_size()[0]},
          masks{exec, matrix->get_size()[0]}
    {
        const auto n = matrix->get_size()[0];
        const auto inv_blocks_count = ceildiv(n, warp_inverse_size);

        cudaMemset(band_inv.get_data(), 0,
                   warp_inverse_size * warp_inverse_size * inv_blocks_count *
                       sizeof(ValueType));
        cudaMemset(masks.get_data(), 0, n * sizeof(uint32));

        const dim3 block_size(default_block_size, 1, 1);
        const dim3 grid_size(ceildiv(n, block_size.x), 1, 1);

        if (is_upper) {
            sptrsvebcrwi_generate_prep_kernel<IndexType, true>
                <<<grid_size, block_size>>>(matrix->get_const_row_ptrs(),
                                            matrix->get_const_col_idxs(),
                                            row_skip_counts.get_data(), n);
            sptrsvebcrwi_generate_inv_kernel<
                decltype(as_cuda_type(ValueType{})), IndexType, true>
                <<<grid_size, block_size>>>(
                    matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                    as_cuda_type(matrix->get_const_values()),
                    row_skip_counts.get_data(),
                    as_cuda_type(band_inv.get_data()), masks.get_data(), n,
                    unit_diag);
        } else {
            sptrsvebcrwi_generate_prep_kernel<IndexType, false>
                <<<grid_size, block_size>>>(matrix->get_const_row_ptrs(),
                                            matrix->get_const_col_idxs(),
                                            row_skip_counts.get_data(), n);
            sptrsvebcrwi_generate_inv_kernel<
                decltype(as_cuda_type(ValueType{})), IndexType, false>
                <<<grid_size, block_size>>>(
                    matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                    as_cuda_type(matrix->get_const_values()),
                    row_skip_counts.get_data(),
                    as_cuda_type(band_inv.get_data()), masks.get_data(), n,
                    unit_diag);
        }
    }

    void solve(std::shared_ptr<const CudaExecutor> exec,
               const matrix::Csr<ValueType, IndexType>* matrix,
               const matrix::Dense<ValueType>* b,
               matrix::Dense<ValueType>* x) const
    {
        const auto n = matrix->get_size()[0];
        const auto nrhs = b->get_size()[1];

        // TODO: Optimize for multiple rhs, by calling to a device gemm.

        dense::fill(exec, x, nan<ValueType>());

        array<bool> nan_produced(exec, {false});

        const dim3 block_size(default_block_size, 1, 1);
        const dim3 grid_size(ceildiv(n, block_size.x), nrhs, 1);
        if (is_upper) {
            sptrsvebcrwi_kernel<decltype(as_cuda_type(ValueType{})), IndexType,
                                true><<<grid_size, block_size>>>(
                matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                row_skip_counts.get_const_data(),
                as_cuda_type(matrix->get_const_values()),
                as_cuda_type(b->get_const_values()), b->get_stride(),
                as_cuda_type(x->get_values()), x->get_stride(),
                as_cuda_type(band_inv.get_const_data()), masks.get_const_data(),
                n, nrhs, nan_produced.get_data());
        } else {
            sptrsvebcrwi_kernel<decltype(as_cuda_type(ValueType{})), IndexType,
                                false><<<grid_size, block_size>>>(
                matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                row_skip_counts.get_const_data(),
                as_cuda_type(matrix->get_const_values()),
                as_cuda_type(b->get_const_values()), b->get_stride(),
                as_cuda_type(x->get_values()), x->get_stride(),
                as_cuda_type(band_inv.get_const_data()), masks.get_const_data(),
                n, nrhs, nan_produced.get_data());
        }
    }
};


template <bool is_upper, typename IndexType>
__global__ void sptrsvebcrwvs_write_vwarp_ids(
    const IndexType* const vwarp_offsets, IndexType* const vwarp_ids,
    const IndexType num_vwarps, const IndexType n)
{
    const auto gid = thread::get_thread_id_flat();

    if (gid >= num_vwarps) {
        return;
    }

    const auto vwarp_start = vwarp_offsets[gid];
    const auto vwarp_end = vwarp_offsets[gid + 1];

    for (IndexType i = vwarp_start; i < vwarp_end; ++i) {
        vwarp_ids[i] = is_upper ? n - gid - 1 : gid;
    }
}

// This is "heavily inspired" by cppreference.
template <class T>
__device__ const T* lower_bound(const T* first, const T* const last,
                                const T value)
{
    const T* p;
    auto count = last - first;
    auto step = count;
    while (count > 0) {
        p = first;
        step = count / 2;
        p += step;
        if (*p < value) {
            first = ++p;
            count -= step + 1;
        } else {
            count = step;
        }
    }
    return first;
}

template <bool is_upper, typename IndexType>
__global__ void sptrsvebcrwvs_generate_assigned_sizes(
    const IndexType* const rowptrs, const IndexType* const colidxs,
    const double avg_threads_per_row, IndexType* const assigned_sizes,
    IndexType* const entry_counts, const IndexType n, const IndexType nnz)
{
    const IndexType gid = thread::get_thread_id_flat();
    const auto row = is_upper ? n - gid - 1 : gid;
    const auto row_write_location = gid;
    const int32 thread = threadIdx.x;

    if (gid >= n) {
        return;
    }

    const auto diag_pos =
        lower_bound(colidxs + rowptrs[row], colidxs + rowptrs[row + 1], row) -
        (colidxs + rowptrs[row]);
    const auto valid_entry_count =
        is_upper ? rowptrs[row + 1] - rowptrs[row] - diag_pos : diag_pos + 1;
    entry_counts[row] = valid_entry_count;

    const double avg_nnz = (double)nnz / n;
    const double perfect_size =
        (valid_entry_count)*avg_threads_per_row / avg_nnz;
    const IndexType assigned_size = std::max(
        std::min((IndexType)__double2int_rn(perfect_size), (IndexType)32),
        (IndexType)1);

    volatile __shared__ int32 block_size_assigner[1];
    volatile __shared__ int32 block_size_assigner_lock[1];

    *block_size_assigner = 0;
    *block_size_assigner_lock = -1;

    __syncthreads();

    while (*block_size_assigner_lock != thread - 1) {
    }

    const auto prev_offset = *block_size_assigner;
    *block_size_assigner += assigned_size;

    int32 shrinked_size = 0;
    if ((prev_offset + assigned_size) / 32 > prev_offset / 32) {
        shrinked_size = ((prev_offset + assigned_size) / 32) * 32 - prev_offset;
        *block_size_assigner += shrinked_size - assigned_size;
    }

    __threadfence();
    *block_size_assigner_lock = thread;

    assigned_sizes[row_write_location] =
        shrinked_size == 0 ? assigned_size : shrinked_size;

    // This part to ensure each assigner block starts on 32*k, meaning the cuts
    // are well-placed.
    if (thread == default_block_size - 1) {
        if ((prev_offset +
             (shrinked_size == 0 ? assigned_size : shrinked_size)) %
                32 !=
            0) {
            assigned_sizes[row_write_location] =
                (prev_offset / 32 + 1) * 32 - prev_offset;
        }
    }
}


template <bool is_upper, typename ValueType, typename IndexType>
__global__ void sptrsvebcrwvs_kernel(
    const IndexType* const rowptrs, const IndexType* const colidxs,
    const ValueType* const vals, const IndexType* const vwarp_ids,
    const IndexType* const vwarp_offsets, const IndexType* const entry_counts,
    const ValueType* const b, const size_type b_stride, ValueType* const x,
    const size_type x_stride, bool* const nan_produced,
    const IndexType num_vthreads, const IndexType n, const IndexType nrhs,
    const bool unit_diag)
{
    const auto gid = blockIdx.x * blockDim.x + threadIdx.x;
    const auto thread = threadIdx.x;
    const auto rhs = blockIdx.y * blockDim.y + threadIdx.y;

    if (gid >= num_vthreads) {
        return;
    }
    if (rhs >= nrhs) {
        return;
    }

    const auto vwarp = vwarp_ids[gid];
    const auto vwarp_start = vwarp_offsets[is_upper ? n - vwarp - 1 : vwarp];
    const auto vwarp_end = vwarp_offsets[is_upper ? n - vwarp : vwarp + 1];
    const auto vwarp_size = vwarp_end - vwarp_start;
    const auto row = vwarp;
    const IndexType vthread = gid - vwarp_start;

    if (row >= n) {
        return;
    }

    const auto row_begin = is_upper ? rowptrs[row + 1] - 1 : rowptrs[row];
    const auto row_end = is_upper ? rowptrs[row] : rowptrs[row + 1] - 1;
    const IndexType row_step = is_upper ? -1 : 1;

    const auto valid_entry_count = entry_counts[row];
    const auto start_offset = (valid_entry_count - 1) % vwarp_size;

    auto sum = zero<ValueType>();
    // i is adjusted for vthread 0 to hit the diagonal
    IndexType i =
        unit_diag
            ? row_begin + row_step * vthread
            : row_begin +
                  ((row_step * vthread + row_step * start_offset) % vwarp_size);
    for (; (is_upper && i > row_end) || (!is_upper && i < row_end);
         i += row_step * vwarp_size) {
        const auto dependency = colidxs[i];

        if (is_upper ? dependency <= row : dependency >= row) {
            break;
        }

        volatile auto x_p = &x[x_stride * dependency + rhs];

        auto l = *x_p;
        while (is_nan(l)) {
            l = load(x_p, 0);
        }


        sum += l * vals[i];
    }

    uint32 syncmask = ((1 << vwarp_size) - 1) << (vwarp_start & 31);

    ValueType total = sum;
    for (int offset = 1; offset < vwarp_size; ++offset) {
        auto a = real(sum);
        const auto received_a = __shfl_down_sync(syncmask, a, offset);
        const auto should_add = (syncmask >> ((thread & 31) + offset)) & 1 == 1;
        total += should_add * received_a;
        if (gko::is_complex<ValueType>()) {
            auto b = imag(sum);
            const auto received_b = __shfl_down_sync(syncmask, b, offset);
            auto ptotal =
                (thrust::complex<gko::remove_complex<ValueType>>*)&total;
            *ptotal += should_add * received_b *
                       (thrust::complex<gko::remove_complex<ValueType>>)
                           unit_root<ValueType>(4);
        }
    }

    if (vthread == 0) {
        const auto diag = unit_diag ? one<ValueType>() : vals[i];
        const auto r = (b[row * b_stride + rhs] - total) / diag;
        x[row * x_stride + rhs] = r;

        // This check to ensure no infinte loops happen.
        if (is_nan(r)) {
            x[row * x_stride + rhs] = zero<ValueType>();
            *nan_produced = true;
        }
    }
}


template <typename ValueType, typename IndexType>
struct SptrsvebrwvSolveStruct : gko::solver::SolveStruct {
    bool is_upper;
    bool unit_diag;
    IndexType vthread_count;
    array<IndexType> vwarp_ids;
    array<IndexType> vwarp_offsets;
    array<IndexType> entry_counts;

    SptrsvebrwvSolveStruct(std::shared_ptr<const gko::CudaExecutor> exec,
                           const matrix::Csr<ValueType, IndexType>* matrix,
                           size_type, bool is_upper, bool unit_diag)
        : is_upper{is_upper},
          unit_diag{unit_diag},
          vwarp_offsets{exec, matrix->get_size()[0] + 1},
          entry_counts{exec, matrix->get_size()[0]},
          vwarp_ids{exec}
    {
        const auto desired_avg_threads_per_row = 1.0;

        const IndexType n = matrix->get_size()[0];
        const IndexType nnz = matrix->get_num_stored_elements();

        array<IndexType> assigned_sizes(exec, n);

        const auto block_size = default_block_size;
        const auto block_count = (n + block_size - 1) / block_size;

        if (is_upper) {
            sptrsvebcrwvs_generate_assigned_sizes<true>
                <<<block_count, block_size>>>(
                    matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                    desired_avg_threads_per_row, assigned_sizes.get_data(),
                    entry_counts.get_data(), n, nnz);
        } else {
            sptrsvebcrwvs_generate_assigned_sizes<false>
                <<<block_count, block_size>>>(
                    matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                    desired_avg_threads_per_row, assigned_sizes.get_data(),
                    entry_counts.get_data(), n, nnz);
        }

        cudaMemcpy(vwarp_offsets.get_data(), assigned_sizes.get_const_data(),
                   n * sizeof(IndexType), cudaMemcpyDeviceToDevice);
        components::prefix_sum(exec, vwarp_offsets.get_data(), n + 1);

        cudaMemcpy(&vthread_count, vwarp_offsets.get_const_data() + n,
                   sizeof(IndexType), cudaMemcpyDeviceToHost);

        vwarp_ids.resize_and_reset(vthread_count);
        const auto block_size_vwarped = default_block_size;
        const auto block_count_vwarped =
            (n + block_size_vwarped - 1) / block_size_vwarped;
        if (is_upper) {
            sptrsvebcrwvs_write_vwarp_ids<true>
                <<<block_count_vwarped, block_size_vwarped>>>(
                    vwarp_offsets.get_const_data(), vwarp_ids.get_data(), n, n);
        } else {
            sptrsvebcrwvs_write_vwarp_ids<false>
                <<<block_count_vwarped, block_size_vwarped>>>(
                    vwarp_offsets.get_const_data(), vwarp_ids.get_data(), n, n);
        }
    }

    void solve(std::shared_ptr<const CudaExecutor> exec,
               const matrix::Csr<ValueType, IndexType>* matrix,
               const matrix::Dense<ValueType>* b,
               matrix::Dense<ValueType>* x) const
    {
        const IndexType n = matrix->get_size()[0];
        const IndexType nrhs = b->get_size()[1];

        // TODO: Optimize for multiple rhs.

        dense::fill(exec, x, nan<ValueType>());

        array<bool> nan_produced(exec, {false});

        const dim3 block_size(default_block_size, 1024 / default_block_size, 1);
        const dim3 grid_size(ceildiv(vthread_count, block_size.x),
                             ceildiv(nrhs, block_size.y), 1);

        if (is_upper) {
            sptrsvebcrwvs_kernel<true><<<grid_size, block_size>>>(
                matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                as_cuda_type(matrix->get_const_values()),
                vwarp_ids.get_const_data(), vwarp_offsets.get_const_data(),
                entry_counts.get_const_data(),
                as_cuda_type(b->get_const_values()), b->get_stride(),
                as_cuda_type(x->get_values()), x->get_stride(),
                nan_produced.get_data(), vthread_count, n, nrhs, unit_diag);
        } else {
            sptrsvebcrwvs_kernel<false><<<grid_size, block_size>>>(
                matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                as_cuda_type(matrix->get_const_values()),
                vwarp_ids.get_const_data(), vwarp_offsets.get_const_data(),
                entry_counts.get_const_data(),
                as_cuda_type(b->get_const_values()), b->get_stride(),
                as_cuda_type(x->get_values()), x->get_stride(),
                nan_produced.get_data(), vthread_count, n, nrhs, unit_diag);
        }

#if GKO_VERBOSE_LEVEL >= 1
        if (exec->copy_val_to_host(nan_produced.get_const_data())) {
            std::cerr << "Error: triangular solve produced NaN, either not all "
                         "diagonal "
                         "elements are nonzero, or the system is very "
                         "ill-conditioned. "
                         "The NaN will be replaced with a zero.\n";
        }
#endif  // GKO_VERBOSE_LEVEL >= 1
    }
};


template <typename ValueType, typename IndexType>
struct BlockedSolveStruct : solver::SolveStruct {
    struct pos_size_depth {
        std::pair<IndexType, IndexType> pos;
        std::pair<IndexType, IndexType> size;
        IndexType depth;

        pos_size_depth left_child(IndexType max_depth) const
        {
            if (depth == max_depth - 1) {  // Check if triangle
                return pos_size_depth{
                    std::make_pair(pos.first - size.second, pos.second),
                    std::make_pair(size.second, size.second), max_depth};
            } else {
                return pos_size_depth{
                    std::make_pair(pos.first - size.first / 2, pos.second),
                    std::make_pair(size.first / 2,
                                   size.second - size.first / 2),
                    depth + 1};
            }
        }

        pos_size_depth right_child(IndexType max_depth) const
        {
            if (depth == max_depth - 1) {
                return pos_size_depth{
                    std::make_pair(pos.first, pos.second + size.second),
                    std::make_pair(size.first, size.first), max_depth};
            } else {
                return pos_size_depth{
                    std::make_pair(pos.first + size.first / 2,
                                   pos.second + size.second),
                    std::make_pair(
                        ceildiv(size.first, 2),
                        (pos.first + size.first - (pos.second + size.second)) /
                            2),
                    depth + 1};
            }
        }
    };

    std::vector<std::shared_ptr<solver::SolveStruct>> solvers;
    std::vector<std::shared_ptr<matrix::Csr<ValueType, IndexType>>> blocks;
    std::vector<pos_size_depth> block_coords;

    void solve(std::shared_ptr<const CudaExecutor> exec,
               const matrix::Csr<ValueType, IndexType>* matrix,
               const matrix::Dense<ValueType>* b,
               matrix::Dense<ValueType>* x) const
    {
        auto mb = matrix::Dense<ValueType>::create(exec);
        mb->copy_from(b);

        const auto block_count = blocks.size();
        for (IndexType i = 0; i < block_count; ++i) {
            if (i % 2 == 0) {
                const auto bv =
                    mb->create_submatrix(span{block_coords[i].pos.second,
                                              block_coords[i].pos.second +
                                                  block_coords[i].size.second},
                                         span{0, 1});
                auto xv =
                    x->create_submatrix(span{block_coords[i].pos.first,
                                             block_coords[i].pos.first +
                                                 block_coords[i].size.first},
                                        span{0, 1});

                solvers[i / 2]->solve(exec, blocks[i].get(), bv.get(), xv.get(),
                                      bv.get(), xv.get());
            } else {
                const auto xv =
                    x->create_submatrix(span{block_coords[i].pos.second,
                                             block_coords[i].pos.second +
                                                 block_coords[i].size.second},
                                        span{0, 1});
                auto bv =
                    mb->create_submatrix(span{block_coords[i].pos.first,
                                              block_coords[i].pos.first +
                                                  block_coords[i].size.first},
                                         span{0, 1});
                auto neg_one =
                    gko::initialize<gko::matrix::Dense<ValueType>>({-1}, exec);
                auto one =
                    gko::initialize<gko::matrix::Dense<ValueType>>({1}, exec);
                blocks[i]->apply(neg_one.get(), xv.get(), one.get(), bv.get());
            }
        }
    }


    BlockedSolveStruct(
        std::shared_ptr<const CudaExecutor> exec,
        const matrix::Csr<ValueType, IndexType>* matrix,
        const gko::size_type num_rhs, bool is_upper, bool unit_diag,
        std::shared_ptr<
            std::vector<std::shared_ptr<gko::solver::trisolve_strategy>>>
            solver_ids)
    {
        const auto host_exec = exec->get_master();
        const auto n = matrix->get_size()[0];
        const auto sptrsv_count = solver_ids->size();
        const auto block_count = 2 * sptrsv_count - 1;
        const auto depth = get_significant_bit(sptrsv_count);

        // Generate the block sizes and positions.
        array<pos_size_depth> blocks(host_exec, block_count);
        pos_size_depth* blocksp = blocks.get_data();
        blocksp[0] = pos_size_depth{std::make_pair(n / 2, 0),
                                    std::make_pair(ceildiv(n, 2), n / 2), 0};
        IndexType write = 1;
        for (IndexType read = 0; write < block_count; ++read) {
            const auto cur = blocksp[read];
            blocksp[write++] = cur.left_child(depth);
            blocksp[write++] = cur.right_child(depth);
        }

        // Generate a permutation to execution order
        array<IndexType> perm(host_exec, block_count);
        IndexType* permp = perm.get_data();
        for (IndexType i = 0; i <= depth; ++i) {
            const auto step = 2 << i;
            const auto start = (1 << i) - 1;
            const auto add = sptrsv_count / (1 << i) - 1;

            for (IndexType j = start; j < block_count; j += step) {
                permp[j] = (j - start) / step + add;
            }
        }

        // Apply the perm
        // For upper_trs, we also need to reflect the cuts
        for (IndexType i = 0; i < block_count; ++i) {
            auto block = blocksp[permp[i]];

            if (is_upper) {
                std::swap(block.pos.first, block.pos.second);
                std::swap(block.size.first, block.size.second);
            }

            block_coords.push_back(block);
            this->blocks.push_back(std::move(matrix->create_submatrix(
                span{block.pos.first, block.pos.first + block.size.first},
                span{block.pos.second, block.pos.second + block.size.second})));
            this->blocks[i]->set_strategy(
                std::make_shared<
                    typename matrix::Csr<ValueType, IndexType>::automatical>(
                    exec));
        }

        if (is_upper) {
            for (auto i = 0; i < block_count / 2; ++i) {
                std::swap(block_coords[i], block_coords[block_count - i - 1]);
                std::swap(this->blocks[i], this->blocks[block_count - i - 1]);
            }
        }

        // Finally create the appropriate solvers
        for (IndexType i = 0; i < sptrsv_count; ++i) {
            this->solvers.push_back(std::make_shared<solver::SolveStruct>());
            solver::SolveStruct::generate(
                exec, this->blocks[2 * i].get(), this->solvers[i], num_rhs,
                solver_ids.get()->at(i).get(), is_upper, unit_diag);
        }
    }
};


}  // namespace
}  // namespace cuda
}  // namespace kernels


template <typename ValueType, typename IndexType>
void gko::solver::SolveStruct::generate<ValueType, IndexType>(
    std::shared_ptr<const CudaExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* matrix,
    std::shared_ptr<solver::SolveStruct>& solve_struct,
    const gko::size_type num_rhs,
    const gko::solver::trisolve_strategy* strategy, bool is_upper,
    bool unit_diag)
{
    if (matrix->get_size()[0] == 0) {
        return;
    }
    if (strategy->type == gko::solver::trisolve_type::sparselib) {
        if (gko::kernels::cuda::cusparse::is_supported<ValueType,
                                                       IndexType>::value) {
            solve_struct = std::make_shared<
                gko::kernels::cuda::CudaSolveStruct<ValueType, IndexType>>(
                exec, matrix, num_rhs, is_upper, unit_diag);
        } else {
            GKO_NOT_SUPPORTED(solve_struct);
        }
    } else if (strategy->type == gko::solver::trisolve_type::level) {
        solve_struct = std::make_shared<
            gko::kernels::cuda::SptrsvlrSolveStruct<ValueType, IndexType>>(
            exec, matrix, num_rhs, is_upper, unit_diag);
    } else if (strategy->type == gko::solver::trisolve_type::winv) {
        solve_struct = std::make_shared<
            gko::kernels::cuda::SptrsvebrwiSolveStruct<ValueType, IndexType>>(
            exec, matrix, num_rhs, is_upper, unit_diag);
    } else if (strategy->type == gko::solver::trisolve_type::wvar) {
        solve_struct = std::make_shared<
            gko::kernels::cuda::SptrsvebrwvSolveStruct<ValueType, IndexType>>(
            exec, matrix, num_rhs, is_upper, unit_diag);
    } else if (strategy->type == gko::solver::trisolve_type::thinned) {
        solve_struct = std::make_shared<
            gko::kernels::cuda::SptrsvebcrnmSolveStruct<ValueType, IndexType>>(
            exec, matrix, num_rhs, is_upper, unit_diag, strategy->thinned_m);
    } else if (strategy->type == gko::solver::trisolve_type::block) {
        solve_struct = std::make_shared<
            gko::kernels::cuda::BlockedSolveStruct<ValueType, IndexType>>(
            exec, matrix, num_rhs, is_upper, unit_diag, strategy->block_inner);
    } else if (strategy->type == gko::solver::trisolve_type::syncfree) {
        solve_struct = std::make_shared<
            gko::kernels::cuda::SptrsvebcrnSolveStruct<ValueType, IndexType>>(
            exec, matrix, num_rhs, is_upper, unit_diag);
    }
}

#define GKO_DECLARE_SOLVER_SOLVESTRUCT_GENERATE(_vtype, _itype)        \
    void gko::solver::SolveStruct::generate(                           \
        std::shared_ptr<const CudaExecutor> exec,                      \
        const matrix::Csr<_vtype, _itype>* matrix,                     \
        std::shared_ptr<solver::SolveStruct>& solve_struct,            \
        const gko::size_type num_rhs,                                  \
        const gko::solver::trisolve_strategy* strategy, bool is_upper, \
        bool unit_diag)

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SOLVER_SOLVESTRUCT_GENERATE);


template <typename ValueType, typename IndexType>
void gko::solver::SolveStruct::solve(
    std::shared_ptr<const CudaExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* matrix,
    matrix::Dense<ValueType>* trans_b, matrix::Dense<ValueType>* trans_x,
    const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* x) const
{
    if (matrix->get_size()[0] == 0 || b->get_size()[1] == 0) {
        return;
    }
    if (auto sptrsvebcrn_struct =
            dynamic_cast<const gko::kernels::cuda::SptrsvebcrnSolveStruct<
                ValueType, IndexType>*>(this)) {
        sptrsvebcrn_struct->solve(exec, matrix, b, x);
    } else if (auto sptrsvlr_struct =
                   dynamic_cast<const gko::kernels::cuda::SptrsvlrSolveStruct<
                       ValueType, IndexType>*>(this)) {
        sptrsvlr_struct->solve(exec, matrix, b, x);
    } else if (auto sptrsvebcrnm_struct = dynamic_cast<
                   const gko::kernels::cuda::SptrsvebcrnmSolveStruct<
                       ValueType, IndexType>*>(this)) {
        sptrsvebcrnm_struct->solve(exec, matrix, b, x);
    } else if (auto sptrsvebrwi_struct = dynamic_cast<
                   const gko::kernels::cuda::SptrsvebrwiSolveStruct<
                       ValueType, IndexType>*>(this)) {
        sptrsvebrwi_struct->solve(exec, matrix, b, x);
    } else if (auto sptrsvb_struct =
                   dynamic_cast<const gko::kernels::cuda::BlockedSolveStruct<
                       ValueType, IndexType>*>(this)) {
        sptrsvb_struct->solve(exec, matrix, b, x);
    } else if (auto sptrsvwv_struct = dynamic_cast<
                   const gko::kernels::cuda::SptrsvebrwvSolveStruct<
                       ValueType, IndexType>*>(this)) {
        sptrsvwv_struct->solve(exec, matrix, b, x);
    } else if (gko::kernels::cuda::cusparse::is_supported<
                   ValueType,
                   IndexType>::value) {  // Must always be last check
        if (auto cuda_solve_struct =
                dynamic_cast<const gko::kernels::cuda::CudaSolveStruct<
                    ValueType, IndexType>*>(this)) {
            cuda_solve_struct->solve(matrix, b, x, trans_b, trans_x);
        } else {
            GKO_NOT_SUPPORTED(this);
        }
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

#define GKO_DECLARE_SOLVER_SOLVESTRUCT_SOLVE(_vtype, _itype)            \
    void gko::solver::SolveStruct::solve(                               \
        std::shared_ptr<const CudaExecutor> exec,                       \
        const matrix::Csr<_vtype, _itype>* matrix,                      \
        matrix::Dense<_vtype>* trans_b, matrix::Dense<_vtype>* trans_x, \
        const matrix::Dense<_vtype>* b, matrix::Dense<_vtype>* x) const

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SOLVER_SOLVESTRUCT_SOLVE);


}  // namespace gko
