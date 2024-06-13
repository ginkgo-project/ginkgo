// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CUDA_SOLVER_COMMON_TRS_KERNELS_CUH_
#define GKO_CUDA_SOLVER_COMMON_TRS_KERNELS_CUH_


#include <functional>
#include <iostream>
#include <memory>


#include <cuda.h>
#include <cusparse.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>


#include "core/base/array_access.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "cuda/base/cusparse_bindings.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/pointer_mode_guard.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/atomic.cuh"
#include "cuda/components/memory.cuh"
#include "cuda/components/thread_ids.cuh"
#include "cuda/components/uninitialized_array.hpp"


namespace gko {
namespace solver {


struct SolveStruct {
    virtual ~SolveStruct() = default;
};


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
        // also make sure they are aligned by 16 bytes
        auto descr_b = cusparse::create_dnmat(
            dim<2>{matrix->get_size()[0], num_rhs}, matrix->get_size()[1],
            reinterpret_cast<ValueType*>(0xDEAD0));
        auto descr_c = cusparse::create_dnmat(
            dim<2>{matrix->get_size()[0], num_rhs}, matrix->get_size()[1],
            reinterpret_cast<ValueType*>(0xDEAF0));

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


#else

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


template <typename ValueType, typename IndexType>
void generate_kernel(std::shared_ptr<const CudaExecutor> exec,
                     const matrix::Csr<ValueType, IndexType>* matrix,
                     std::shared_ptr<solver::SolveStruct>& solve_struct,
                     const gko::size_type num_rhs, bool is_upper,
                     bool unit_diag)
{
    if (matrix->get_size()[0] == 0) {
        return;
    }
    if (cusparse::is_supported<ValueType, IndexType>::value) {
        solve_struct = std::make_shared<CudaSolveStruct<ValueType, IndexType>>(
            exec, matrix, num_rhs, is_upper, unit_diag);
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
    if (matrix->get_size()[0] == 0 || b->get_size()[1] == 0) {
        return;
    }
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
        ValueType val{};
        if (shmem_possible) {
            const auto dependency_shid = dependency_gid % default_block_size;
            while (is_nan(val = load_relaxed_shared(x_s + dependency_shid))) {
            }
        } else {
            while (
                is_nan(val = load_relaxed(x + dependency * x_stride + rhs))) {
            }
        }

        sum += val * vals[i];
    }

    // The first entry past the triangular part will be the diagonal
    const auto diag = unit_diag ? one<ValueType>() : vals[i];
    const auto r = (b[row * b_stride + rhs] - sum) / diag;

    store_relaxed_shared(x_s + self_shid, r);
    store_relaxed(x + row * x_stride + rhs, r);

    // This check to ensure no infinite loops happen.
    if (is_nan(r)) {
        store_relaxed_shared(x_s + self_shid, zero<ValueType>());
        store_relaxed(x + row * x_stride + rhs, zero<ValueType>());
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

    ValueType sum = 0.0;
    auto j = row_begin;
    auto col = colidxs[j];
    while (j != row_end) {
        auto x_val = load_relaxed(x + col * x_stride + rhs);
        while (!is_nan(x_val)) {
            sum += vals[j] * x_val;
            j += row_step;
            col = colidxs[j];
            x_val = load_relaxed(x + col * x_stride + rhs);
        }
        // to avoid the kernel hanging on matrices without diagonal,
        // we bail out if we are past the triangle, even if it's not
        // the diagonal entry. This may lead to incorrect results,
        // but prevents an infinite loop.
        if (is_upper ? row >= col : row <= col) {
            // assert(row == col);
            auto diag = unit_diag ? one<ValueType>() : vals[j];
            const auto r = (b[row * b_stride + rhs] - sum) / diag;
            store_relaxed(x + row * x_stride + rhs, r);
            // after we encountered the diagonal, we are done
            // this also skips entries outside the triangle
            j = row_end;
            if (is_nan(r)) {
                store_relaxed(x + row * x_stride + rhs, zero<ValueType>());
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


template <bool is_upper, typename ValueType, typename IndexType>
void sptrsv_naive_caching(std::shared_ptr<const CudaExecutor> exec,
                          const matrix::Csr<ValueType, IndexType>* matrix,
                          bool unit_diag, const matrix::Dense<ValueType>* b,
                          matrix::Dense<ValueType>* x)
{
    // Pre-Volta GPUs may deadlock due to missing independent thread scheduling.
    const auto is_fallback_required = exec->get_major_version() < 7;

    const auto n = matrix->get_size()[0];
    const auto nrhs = b->get_size()[1];

    // Initialize x to all NaNs.
    dense::fill(exec, x, nan<ValueType>());

    array<bool> nan_produced(exec, 1);
    array<IndexType> atomic_counter(exec, 1);
    sptrsv_init_kernel<<<1, 1, 0, exec->get_stream()>>>(
        nan_produced.get_data(), atomic_counter.get_data());

    const dim3 block_size(
        is_fallback_required ? fallback_block_size : default_block_size, 1, 1);
    const dim3 grid_size(ceildiv(n * nrhs, block_size.x), 1, 1);

    if (is_fallback_required) {
        sptrsv_naive_legacy_kernel<is_upper>
            <<<grid_size, block_size, 0, exec->get_stream()>>>(
                matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                as_device_type(matrix->get_const_values()),
                as_device_type(b->get_const_values()), b->get_stride(),
                as_device_type(x->get_values()), x->get_stride(), n, nrhs,
                unit_diag, nan_produced.get_data(), atomic_counter.get_data());
    } else {
        sptrsv_naive_caching_kernel<is_upper>
            <<<grid_size, block_size, 0, exec->get_stream()>>>(
                matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                as_device_type(matrix->get_const_values()),
                as_device_type(b->get_const_values()), b->get_stride(),
                as_device_type(x->get_values()), x->get_stride(), n, nrhs,
                unit_diag, nan_produced.get_data(), atomic_counter.get_data());
    }

#if GKO_VERBOSE_LEVEL >= 1
    if (get_element(nan_produced, 0)) {
        std::cerr
            << "Error: triangular solve produced NaN, either not all diagonal "
               "elements are nonzero, or the system is very ill-conditioned. "
               "The NaN will be replaced with a zero.\n";
    }
#endif  // GKO_VERBOSE_LEVEL >= 1
}


}  // namespace
}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CUDA_SOLVER_COMMON_TRS_KERNELS_CUH_
