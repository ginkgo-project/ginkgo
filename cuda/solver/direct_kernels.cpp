// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/direct_kernels.hpp"


#if GKO_HAVE_CUDSS


#include <type_traits>

#include <cudss.h>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "cuda/base/types.hpp"


namespace gko {
namespace kernels {
namespace cuda {
namespace direct {
namespace {


#define GKO_ASSERT_NO_CUDSS_ERRORS(_cudss_call)                       \
    do {                                                              \
        auto _status = (_cudss_call);                                 \
        if (_status != CUDSS_STATUS_SUCCESS) {                        \
            throw ::gko::InvalidStateError(                           \
                __FILE__, __LINE__, __func__,                         \
                std::string("cuDSS error (status ") +                 \
                    std::to_string(static_cast<int>(_status)) + ")"); \
        }                                                             \
    } while (false)


template <typename T>
constexpr bool is_cudss_supported_type()
{
    return std::is_same_v<T, float> || std::is_same_v<T, double> ||
           std::is_same_v<T, std::complex<float>> ||
           std::is_same_v<T, std::complex<double>>;
}


}  // anonymous namespace
}  // namespace direct
}  // namespace cuda
}  // namespace kernels


/**
 * Full definition of the opaque direct_vendor_state.
 * Only compiled in the CUDA module where cuDSS headers are available.
 */
namespace experimental {
namespace solver {


struct direct_vendor_state {
    cudssHandle_t handle = nullptr;
    cudssConfig_t config = nullptr;
    cudssData_t data = nullptr;
    cudssMatrix_t A = nullptr;
    cudaStream_t stream = nullptr;

    ~direct_vendor_state()
    {
        // Synchronize the stream before cleanup to ensure all async
        // cuDSS operations are complete.
        if (stream) {
            cudaStreamSynchronize(stream);
        }
        if (A) {
            cudssMatrixDestroy(A);
        }
        if (data && handle) {
            cudssDataDestroy(handle, data);
        }
        if (config) {
            cudssConfigDestroy(config);
        }
        if (handle) {
            cudssDestroy(handle);
        }
    }
};


}  // namespace solver
}  // namespace experimental


namespace kernels {
namespace cuda {
namespace direct {


template <typename ValueType, typename IndexType>
void generate(
    std::shared_ptr<const CudaExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* matrix,
    std::shared_ptr<experimental::solver::direct_vendor_state>& solve_state,
    const experimental::solver::vendor_parameters& params)
{
    if constexpr (is_cudss_supported_type<ValueType>()) {
        using state = experimental::solver::direct_vendor_state;
        auto st = std::make_shared<state>();

        GKO_ASSERT_NO_CUDSS_ERRORS(cudssCreate(&st->handle));
        st->stream = exec->get_stream();
        GKO_ASSERT_NO_CUDSS_ERRORS(cudssSetStream(st->handle, st->stream));

        GKO_ASSERT_NO_CUDSS_ERRORS(cudssConfigCreate(&st->config));

        auto reorder_alg = static_cast<cudssAlgType_t>(params.reordering_alg);
        GKO_ASSERT_NO_CUDSS_ERRORS(
            cudssConfigSet(st->config, CUDSS_CONFIG_REORDERING_ALG,
                           &reorder_alg, sizeof(reorder_alg)));

        if (params.hybrid_execute) {
            int hybrid_mode = 1;
            GKO_ASSERT_NO_CUDSS_ERRORS(
                cudssConfigSet(st->config, CUDSS_CONFIG_HYBRID_EXECUTE_MODE,
                               &hybrid_mode, sizeof(hybrid_mode)));
        }

        if (params.hybrid_memory) {
            int hybrid_mode = 1;
            GKO_ASSERT_NO_CUDSS_ERRORS(
                cudssConfigSet(st->config, CUDSS_CONFIG_HYBRID_MODE,
                               &hybrid_mode, sizeof(hybrid_mode)));
        }

        GKO_ASSERT_NO_CUDSS_ERRORS(cudssDataCreate(st->handle, &st->data));

        const auto nrows = static_cast<int64_t>(matrix->get_size()[0]);
        const auto ncols = static_cast<int64_t>(matrix->get_size()[1]);
        const auto nnz =
            static_cast<int64_t>(matrix->get_num_stored_elements());

        auto mtype = static_cast<cudssMatrixType_t>(params.matrix_type);
        auto mview = static_cast<cudssMatrixViewType_t>(params.matrix_view);

        GKO_ASSERT_NO_CUDSS_ERRORS(cudssMatrixCreateCsr(
            &st->A, nrows, ncols, nnz,
            const_cast<IndexType*>(matrix->get_const_row_ptrs()), nullptr,
            const_cast<IndexType*>(matrix->get_const_col_idxs()),
            const_cast<ValueType*>(matrix->get_const_values()),
            cuda_data_type<IndexType>(), cuda_data_type<ValueType>(), mtype,
            mview, CUDSS_BASE_ZERO));

        // Allocate temporary dense vectors for analysis/factorization.
        // Some cuDSS versions require non-null data pointers.
        ValueType* tmp_b_data = nullptr;
        ValueType* tmp_x_data = nullptr;
        cudaMalloc(&tmp_b_data, nrows * sizeof(ValueType));
        cudaMalloc(&tmp_x_data, nrows * sizeof(ValueType));
        cudaMemset(tmp_b_data, 0, nrows * sizeof(ValueType));
        cudaMemset(tmp_x_data, 0, nrows * sizeof(ValueType));

        cudssMatrix_t tmp_b = nullptr;
        cudssMatrix_t tmp_x = nullptr;
        GKO_ASSERT_NO_CUDSS_ERRORS(cudssMatrixCreateDn(
            &tmp_b, nrows, 1, nrows, tmp_b_data, cuda_data_type<ValueType>(),
            CUDSS_LAYOUT_COL_MAJOR));
        GKO_ASSERT_NO_CUDSS_ERRORS(cudssMatrixCreateDn(
            &tmp_x, nrows, 1, nrows, tmp_x_data, cuda_data_type<ValueType>(),
            CUDSS_LAYOUT_COL_MAJOR));

        GKO_ASSERT_NO_CUDSS_ERRORS(
            cudssExecute(st->handle, CUDSS_PHASE_ANALYSIS, st->config, st->data,
                         st->A, tmp_x, tmp_b));

        GKO_ASSERT_NO_CUDSS_ERRORS(
            cudssExecute(st->handle, CUDSS_PHASE_FACTORIZATION, st->config,
                         st->data, st->A, tmp_x, tmp_b));

        cudssMatrixDestroy(tmp_b);
        cudssMatrixDestroy(tmp_x);
        cudaFree(tmp_b_data);
        cudaFree(tmp_x_data);

        solve_state = std::move(st);
    } else {
        GKO_NOT_SUPPORTED(exec);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DIRECT_GENERATE_KERNEL);


template <typename ValueType>
void solve(std::shared_ptr<const CudaExecutor> exec,
           const experimental::solver::direct_vendor_state* solve_state,
           const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* x)
{
    if constexpr (is_cudss_supported_type<ValueType>()) {
        const auto nrows = static_cast<int64_t>(b->get_size()[0]);
        const auto nrhs = static_cast<int64_t>(b->get_size()[1]);

        if (nrows == 0 || nrhs == 0) {
            return;
        }

        // cuDSS requires column-major contiguous dense vectors.
        // Ginkgo Dense submatrix views can have stride > ncols.
        GKO_ASSERT(nrhs == 1);

        const bool b_strided = (b->get_stride() != b->get_size()[1]);
        const bool x_strided = (x->get_stride() != x->get_size()[1]);

        ValueType* b_data = const_cast<ValueType*>(b->get_const_values());
        ValueType* x_data = x->get_values();
        ValueType* b_buf = nullptr;
        ValueType* x_buf = nullptr;

        if (b_strided) {
            cudaMalloc(&b_buf, nrows * sizeof(ValueType));
            cudaMemcpy2D(b_buf, sizeof(ValueType), b->get_const_values(),
                         b->get_stride() * sizeof(ValueType), sizeof(ValueType),
                         nrows, cudaMemcpyDeviceToDevice);
            b_data = b_buf;
        }
        if (x_strided) {
            cudaMalloc(&x_buf, nrows * sizeof(ValueType));
            cudaMemset(x_buf, 0, nrows * sizeof(ValueType));
            x_data = x_buf;
        }

        cudssMatrix_t cudss_b = nullptr;
        cudssMatrix_t cudss_x = nullptr;

        GKO_ASSERT_NO_CUDSS_ERRORS(cudssMatrixCreateDn(
            &cudss_b, nrows, 1, nrows, b_data, cuda_data_type<ValueType>(),
            CUDSS_LAYOUT_COL_MAJOR));
        GKO_ASSERT_NO_CUDSS_ERRORS(cudssMatrixCreateDn(
            &cudss_x, nrows, 1, nrows, x_data, cuda_data_type<ValueType>(),
            CUDSS_LAYOUT_COL_MAJOR));

        GKO_ASSERT_NO_CUDSS_ERRORS(cudssExecute(
            solve_state->handle, CUDSS_PHASE_SOLVE, solve_state->config,
            solve_state->data, solve_state->A, cudss_x, cudss_b));

        cudssMatrixDestroy(cudss_b);
        cudssMatrixDestroy(cudss_x);

        if (x_strided) {
            cudaMemcpy2D(x->get_values(), x->get_stride() * sizeof(ValueType),
                         x_buf, sizeof(ValueType), sizeof(ValueType), nrows,
                         cudaMemcpyDeviceToDevice);
            cudaFree(x_buf);
        }
        if (b_buf) {
            cudaFree(b_buf);
        }
    } else {
        GKO_NOT_SUPPORTED(exec);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DIRECT_SOLVE_KERNEL);


}  // namespace direct
}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // GKO_HAVE_CUDSS
