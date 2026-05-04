// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <set>
#include <type_traits>

#include <cuda_runtime.h>
#include <cudss.h>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/extensions/cuda/solver/cudss.hpp>


namespace gko {
namespace ext {
namespace cuda {
namespace solver {
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
struct cuda_data_type_impl {};

#define GKO_EXT_CUDA_DATA_TYPE(_type, _value)           \
    template <>                                         \
    struct cuda_data_type_impl<_type> {                 \
        static constexpr cudaDataType_t value = _value; \
    }

GKO_EXT_CUDA_DATA_TYPE(float, CUDA_R_32F);
GKO_EXT_CUDA_DATA_TYPE(double, CUDA_R_64F);
GKO_EXT_CUDA_DATA_TYPE(std::complex<float>, CUDA_C_32F);
GKO_EXT_CUDA_DATA_TYPE(std::complex<double>, CUDA_C_64F);
GKO_EXT_CUDA_DATA_TYPE(int32, CUDA_R_32I);
GKO_EXT_CUDA_DATA_TYPE(int64, CUDA_R_64I);

#undef GKO_EXT_CUDA_DATA_TYPE

template <typename T>
constexpr cudaDataType_t cuda_data_type()
{
    return cuda_data_type_impl<T>::value;
}


}  // anonymous namespace


template <typename ValueType, typename IndexType>
struct CuDss<ValueType, IndexType>::state {
    cudssHandle_t handle = nullptr;
    cudssConfig_t config = nullptr;
    cudssData_t data = nullptr;
    cudssMatrix_t A = nullptr;
    cudaStream_t stream = nullptr;

    ~state()
    {
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


template <typename ValueType, typename IndexType>
CuDss<ValueType, IndexType>::CuDss(std::shared_ptr<const Executor> exec)
    : EnableLinOp<CuDss>{exec}
{}


template <typename ValueType, typename IndexType>
CuDss<ValueType, IndexType>::CuDss(const CuDss& other)
    : EnableLinOp<CuDss>{other.get_executor()}
{
    *this = other;
}


template <typename ValueType, typename IndexType>
CuDss<ValueType, IndexType>::CuDss(CuDss&& other) noexcept
    : EnableLinOp<CuDss>{other.get_executor()}
{
    *this = std::move(other);
}


template <typename ValueType, typename IndexType>
CuDss<ValueType, IndexType>& CuDss<ValueType, IndexType>::operator=(
    const CuDss& other)
{
    if (this != &other) {
        EnableLinOp<CuDss>::operator=(other);
        system_matrix_ = other.system_matrix_;
        state_ = other.state_;
    }
    return *this;
}


template <typename ValueType, typename IndexType>
CuDss<ValueType, IndexType>& CuDss<ValueType, IndexType>::operator=(
    CuDss&& other) noexcept
{
    if (this != &other) {
        EnableLinOp<CuDss>::operator=(std::move(other));
        system_matrix_ = std::move(other.system_matrix_);
        state_ = std::move(other.state_);
    }
    return *this;
}


template <typename ValueType, typename IndexType>
CuDss<ValueType, IndexType>::CuDss(const Factory* factory,
                                   std::shared_ptr<const LinOp> system_matrix)
    : EnableLinOp<CuDss>{factory->get_executor(), system_matrix->get_size()}
{
    const auto exec = this->get_executor();
    auto cuda_exec = std::dynamic_pointer_cast<const CudaExecutor>(exec);
    if (!cuda_exec) {
        GKO_NOT_SUPPORTED(exec);
    }

    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);

    using CsrType = matrix::Csr<ValueType, IndexType>;
    auto csr = copy_and_convert_to<CsrType>(exec, system_matrix);
    system_matrix_ = csr;

    const auto& params = factory->get_parameters();
    auto st = std::make_shared<state>();

    GKO_ASSERT_NO_CUDSS_ERRORS(cudssCreate(&st->handle));
    st->stream = cuda_exec->get_stream();
    GKO_ASSERT_NO_CUDSS_ERRORS(cudssSetStream(st->handle, st->stream));

    GKO_ASSERT_NO_CUDSS_ERRORS(cudssConfigCreate(&st->config));

    auto reorder_alg = static_cast<cudssAlgType_t>(params.reordering_alg);
    GKO_ASSERT_NO_CUDSS_ERRORS(
        cudssConfigSet(st->config, CUDSS_CONFIG_REORDERING_ALG, &reorder_alg,
                       sizeof(reorder_alg)));

    if (params.hybrid_execute) {
        int hybrid_mode = 1;
        GKO_ASSERT_NO_CUDSS_ERRORS(
            cudssConfigSet(st->config, CUDSS_CONFIG_HYBRID_EXECUTE_MODE,
                           &hybrid_mode, sizeof(hybrid_mode)));
    }

    if (params.hybrid_memory) {
        int hybrid_mode = 1;
        GKO_ASSERT_NO_CUDSS_ERRORS(
            cudssConfigSet(st->config, CUDSS_CONFIG_HYBRID_MODE, &hybrid_mode,
                           sizeof(hybrid_mode)));
    }

    GKO_ASSERT_NO_CUDSS_ERRORS(cudssDataCreate(st->handle, &st->data));

    const auto nrows = static_cast<int64_t>(csr->get_size()[0]);
    const auto ncols = static_cast<int64_t>(csr->get_size()[1]);
    const auto nnz = static_cast<int64_t>(csr->get_num_stored_elements());

    auto mtype = static_cast<cudssMatrixType_t>(params.matrix_type);
    auto mview = static_cast<cudssMatrixViewType_t>(params.matrix_view);

    GKO_ASSERT_NO_CUDSS_ERRORS(cudssMatrixCreateCsr(
        &st->A, nrows, ncols, nnz,
        const_cast<IndexType*>(csr->get_const_row_ptrs()), nullptr,
        const_cast<IndexType*>(csr->get_const_col_idxs()),
        const_cast<ValueType*>(csr->get_const_values()),
        cuda_data_type<IndexType>(), cuda_data_type<ValueType>(), mtype, mview,
        CUDSS_BASE_ZERO));

    cudssMatrix_t tmp_b = nullptr;
    cudssMatrix_t tmp_x = nullptr;

    GKO_ASSERT_NO_CUDSS_ERRORS(cudssExecute(st->handle, CUDSS_PHASE_ANALYSIS,
                                            st->config, st->data, st->A, tmp_x,
                                            tmp_b));

    GKO_ASSERT_NO_CUDSS_ERRORS(
        cudssExecute(st->handle, CUDSS_PHASE_FACTORIZATION, st->config,
                     st->data, st->A, tmp_x, tmp_b));

    state_ = std::move(st);
}


template <typename ValueType, typename IndexType>
void CuDss<ValueType, IndexType>::refactorize(
    std::shared_ptr<const LinOp> new_matrix)
{
    const auto exec = this->get_executor();
    using CsrType = matrix::Csr<ValueType, IndexType>;
    auto csr = copy_and_convert_to<CsrType>(exec, new_matrix);

    GKO_ASSERT_EQUAL_DIMENSIONS(csr, system_matrix_);
    const auto old_csr = dynamic_cast<const CsrType*>(system_matrix_.get());
    GKO_ASSERT(old_csr);
    GKO_ASSERT(csr->get_num_stored_elements() ==
               old_csr->get_num_stored_elements());

    system_matrix_ = csr;

    const auto nrows = static_cast<int64_t>(csr->get_size()[0]);
    const auto ncols = static_cast<int64_t>(csr->get_size()[1]);
    const auto nnz = static_cast<int64_t>(csr->get_num_stored_elements());

    if (state_->A) {
        cudssMatrixDestroy(state_->A);
        state_->A = nullptr;
    }

    const auto& params = this->get_parameters();
    auto mtype = static_cast<cudssMatrixType_t>(params.matrix_type);
    auto mview = static_cast<cudssMatrixViewType_t>(params.matrix_view);

    GKO_ASSERT_NO_CUDSS_ERRORS(cudssMatrixCreateCsr(
        &state_->A, nrows, ncols, nnz,
        const_cast<IndexType*>(csr->get_const_row_ptrs()), nullptr,
        const_cast<IndexType*>(csr->get_const_col_idxs()),
        const_cast<ValueType*>(csr->get_const_values()),
        cuda_data_type<IndexType>(), cuda_data_type<ValueType>(), mtype, mview,
        CUDSS_BASE_ZERO));

    cudssMatrix_t tmp_b = nullptr;
    cudssMatrix_t tmp_x = nullptr;

    // Re-run numeric factorization only — symbolic analysis is reused
    GKO_ASSERT_NO_CUDSS_ERRORS(
        cudssExecute(state_->handle, CUDSS_PHASE_REFACTORIZATION,
                     state_->config, state_->data, state_->A, tmp_x, tmp_b));
}


template <typename ValueType, typename IndexType>
void CuDss<ValueType, IndexType>::apply_impl(const LinOp* b, LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            using Dense = matrix::Dense<ValueType>;
            const auto exec = this->get_executor();
            const auto nrhs = dense_b->get_size()[1];
            if (nrhs <= 1) {
                const auto nrows = dense_b->get_size()[0];
                const auto nrows_i64 = static_cast<int64_t>(nrows);

                if (nrows == 0) {
                    return;
                }

                const bool b_strided =
                    (dense_b->get_stride() != dense_b->get_size()[1]);
                const bool x_strided =
                    (dense_x->get_stride() != dense_x->get_size()[1]);

                ValueType* b_data =
                    const_cast<ValueType*>(dense_b->get_const_values());
                ValueType* x_data = dense_x->get_values();
                std::unique_ptr<Dense> b_buf;
                std::unique_ptr<Dense> x_buf;

                if (b_strided) {
                    b_buf = Dense::create(exec, dim<2>{nrows, 1});
                    auto mut_b = const_cast<std::remove_const_t<
                        std::remove_pointer_t<decltype(dense_b)>>*>(dense_b);
                    mut_b->create_submatrix(span{0, nrows}, span{0, 1})
                        ->convert_to(b_buf);
                    b_data = b_buf->get_values();
                }
                if (x_strided) {
                    x_buf = Dense::create(exec, dim<2>{nrows, 1});
                    x_buf->fill(zero<ValueType>());
                    x_data = x_buf->get_values();
                }

                cudssMatrix_t cudss_b = nullptr;
                cudssMatrix_t cudss_x = nullptr;

                GKO_ASSERT_NO_CUDSS_ERRORS(cudssMatrixCreateDn(
                    &cudss_b, nrows_i64, 1, nrows_i64, b_data,
                    cuda_data_type<ValueType>(), CUDSS_LAYOUT_COL_MAJOR));
                GKO_ASSERT_NO_CUDSS_ERRORS(cudssMatrixCreateDn(
                    &cudss_x, nrows_i64, 1, nrows_i64, x_data,
                    cuda_data_type<ValueType>(), CUDSS_LAYOUT_COL_MAJOR));

                GKO_ASSERT_NO_CUDSS_ERRORS(cudssExecute(
                    state_->handle, CUDSS_PHASE_SOLVE, state_->config,
                    state_->data, state_->A, cudss_x, cudss_b));

                GKO_ASSERT_NO_CUDSS_ERRORS(cudssMatrixDestroy(cudss_b));
                GKO_ASSERT_NO_CUDSS_ERRORS(cudssMatrixDestroy(cudss_x));

                if (x_strided) {
                    dense_x->create_submatrix(span{0, nrows}, span{0, 1})
                        ->copy_from(x_buf);
                }
            } else {
                const auto nrows = dense_b->get_size()[0];
                auto tmp_b = Dense::create(exec, dim<2>{nrows, 1});
                auto tmp_x = Dense::create(exec, dim<2>{nrows, 1});
                auto mut_b = const_cast<std::remove_const_t<
                    std::remove_pointer_t<decltype(dense_b)>>*>(dense_b);
                for (size_type j = 0; j < nrhs; ++j) {
                    mut_b->create_submatrix(span{0, nrows}, span{j, j + 1})
                        ->convert_to(tmp_b);
                    this->apply_impl(tmp_b.get(), tmp_x.get());
                    dense_x->create_submatrix(span{0, nrows}, span{j, j + 1})
                        ->copy_from(tmp_x);
                }
            }
        },
        b, x);
}


template <typename ValueType, typename IndexType>
void CuDss<ValueType, IndexType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                             const LinOp* beta, LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            auto tmp = dense_x->clone();
            this->apply_impl(dense_b, tmp.get());
            dense_x->scale(dense_beta);
            dense_x->add_scaled(dense_alpha, tmp);
        },
        alpha, b, beta, x);
}


template <typename ValueType, typename IndexType>
typename CuDss<ValueType, IndexType>::parameters_type
CuDss<ValueType, IndexType>::parse(const config::pnode& config,
                                   const config::registry& context,
                                   const config::type_descriptor& td_for_child)
{
    auto params = CuDss::build();
    // config_check_decorator is only available in core, so we manually
    // check for unknown keys here.
    const std::set<std::string> allowed_keys = {
        "type",           "value_type",     "matrix_type",  "matrix_view",
        "reordering_alg", "hybrid_execute", "hybrid_memory"};
    if (config.get_tag() == config::pnode::tag_t::map) {
        for (const auto& [key, _] : config.get_map()) {
            GKO_THROW_IF_INVALID(allowed_keys.count(key),
                                 key + " is not an allowed key.");
        }
    }
    if (const auto& obj = config.get("matrix_type"); obj) {
        params.with_matrix_type(static_cast<int>(obj.get_integer()));
    }
    if (const auto& obj = config.get("matrix_view"); obj) {
        params.with_matrix_view(static_cast<int>(obj.get_integer()));
    }
    if (const auto& obj = config.get("reordering_alg"); obj) {
        params.with_reordering_alg(static_cast<int>(obj.get_integer()));
    }
    if (const auto& obj = config.get("hybrid_execute"); obj) {
        params.with_hybrid_execute(obj.get_boolean());
    }
    if (const auto& obj = config.get("hybrid_memory"); obj) {
        params.with_hybrid_memory(obj.get_boolean());
    }
    return params;
}


template <typename ValueType, typename IndexType>
config::configuration_map CuDss<ValueType, IndexType>::get_config_map()
{
    return {{"ext::cuda::solver::CuDss",
             [](const config::pnode& config, const config::registry& context,
                config::type_descriptor td)
                 -> deferred_factory_parameter<LinOpFactory> {
                 return CuDss<ValueType, IndexType>::parse(config, context, td);
             }}};
}


#define GKO_DECLARE_CUDSS(ValueType, IndexType) \
    class CuDss<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE_BASE(GKO_DECLARE_CUDSS);


}  // namespace solver
}  // namespace cuda
}  // namespace ext
}  // namespace gko
