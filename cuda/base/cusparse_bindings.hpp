// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CUDA_BASE_CUSPARSE_BINDINGS_HPP_
#define GKO_CUDA_BASE_CUSPARSE_BINDINGS_HPP_


#include <cuda.h>
#include <cusparse.h>

#include <ginkgo/core/base/exception_helpers.hpp>

#include "common/cuda_hip/base/types.hpp"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The CUSPARSE namespace.
 *
 * @ingroup cusparse
 */
namespace cusparse {
/**
 * @brief The detail namespace.
 *
 * @ingroup detail
 */
namespace detail {


template <typename... Args>
inline int64 not_implemented(Args...)
{
    return static_cast<int64>(CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
}


}  // namespace detail


template <typename ValueType, typename IndexType>
struct is_supported : std::false_type {};

template <>
struct is_supported<float, int32> : std::true_type {};

template <>
struct is_supported<double, int32> : std::true_type {};

template <>
struct is_supported<std::complex<float>, int32> : std::true_type {};

template <>
struct is_supported<std::complex<double>, int32> : std::true_type {};


template <typename ValueType>
inline void spmv_buffersize(cusparseHandle_t handle, cusparseOperation_t opA,
                            const ValueType* alpha,
                            const cusparseSpMatDescr_t matA,
                            const cusparseDnVecDescr_t vecX,
                            const ValueType* beta,
                            const cusparseDnVecDescr_t vecY,
                            cusparseSpMVAlg_t alg, size_type* bufferSize)
{
    constexpr auto value_type = cuda_data_type<ValueType>();
    cusparseSpMV_bufferSize(handle, opA, alpha, matA, vecX, beta, vecY,
                            value_type, alg, bufferSize);
}

template <typename ValueType>
inline void spmv(cusparseHandle_t handle, cusparseOperation_t opA,
                 const ValueType* alpha, const cusparseSpMatDescr_t matA,
                 const cusparseDnVecDescr_t vecX, const ValueType* beta,
                 const cusparseDnVecDescr_t vecY, cusparseSpMVAlg_t alg,
                 void* externalBuffer)
{
    constexpr auto value_type = cuda_data_type<ValueType>();
    cusparseSpMV(handle, opA, alpha, matA, vecX, beta, vecY, value_type, alg,
                 externalBuffer);
}


template <typename ValueType>
inline void spmm_buffersize(cusparseHandle_t handle, cusparseOperation_t opB,
                            cusparseOperation_t opA, const ValueType* alpha,
                            const cusparseSpMatDescr_t matA,
                            const cusparseDnMatDescr_t vecX,
                            const ValueType* beta,
                            const cusparseDnMatDescr_t vecY,
                            cusparseSpMMAlg_t alg, size_type* bufferSize)
{
    constexpr auto value_type = cuda_data_type<ValueType>();
    cusparseSpMM_bufferSize(handle, opA, opB, alpha, matA, vecX, beta, vecY,
                            value_type, alg, bufferSize);
}

template <typename ValueType>
inline void spmm(cusparseHandle_t handle, cusparseOperation_t opA,
                 cusparseOperation_t opB, const ValueType* alpha,
                 const cusparseSpMatDescr_t matA,
                 const cusparseDnMatDescr_t vecX, const ValueType* beta,
                 const cusparseDnMatDescr_t vecY, cusparseSpMMAlg_t alg,
                 void* externalBuffer)
{
    constexpr auto value_type = cuda_data_type<ValueType>();
    cusparseSpMM(handle, opA, opB, alpha, matA, vecX, beta, vecY, value_type,
                 alg, externalBuffer);
}


#if defined(CUDA_VERSION) && (CUDA_VERSION < 11021)


template <typename ValueType, typename IndexType>
inline void spmv(cusparseHandle_t handle, cusparseAlgMode_t alg,
                 cusparseOperation_t transA, IndexType m, IndexType n,
                 IndexType nnz, const ValueType* alpha,
                 const cusparseMatDescr_t descrA, const ValueType* csrValA,
                 const IndexType* csrRowPtrA, const IndexType* csrColIndA,
                 const ValueType* x, const ValueType* beta, ValueType* y,
                 void* buffer) GKO_NOT_IMPLEMENTED;

#define GKO_BIND_CUSPARSE_SPMV(ValueType)                                      \
    template <>                                                                \
    inline void spmv<ValueType, int32>(                                        \
        cusparseHandle_t handle, cusparseAlgMode_t alg,                        \
        cusparseOperation_t transA, int32 m, int32 n, int32 nnz,               \
        const ValueType* alpha, const cusparseMatDescr_t descrA,               \
        const ValueType* csrValA, const int32* csrRowPtrA,                     \
        const int32* csrColIndA, const ValueType* x, const ValueType* beta,    \
        ValueType* y, void* buffer)                                            \
    {                                                                          \
        auto data_type = gko::kernels::cuda::cuda_data_type<ValueType>();      \
        if (data_type == CUDA_C_8U) {                                          \
            GKO_NOT_IMPLEMENTED;                                               \
        }                                                                      \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseCsrmvEx(                         \
            handle, alg, transA, m, n, nnz, alpha, data_type, descrA, csrValA, \
            data_type, csrRowPtrA, csrColIndA, x, data_type, beta, data_type,  \
            y, data_type, data_type, buffer));                                 \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE_SPMV(float);
GKO_BIND_CUSPARSE_SPMV(double);
GKO_BIND_CUSPARSE_SPMV(std::complex<float>);
GKO_BIND_CUSPARSE_SPMV(std::complex<double>);


#undef GKO_BIND_CUSPARSE_SPMV


template <typename ValueType, typename IndexType>
inline void spmv_buffersize(cusparseHandle_t handle, cusparseAlgMode_t alg,
                            cusparseOperation_t transA, IndexType m,
                            IndexType n, IndexType nnz, const ValueType* alpha,
                            const cusparseMatDescr_t descrA,
                            const ValueType* csrValA,
                            const IndexType* csrRowPtrA,
                            const IndexType* csrColIndA, const ValueType* x,
                            const ValueType* beta, ValueType* y,
                            size_type* bufferSizeInBytes) GKO_NOT_IMPLEMENTED;

#define GKO_BIND_CUSPARSE_SPMV_BUFFERSIZE(ValueType)                           \
    template <>                                                                \
    inline void spmv_buffersize<ValueType, int32>(                             \
        cusparseHandle_t handle, cusparseAlgMode_t alg,                        \
        cusparseOperation_t transA, int32 m, int32 n, int32 nnz,               \
        const ValueType* alpha, const cusparseMatDescr_t descrA,               \
        const ValueType* csrValA, const int32* csrRowPtrA,                     \
        const int32* csrColIndA, const ValueType* x, const ValueType* beta,    \
        ValueType* y, size_type* bufferSizeInBytes)                            \
    {                                                                          \
        auto data_type = gko::kernels::cuda::cuda_data_type<ValueType>();      \
        if (data_type == CUDA_C_8U) {                                          \
            GKO_NOT_IMPLEMENTED;                                               \
        }                                                                      \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseCsrmvEx_bufferSize(              \
            handle, alg, transA, m, n, nnz, alpha, data_type, descrA, csrValA, \
            data_type, csrRowPtrA, csrColIndA, x, data_type, beta, data_type,  \
            y, data_type, data_type, bufferSizeInBytes));                      \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE_SPMV_BUFFERSIZE(float);
GKO_BIND_CUSPARSE_SPMV_BUFFERSIZE(double);
GKO_BIND_CUSPARSE_SPMV_BUFFERSIZE(std::complex<float>);
GKO_BIND_CUSPARSE_SPMV_BUFFERSIZE(std::complex<double>);


#undef GKO_BIND_CUSPARSE_SPMV_BUFFERSIZE


#endif  // defined(CUDA_VERSION) && (CUDA_VERSION < 11021)


template <typename ValueType>
void spgemm_work_estimation(cusparseHandle_t handle, const ValueType* alpha,
                            cusparseSpMatDescr_t a_descr,
                            cusparseSpMatDescr_t b_descr, const ValueType* beta,
                            cusparseSpMatDescr_t c_descr,
                            cusparseSpGEMMDescr_t spgemm_descr,
                            size_type& buffer1_size, void* buffer1)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseSpGEMM_workEstimation(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, a_descr, b_descr, beta,
        c_descr, cuda_data_type<ValueType>(), CUSPARSE_SPGEMM_DEFAULT,
        spgemm_descr, &buffer1_size, buffer1));
}


template <typename ValueType>
void spgemm_compute(cusparseHandle_t handle, const ValueType* alpha,
                    cusparseSpMatDescr_t a_descr, cusparseSpMatDescr_t b_descr,
                    const ValueType* beta, cusparseSpMatDescr_t c_descr,
                    cusparseSpGEMMDescr_t spgemm_descr, void* buffer1,
                    size_type& buffer2_size, void* buffer2)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseSpGEMM_compute(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, a_descr, b_descr, beta,
        c_descr, cuda_data_type<ValueType>(), CUSPARSE_SPGEMM_DEFAULT,
        spgemm_descr, &buffer2_size, buffer2));
}


template <typename ValueType>
void spgemm_copy(cusparseHandle_t handle, const ValueType* alpha,
                 cusparseSpMatDescr_t a_descr, cusparseSpMatDescr_t b_descr,
                 const ValueType* beta, cusparseSpMatDescr_t c_descr,
                 cusparseSpGEMMDescr_t spgemm_descr)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(
        cusparseSpGEMM_copy(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, a_descr,
                            b_descr, beta, c_descr, cuda_data_type<ValueType>(),
                            CUSPARSE_SPGEMM_DEFAULT, spgemm_descr));
}


inline size_type sparse_matrix_nnz(cusparseSpMatDescr_t descr)
{
    int64 dummy1{};
    int64 dummy2{};
    int64 nnz{};
    cusparseSpMatGetSize(descr, &dummy1, &dummy2, &nnz);
    return static_cast<size_type>(nnz);
}


template <typename ValueType, typename IndexType>
void csr_set_pointers(cusparseSpMatDescr_t descr, IndexType* row_ptrs,
                      IndexType* col_idxs, ValueType* vals)
{
    cusparseCsrSetPointers(descr, row_ptrs, col_idxs, vals);
}


template <typename ValueType, typename IndexType>
inline void transpose_buffersize(
    cusparseHandle_t handle, size_type m, size_type n, size_type nnz,
    const ValueType* OrigValA, const IndexType* OrigRowPtrA,
    const IndexType* OrigColIndA, ValueType* TransValA, IndexType* TransRowPtrA,
    IndexType* TransColIndA, cudaDataType_t valType,
    cusparseAction_t copyValues, cusparseIndexBase_t idxBase,
    cusparseCsr2CscAlg_t alg, size_type* buffer_size) GKO_NOT_IMPLEMENTED;

#define GKO_BIND_CUSPARSE_TRANSPOSE_BUFFERSIZE32(ValueType)                   \
    template <>                                                               \
    inline void transpose_buffersize<ValueType, int32>(                       \
        cusparseHandle_t handle, size_type m, size_type n, size_type nnz,     \
        const ValueType* OrigValA, const int32* OrigRowPtrA,                  \
        const int32* OrigColIndA, ValueType* TransValA, int32* TransRowPtrA,  \
        int32* TransColIndA, cudaDataType_t valType,                          \
        cusparseAction_t copyValues, cusparseIndexBase_t idxBase,             \
        cusparseCsr2CscAlg_t alg, size_type* buffer_size)                     \
    {                                                                         \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseCsr2cscEx2_bufferSize(          \
            handle, m, n, nnz, OrigValA, OrigRowPtrA, OrigColIndA, TransValA, \
            TransRowPtrA, TransColIndA, valType, copyValues, idxBase, alg,    \
            buffer_size));                                                    \
    }                                                                         \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE_TRANSPOSE_BUFFERSIZE32(float);
GKO_BIND_CUSPARSE_TRANSPOSE_BUFFERSIZE32(double);
GKO_BIND_CUSPARSE_TRANSPOSE_BUFFERSIZE32(std::complex<float>);
GKO_BIND_CUSPARSE_TRANSPOSE_BUFFERSIZE32(std::complex<double>);

template <typename ValueType, typename IndexType>
inline void transpose(cusparseHandle_t handle, size_type m, size_type n,
                      size_type nnz, const ValueType* OrigValA,
                      const IndexType* OrigRowPtrA,
                      const IndexType* OrigColIndA, ValueType* TransValA,
                      IndexType* TransRowPtrA, IndexType* TransColIndA,
                      cudaDataType_t valType, cusparseAction_t copyValues,
                      cusparseIndexBase_t idxBase, cusparseCsr2CscAlg_t alg,
                      void* buffer) GKO_NOT_IMPLEMENTED;

#define GKO_BIND_CUSPARSE_TRANSPOSE32(ValueType)                              \
    template <>                                                               \
    inline void transpose<ValueType, int32>(                                  \
        cusparseHandle_t handle, size_type m, size_type n, size_type nnz,     \
        const ValueType* OrigValA, const int32* OrigRowPtrA,                  \
        const int32* OrigColIndA, ValueType* TransValA, int32* TransRowPtrA,  \
        int32* TransColIndA, cudaDataType_t valType,                          \
        cusparseAction_t copyValues, cusparseIndexBase_t idxBase,             \
        cusparseCsr2CscAlg_t alg, void* buffer)                               \
    {                                                                         \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseCsr2cscEx2(                     \
            handle, m, n, nnz, OrigValA, OrigRowPtrA, OrigColIndA, TransValA, \
            TransRowPtrA, TransColIndA, valType, copyValues, idxBase, alg,    \
            buffer));                                                         \
    }                                                                         \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE_TRANSPOSE32(float);
GKO_BIND_CUSPARSE_TRANSPOSE32(double);
GKO_BIND_CUSPARSE_TRANSPOSE32(std::complex<float>);
GKO_BIND_CUSPARSE_TRANSPOSE32(std::complex<double>);


inline cusparseMatDescr_t create_mat_descr()
{
    cusparseMatDescr_t descr{};
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseCreateMatDescr(&descr));
    GKO_ASSERT_NO_CUSPARSE_ERRORS(
        cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
    GKO_ASSERT_NO_CUSPARSE_ERRORS(
        cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    GKO_ASSERT_NO_CUSPARSE_ERRORS(
        cusparseSetMatDiagType(descr, CUSPARSE_DIAG_TYPE_NON_UNIT));
    return descr;
}


inline void set_mat_fill_mode(cusparseMatDescr_t descr,
                              cusparseFillMode_t fill_mode)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseSetMatFillMode(descr, fill_mode));
}


inline void set_mat_diag_type(cusparseMatDescr_t descr,
                              cusparseDiagType_t diag_type)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseSetMatDiagType(descr, diag_type));
}


inline void destroy(cusparseMatDescr_t descr)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseDestroyMatDescr(descr));
}


inline cusparseSpGEMMDescr_t create_spgemm_descr()
{
    cusparseSpGEMMDescr_t descr{};
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseSpGEMM_createDescr(&descr));
    return descr;
}


inline void destroy(cusparseSpGEMMDescr_t info)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseSpGEMM_destroyDescr(info));
}


template <typename ValueType>
inline cusparseDnVecDescr_t create_dnvec(int64 size, ValueType* values)
{
    cusparseDnVecDescr_t descr{};
    constexpr auto value_type = cuda_data_type<ValueType>();
    GKO_ASSERT_NO_CUSPARSE_ERRORS(
        cusparseCreateDnVec(&descr, size, values, value_type));
    return descr;
}


template <typename ValueType>
inline cusparseDnMatDescr_t create_dnmat(gko::dim<2> size, size_type stride,
                                         ValueType* values)
{
    cusparseDnMatDescr_t descr{};
    constexpr auto value_type = cuda_data_type<ValueType>();
    GKO_ASSERT_NO_CUSPARSE_ERRORS(
        cusparseCreateDnMat(&descr, size[0], size[1], stride, values,
                            value_type, CUSPARSE_ORDER_ROW));
    return descr;
}


inline void destroy(cusparseDnVecDescr_t descr)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseDestroyDnVec(descr));
}


inline void destroy(cusparseDnMatDescr_t descr)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseDestroyDnMat(descr));
}


template <typename ValueType, typename IndexType>
inline cusparseSpVecDescr_t create_spvec(int64 size, int64 nnz,
                                         IndexType* indices, ValueType* values)
{
    cusparseSpVecDescr_t descr{};
    constexpr auto index_type = cusparse_index_type<IndexType>();
    constexpr auto value_type = cuda_data_type<ValueType>();
    GKO_ASSERT_NO_CUSPARSE_ERRORS(
        cusparseCreateSpVec(&descr, size, nnz, indices, values, index_type,
                            CUSPARSE_INDEX_BASE_ZERO, value_type));
    return descr;
}


inline void destroy(cusparseSpVecDescr_t descr)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseDestroySpVec(descr));
}


template <typename IndexType, typename ValueType>
inline cusparseSpMatDescr_t create_csr(int64 rows, int64 cols, int64 nnz,
                                       IndexType* csrRowOffsets,
                                       IndexType* csrColInd,
                                       ValueType* csrValues)
{
    cusparseSpMatDescr_t descr{};
    constexpr auto index_type = cusparse_index_type<IndexType>();
    constexpr auto value_type = cuda_data_type<ValueType>();
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseCreateCsr(
        &descr, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues,
        index_type, index_type, CUSPARSE_INDEX_BASE_ZERO, value_type));
    return descr;
}


inline void destroy(cusparseSpMatDescr_t descr)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseDestroySpMat(descr));
}


#if defined(CUDA_VERSION) && (CUDA_VERSION >= 11031)


template <typename AttribType>
inline void set_attribute(cusparseSpMatDescr_t desc,
                          cusparseSpMatAttribute_t attr, AttribType val)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(
        cusparseSpMatSetAttribute(desc, attr, &val, sizeof(val)));
}


inline cusparseSpSMDescr_t create_spsm_descr()
{
    cusparseSpSMDescr_t desc{};
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseSpSM_createDescr(&desc));
    return desc;
}


inline void destroy(cusparseSpSMDescr_t info)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseSpSM_destroyDescr(info));
}


#endif  // CUDA_VERSION >= 11031


#if defined(CUDA_VERSION) && (CUDA_VERSION < 11031)


inline csrsm2Info_t create_solve_info()
{
    csrsm2Info_t info{};
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseCreateCsrsm2Info(&info));
    return info;
}


inline void destroy(csrsm2Info_t info)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseDestroyCsrsm2Info(info));
}


#endif  // defined(CUDA_VERSION) && (CUDA_VERSION < 11031)


GKO_BEGIN_DISABLE_DEPRECATION_WARNINGS
inline csrilu02Info_t create_ilu0_info()
{
    csrilu02Info_t info{};
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseCreateCsrilu02Info(&info));
    return info;
}


inline void destroy_ilu0_info(csrilu02Info_t info)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseDestroyCsrilu02Info(info));
}


inline csric02Info_t create_ic0_info()
{
    csric02Info_t info{};
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseCreateCsric02Info(&info));
    return info;
}


inline void destroy_ic0_info(csric02Info_t info)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseDestroyCsric02Info(info));
}
GKO_END_DISABLE_DEPRECATION_WARNINGS


#if (defined(CUDA_VERSION) && (CUDA_VERSION < 11031))


#define GKO_BIND_CUSPARSE32_BUFFERSIZEEXT(ValueType, CusparseName)            \
    inline void buffer_size_ext(                                              \
        cusparseHandle_t handle, int algo, cusparseOperation_t trans1,        \
        cusparseOperation_t trans2, size_type m, size_type n, size_type nnz,  \
        ValueType one, const cusparseMatDescr_t descr,                        \
        const ValueType* csrVal, const int32* csrRowPtr,                      \
        const int32* csrColInd, const ValueType* rhs, int32 sol_size,         \
        csrsm2Info_t factor_info, cusparseSolvePolicy_t policy,               \
        size_type* factor_work_size)                                          \
    {                                                                         \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(                                        \
            CusparseName(handle, algo, trans1, trans2, m, n, nnz,             \
                         as_culibs_type(&one), descr, as_culibs_type(csrVal), \
                         csrRowPtr, csrColInd, as_culibs_type(rhs), sol_size, \
                         factor_info, policy, factor_work_size));             \
    }                                                                         \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")

#define GKO_BIND_CUSPARSE64_BUFFERSIZEEXT(ValueType, CusparseName)           \
    inline void buffer_size_ext(                                             \
        cusparseHandle_t handle, int algo, cusparseOperation_t trans1,       \
        cusparseOperation_t trans2, size_type m, size_type n, size_type nnz, \
        ValueType one, const cusparseMatDescr_t descr,                       \
        const ValueType* csrVal, const int64* csrRowPtr,                     \
        const int64* csrColInd, const ValueType* rhs, int64 sol_size,        \
        csrsm2Info_t factor_info, cusparseSolvePolicy_t policy,              \
        size_type* factor_work_size) GKO_NOT_IMPLEMENTED;                    \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE32_BUFFERSIZEEXT(float, cusparseScsrsm2_bufferSizeExt);
GKO_BIND_CUSPARSE32_BUFFERSIZEEXT(double, cusparseDcsrsm2_bufferSizeExt);
GKO_BIND_CUSPARSE32_BUFFERSIZEEXT(std::complex<float>,
                                  cusparseCcsrsm2_bufferSizeExt);
GKO_BIND_CUSPARSE32_BUFFERSIZEEXT(std::complex<double>,
                                  cusparseZcsrsm2_bufferSizeExt);
GKO_BIND_CUSPARSE64_BUFFERSIZEEXT(float, cusparseScsrsm2_bufferSizeExt);
GKO_BIND_CUSPARSE64_BUFFERSIZEEXT(double, cusparseDcsrsm2_bufferSizeExt);
GKO_BIND_CUSPARSE64_BUFFERSIZEEXT(std::complex<float>,
                                  cusparseCcsrsm2_bufferSizeExt);
GKO_BIND_CUSPARSE64_BUFFERSIZEEXT(std::complex<double>,
                                  cusparseZcsrsm2_bufferSizeExt);
template <typename ValueType>
GKO_BIND_CUSPARSE32_BUFFERSIZEEXT(ValueType, detail::not_implemented);
template <typename ValueType>
GKO_BIND_CUSPARSE64_BUFFERSIZEEXT(ValueType, detail::not_implemented);
#undef GKO_BIND_CUSPARSE32_BUFFERSIZEEXT
#undef GKO_BIND_CUSPARSE64_BUFFERSIZEEXT


#define GKO_BIND_CUSPARSE32_CSRSM2_ANALYSIS(ValueType, CusparseName)          \
    inline void csrsm2_analysis(                                              \
        cusparseHandle_t handle, int algo, cusparseOperation_t trans1,        \
        cusparseOperation_t trans2, size_type m, size_type n, size_type nnz,  \
        ValueType one, const cusparseMatDescr_t descr,                        \
        const ValueType* csrVal, const int32* csrRowPtr,                      \
        const int32* csrColInd, const ValueType* rhs, int32 sol_size,         \
        csrsm2Info_t factor_info, cusparseSolvePolicy_t policy,               \
        void* factor_work_vec)                                                \
    {                                                                         \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(                                        \
            CusparseName(handle, algo, trans1, trans2, m, n, nnz,             \
                         as_culibs_type(&one), descr, as_culibs_type(csrVal), \
                         csrRowPtr, csrColInd, as_culibs_type(rhs), sol_size, \
                         factor_info, policy, factor_work_vec));              \
    }                                                                         \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")

#define GKO_BIND_CUSPARSE64_CSRSM2_ANALYSIS(ValueType, CusparseName)         \
    inline void csrsm2_analysis(                                             \
        cusparseHandle_t handle, int algo, cusparseOperation_t trans1,       \
        cusparseOperation_t trans2, size_type m, size_type n, size_type nnz, \
        ValueType one, const cusparseMatDescr_t descr,                       \
        const ValueType* csrVal, const int64* csrRowPtr,                     \
        const int64* csrColInd, const ValueType* rhs, int64 sol_size,        \
        csrsm2Info_t factor_info, cusparseSolvePolicy_t policy,              \
        void* factor_work_vec) GKO_NOT_IMPLEMENTED;                          \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE32_CSRSM2_ANALYSIS(float, cusparseScsrsm2_analysis);
GKO_BIND_CUSPARSE32_CSRSM2_ANALYSIS(double, cusparseDcsrsm2_analysis);
GKO_BIND_CUSPARSE32_CSRSM2_ANALYSIS(std::complex<float>,
                                    cusparseCcsrsm2_analysis);
GKO_BIND_CUSPARSE32_CSRSM2_ANALYSIS(std::complex<double>,
                                    cusparseZcsrsm2_analysis);
GKO_BIND_CUSPARSE64_CSRSM2_ANALYSIS(float, cusparseScsrsm2_analysis);
GKO_BIND_CUSPARSE64_CSRSM2_ANALYSIS(double, cusparseDcsrsm2_analysis);
GKO_BIND_CUSPARSE64_CSRSM2_ANALYSIS(std::complex<float>,
                                    cusparseCcsrsm2_analysis);
GKO_BIND_CUSPARSE64_CSRSM2_ANALYSIS(std::complex<double>,
                                    cusparseZcsrsm2_analysis);
template <typename ValueType>
GKO_BIND_CUSPARSE32_CSRSM2_ANALYSIS(ValueType, detail::not_implemented);
template <typename ValueType>
GKO_BIND_CUSPARSE64_CSRSM2_ANALYSIS(ValueType, detail::not_implemented);
#undef GKO_BIND_CUSPARSE32_CSRSM2_ANALYSIS
#undef GKO_BIND_CUSPARSE64_CSRSM2_ANALYSIS


#define GKO_BIND_CUSPARSE32_CSRSM2_SOLVE(ValueType, CusparseName)             \
    inline void csrsm2_solve(                                                 \
        cusparseHandle_t handle, int algo, cusparseOperation_t trans1,        \
        cusparseOperation_t trans2, size_type m, size_type n, size_type nnz,  \
        ValueType one, const cusparseMatDescr_t descr,                        \
        const ValueType* csrVal, const int32* csrRowPtr,                      \
        const int32* csrColInd, ValueType* rhs, int32 sol_stride,             \
        csrsm2Info_t factor_info, cusparseSolvePolicy_t policy,               \
        void* factor_work_vec)                                                \
    {                                                                         \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(                                        \
            CusparseName(handle, algo, trans1, trans2, m, n, nnz,             \
                         as_culibs_type(&one), descr, as_culibs_type(csrVal), \
                         csrRowPtr, csrColInd, as_culibs_type(rhs),           \
                         sol_stride, factor_info, policy, factor_work_vec));  \
    }                                                                         \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")

#define GKO_BIND_CUSPARSE64_CSRSM2_SOLVE(ValueType, CusparseName)            \
    inline void csrsm2_solve(                                                \
        cusparseHandle_t handle, int algo, cusparseOperation_t trans1,       \
        cusparseOperation_t trans2, size_type m, size_type n, size_type nnz, \
        ValueType one, const cusparseMatDescr_t descr,                       \
        const ValueType* csrVal, const int64* csrRowPtr,                     \
        const int64* csrColInd, ValueType* rhs, int64 sol_stride,            \
        csrsm2Info_t factor_info, cusparseSolvePolicy_t policy,              \
        void* factor_work_vec) GKO_NOT_IMPLEMENTED;                          \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE32_CSRSM2_SOLVE(float, cusparseScsrsm2_solve);
GKO_BIND_CUSPARSE32_CSRSM2_SOLVE(double, cusparseDcsrsm2_solve);
GKO_BIND_CUSPARSE32_CSRSM2_SOLVE(std::complex<float>, cusparseCcsrsm2_solve);
GKO_BIND_CUSPARSE32_CSRSM2_SOLVE(std::complex<double>, cusparseZcsrsm2_solve);
GKO_BIND_CUSPARSE64_CSRSM2_SOLVE(float, cusparseScsrsm2_solve);
GKO_BIND_CUSPARSE64_CSRSM2_SOLVE(double, cusparseDcsrsm2_solve);
GKO_BIND_CUSPARSE64_CSRSM2_SOLVE(std::complex<float>, cusparseCcsrsm2_solve);
GKO_BIND_CUSPARSE64_CSRSM2_SOLVE(std::complex<double>, cusparseZcsrsm2_solve);
template <typename ValueType>
GKO_BIND_CUSPARSE32_CSRSM2_SOLVE(ValueType, detail::not_implemented);
template <typename ValueType>
GKO_BIND_CUSPARSE64_CSRSM2_SOLVE(ValueType, detail::not_implemented);
#undef GKO_BIND_CUSPARSE32_CSRSM2_SOLVE
#undef GKO_BIND_CUSPARSE64_CSRSM2_SOLVE


#else  // if (defined(CUDA_VERSION) && (CUDA_VERSION >= 11031))


template <typename ValueType>
size_type spsm_buffer_size(cusparseHandle_t handle, cusparseOperation_t op_a,
                           cusparseOperation_t op_b, ValueType alpha,
                           cusparseSpMatDescr_t descr_a,
                           cusparseDnMatDescr_t descr_b,
                           cusparseDnMatDescr_t descr_c, cusparseSpSMAlg_t algo,
                           cusparseSpSMDescr_t spsm_descr)
{
    size_type work_size;
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseSpSM_bufferSize(
        handle, op_a, op_b, &alpha, descr_a, descr_b, descr_c,
        cuda_data_type<ValueType>(), algo, spsm_descr, &work_size));
    return work_size;
}


template <typename ValueType>
void spsm_analysis(cusparseHandle_t handle, cusparseOperation_t op_a,
                   cusparseOperation_t op_b, ValueType alpha,
                   cusparseSpMatDescr_t descr_a, cusparseDnMatDescr_t descr_b,
                   cusparseDnMatDescr_t descr_c, cusparseSpSMAlg_t algo,
                   cusparseSpSMDescr_t spsm_descr, void* work)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseSpSM_analysis(
        handle, op_a, op_b, &alpha, descr_a, descr_b, descr_c,
        cuda_data_type<ValueType>(), algo, spsm_descr, work));
}


template <typename ValueType>
void spsm_solve(cusparseHandle_t handle, cusparseOperation_t op_a,
                cusparseOperation_t op_b, ValueType alpha,
                cusparseSpMatDescr_t descr_a, cusparseDnMatDescr_t descr_b,
                cusparseDnMatDescr_t descr_c, cusparseSpSMAlg_t algo,
                cusparseSpSMDescr_t spsm_descr)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseSpSM_solve(
        handle, op_a, op_b, &alpha, descr_a, descr_b, descr_c,
        cuda_data_type<ValueType>(), algo, spsm_descr));
}


#endif  // (defined(CUDA_VERSION) && (CUDA_VERSION >= 11031))


template <typename IndexType>
void csrsort_buffer_size(cusparseHandle_t handle, IndexType m, IndexType n,
                         IndexType nnz, const IndexType* row_ptrs,
                         const IndexType* col_idxs,
                         size_type& buffer_size) GKO_NOT_IMPLEMENTED;

template <>
inline void csrsort_buffer_size<int32>(cusparseHandle_t handle, int32 m,
                                       int32 n, int32 nnz,
                                       const int32* row_ptrs,
                                       const int32* col_idxs,
                                       size_type& buffer_size)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseXcsrsort_bufferSizeExt(
        handle, m, n, nnz, row_ptrs, col_idxs, &buffer_size));
}


template <typename IndexType>
void csrsort(cusparseHandle_t handle, IndexType m, IndexType n, IndexType nnz,
             const cusparseMatDescr_t descr, const IndexType* row_ptrs,
             IndexType* col_idxs, IndexType* permutation,
             void* buffer) GKO_NOT_IMPLEMENTED;

template <>
inline void csrsort<int32>(cusparseHandle_t handle, int32 m, int32 n, int32 nnz,
                           const cusparseMatDescr_t descr,
                           const int32* row_ptrs, int32* col_idxs,
                           int32* permutation, void* buffer)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseXcsrsort(
        handle, m, n, nnz, descr, row_ptrs, col_idxs, permutation, buffer));
}


inline void gather(cusparseHandle_t handle, cusparseDnVecDescr_t in,
                   cusparseSpVecDescr_t out)
{
    GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseGather(handle, in, out));
}


GKO_BEGIN_DISABLE_DEPRECATION_WARNINGS
template <typename ValueType, typename IndexType>
void ilu0_buffer_size(cusparseHandle_t handle, IndexType m, IndexType nnz,
                      const cusparseMatDescr_t descr, const ValueType* vals,
                      const IndexType* row_ptrs, const IndexType* col_idxs,
                      csrilu02Info_t info,
                      size_type& buffer_size) GKO_NOT_IMPLEMENTED;

#define GKO_BIND_CUSPARSE_ILU0_BUFFER_SIZE(ValueType, CusparseName)          \
    template <>                                                              \
    inline void ilu0_buffer_size<ValueType, int32>(                          \
        cusparseHandle_t handle, int32 m, int32 nnz,                         \
        const cusparseMatDescr_t descr, const ValueType* vals,               \
        const int32* row_ptrs, const int32* col_idxs, csrilu02Info_t info,   \
        size_type& buffer_size)                                              \
    {                                                                        \
        int tmp_buffer_size{};                                               \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(                                       \
            CusparseName(handle, m, nnz, descr,                              \
                         as_culibs_type(const_cast<ValueType*>(vals)),       \
                         row_ptrs, col_idxs, info, &tmp_buffer_size));       \
        buffer_size = tmp_buffer_size;                                       \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE_ILU0_BUFFER_SIZE(float, cusparseScsrilu02_bufferSize);
GKO_BIND_CUSPARSE_ILU0_BUFFER_SIZE(double, cusparseDcsrilu02_bufferSize);
GKO_BIND_CUSPARSE_ILU0_BUFFER_SIZE(std::complex<float>,
                                   cusparseCcsrilu02_bufferSize);
GKO_BIND_CUSPARSE_ILU0_BUFFER_SIZE(std::complex<double>,
                                   cusparseZcsrilu02_bufferSize);

#undef GKO_BIND_CUSPARSE_ILU0_BUFFER_SIZE


template <typename ValueType, typename IndexType>
void ilu0_analysis(cusparseHandle_t handle, IndexType m, IndexType nnz,
                   const cusparseMatDescr_t descr, const ValueType* vals,
                   const IndexType* row_ptrs, const IndexType* col_idxs,
                   csrilu02Info_t info, cusparseSolvePolicy_t policy,
                   void* buffer) GKO_NOT_IMPLEMENTED;

#define GKO_BIND_CUSPARSE_ILU0_ANALYSIS(ValueType, CusparseName)             \
    template <>                                                              \
    inline void ilu0_analysis<ValueType, int32>(                             \
        cusparseHandle_t handle, int32 m, int32 nnz,                         \
        const cusparseMatDescr_t descr, const ValueType* vals,               \
        const int32* row_ptrs, const int32* col_idxs, csrilu02Info_t info,   \
        cusparseSolvePolicy_t policy, void* buffer)                          \
    {                                                                        \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(                                       \
            CusparseName(handle, m, nnz, descr, as_culibs_type(vals),        \
                         row_ptrs, col_idxs, info, policy, buffer));         \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE_ILU0_ANALYSIS(float, cusparseScsrilu02_analysis);
GKO_BIND_CUSPARSE_ILU0_ANALYSIS(double, cusparseDcsrilu02_analysis);
GKO_BIND_CUSPARSE_ILU0_ANALYSIS(std::complex<float>,
                                cusparseCcsrilu02_analysis);
GKO_BIND_CUSPARSE_ILU0_ANALYSIS(std::complex<double>,
                                cusparseZcsrilu02_analysis);

#undef GKO_BIND_CUSPARSE_ILU0_ANALYSIS


template <typename ValueType, typename IndexType>
void ilu0(cusparseHandle_t handle, IndexType m, IndexType nnz,
          const cusparseMatDescr_t descr, ValueType* vals,
          const IndexType* row_ptrs, const IndexType* col_idxs,
          csrilu02Info_t info, cusparseSolvePolicy_t policy,
          void* buffer) GKO_NOT_IMPLEMENTED;

#define GKO_BIND_CUSPARSE_ILU0(ValueType, CusparseName)                      \
    template <>                                                              \
    inline void ilu0<ValueType, int32>(                                      \
        cusparseHandle_t handle, int32 m, int32 nnz,                         \
        const cusparseMatDescr_t descr, ValueType* vals,                     \
        const int32* row_ptrs, const int32* col_idxs, csrilu02Info_t info,   \
        cusparseSolvePolicy_t policy, void* buffer)                          \
    {                                                                        \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(                                       \
            CusparseName(handle, m, nnz, descr, as_culibs_type(vals),        \
                         row_ptrs, col_idxs, info, policy, buffer));         \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE_ILU0(float, cusparseScsrilu02);
GKO_BIND_CUSPARSE_ILU0(double, cusparseDcsrilu02);
GKO_BIND_CUSPARSE_ILU0(std::complex<float>, cusparseCcsrilu02);
GKO_BIND_CUSPARSE_ILU0(std::complex<double>, cusparseZcsrilu02);

#undef GKO_BIND_CUSPARSE_ILU0


template <typename ValueType, typename IndexType>
void ic0_buffer_size(cusparseHandle_t handle, IndexType m, IndexType nnz,
                     const cusparseMatDescr_t descr, const ValueType* vals,
                     const IndexType* row_ptrs, const IndexType* col_idxs,
                     csric02Info_t info,
                     size_type& buffer_size) GKO_NOT_IMPLEMENTED;

#define GKO_BIND_CUSPARSE_IC0_BUFFER_SIZE(ValueType, CusparseName)           \
    template <>                                                              \
    inline void ic0_buffer_size<ValueType, int32>(                           \
        cusparseHandle_t handle, int32 m, int32 nnz,                         \
        const cusparseMatDescr_t descr, const ValueType* vals,               \
        const int32* row_ptrs, const int32* col_idxs, csric02Info_t info,    \
        size_type& buffer_size)                                              \
    {                                                                        \
        int tmp_buffer_size{};                                               \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(                                       \
            CusparseName(handle, m, nnz, descr,                              \
                         as_culibs_type(const_cast<ValueType*>(vals)),       \
                         row_ptrs, col_idxs, info, &tmp_buffer_size));       \
        buffer_size = tmp_buffer_size;                                       \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE_IC0_BUFFER_SIZE(float, cusparseScsric02_bufferSize);
GKO_BIND_CUSPARSE_IC0_BUFFER_SIZE(double, cusparseDcsric02_bufferSize);
GKO_BIND_CUSPARSE_IC0_BUFFER_SIZE(std::complex<float>,
                                  cusparseCcsric02_bufferSize);
GKO_BIND_CUSPARSE_IC0_BUFFER_SIZE(std::complex<double>,
                                  cusparseZcsric02_bufferSize);

#undef GKO_BIND_CUSPARSE_IC0_BUFFER_SIZE


template <typename ValueType, typename IndexType>
void ic0_analysis(cusparseHandle_t handle, IndexType m, IndexType nnz,
                  const cusparseMatDescr_t descr, const ValueType* vals,
                  const IndexType* row_ptrs, const IndexType* col_idxs,
                  csric02Info_t info, cusparseSolvePolicy_t policy,
                  void* buffer) GKO_NOT_IMPLEMENTED;

#define GKO_BIND_CUSPARSE_IC0_ANALYSIS(ValueType, CusparseName)              \
    template <>                                                              \
    inline void ic0_analysis<ValueType, int32>(                              \
        cusparseHandle_t handle, int32 m, int32 nnz,                         \
        const cusparseMatDescr_t descr, const ValueType* vals,               \
        const int32* row_ptrs, const int32* col_idxs, csric02Info_t info,    \
        cusparseSolvePolicy_t policy, void* buffer)                          \
    {                                                                        \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(                                       \
            CusparseName(handle, m, nnz, descr, as_culibs_type(vals),        \
                         row_ptrs, col_idxs, info, policy, buffer));         \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE_IC0_ANALYSIS(float, cusparseScsric02_analysis);
GKO_BIND_CUSPARSE_IC0_ANALYSIS(double, cusparseDcsric02_analysis);
GKO_BIND_CUSPARSE_IC0_ANALYSIS(std::complex<float>, cusparseCcsric02_analysis);
GKO_BIND_CUSPARSE_IC0_ANALYSIS(std::complex<double>, cusparseZcsric02_analysis);

#undef GKO_BIND_CUSPARSE_ILU0_ANALYSIS


template <typename ValueType, typename IndexType>
void ic0(cusparseHandle_t handle, IndexType m, IndexType nnz,
         const cusparseMatDescr_t descr, ValueType* vals,
         const IndexType* row_ptrs, const IndexType* col_idxs,
         csric02Info_t info, cusparseSolvePolicy_t policy,
         void* buffer) GKO_NOT_IMPLEMENTED;

#define GKO_BIND_CUSPARSE_IC0(ValueType, CusparseName)                       \
    template <>                                                              \
    inline void ic0<ValueType, int32>(                                       \
        cusparseHandle_t handle, int32 m, int32 nnz,                         \
        const cusparseMatDescr_t descr, ValueType* vals,                     \
        const int32* row_ptrs, const int32* col_idxs, csric02Info_t info,    \
        cusparseSolvePolicy_t policy, void* buffer)                          \
    {                                                                        \
        GKO_ASSERT_NO_CUSPARSE_ERRORS(                                       \
            CusparseName(handle, m, nnz, descr, as_culibs_type(vals),        \
                         row_ptrs, col_idxs, info, policy, buffer));         \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

GKO_BIND_CUSPARSE_IC0(float, cusparseScsric02);
GKO_BIND_CUSPARSE_IC0(double, cusparseDcsric02);
GKO_BIND_CUSPARSE_IC0(std::complex<float>, cusparseCcsric02);
GKO_BIND_CUSPARSE_IC0(std::complex<double>, cusparseZcsric02);
GKO_END_DISABLE_DEPRECATION_WARNINGS

#undef GKO_BIND_CUSPARSE_IC0


}  // namespace cusparse


namespace sparselib {


using namespace cusparse;


#define SPARSELIB_OPERATION_TRANSPOSE CUSPARSE_OPERATION_TRANSPOSE
#define SPARSELIB_OPERATION_NON_TRANSPOSE CUSPARSE_OPERATION_NON_TRANSPOSE
#define SPARSELIB_SOLVE_POLICY_USE_LEVEL CUSPARSE_SOLVE_POLICY_USE_LEVEL


}  // namespace sparselib
}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CUDA_BASE_CUSPARSE_BINDINGS_HPP_
