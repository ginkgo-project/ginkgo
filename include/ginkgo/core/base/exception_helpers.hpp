// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_EXCEPTION_HELPERS_HPP_
#define GKO_PUBLIC_CORE_BASE_EXCEPTION_HELPERS_HPP_


#include <typeinfo>


#include <ginkgo/core/base/batch_dim.hpp>
#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/name_demangling.hpp>
#include <ginkgo/core/base/utils_helper.hpp>


namespace gko {


/**
 * Adds quotes around the list of expressions passed to the macro.
 *
 * @param ...  a list of expressions
 *
 * @return a C string containing the expression body
 */
#define GKO_QUOTE(...) #__VA_ARGS__


/**
 * Marks a function as not yet implemented.
 *
 * Attempts to call this function will result in a runtime error of type
 * NotImplemented.
 */
#define GKO_NOT_IMPLEMENTED                                                  \
    {                                                                        \
        throw ::gko::NotImplemented(__FILE__, __LINE__, __func__);           \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


/**
 * Marks a function as not compiled.
 *
 * Attempts to call this function will result in a runtime error of type
 * NotCompiled
 *
 * @param _module  the module which should be compiled to enable the function
 */
#define GKO_NOT_COMPILED(_module)                                            \
    {                                                                        \
        throw ::gko::NotCompiled(__FILE__, __LINE__, __func__,               \
                                 GKO_QUOTE(_module));                        \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


namespace detail {


template <typename T, typename T2 = void>
struct dynamic_type_helper {
    static const std::type_info& get(const T& obj) { return typeid(obj); }
};

template <typename T>
struct dynamic_type_helper<T,
                           typename std::enable_if<std::is_pointer<T>::value ||
                                                   have_ownership<T>()>::type> {
    static const std::type_info& get(const T& obj)
    {
        return obj ? typeid(*obj) : typeid(nullptr);
    }
};

template <typename T>
const std::type_info& get_dynamic_type(const T& obj)
{
    return dynamic_type_helper<T>::get(obj);
}


}  // namespace detail


/**
 * Throws a NotSupported exception.
 * This macro sets the correct information about the location of the error
 * and fills the exception with data about _obj, followed by throwing it.
 *
 * @param _obj  the object referenced by NotSupported exception
 */
#define GKO_NOT_SUPPORTED(_obj)                                                \
    {                                                                          \
        throw ::gko::NotSupported(__FILE__, __LINE__, __func__,                \
                                  ::gko::name_demangling::get_type_name(       \
                                      ::gko::detail::get_dynamic_type(_obj))); \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")


namespace detail {


template <typename T>
inline dim<2> get_size(const T& op)
{
    return op->get_size();
}

inline dim<2> get_size(const dim<2>& size) { return size; }


template <typename T>
inline batch_dim<2> get_batch_size(const T& op)
{
    return op->get_size();
}

inline batch_dim<2> get_batch_size(const batch_dim<2>& size) { return size; }


template <typename T>
inline size_type get_num_batch_items(const T& obj)
{
    return obj.get_num_batch_items();
}


}  // namespace detail


/**
 *Asserts that _val1 and _val2 are equal.
 *
 *@throw ValueMisatch if _val1 is different from _val2.
 */
#define GKO_ASSERT_EQ(_val1, _val2)                                            \
    if (_val1 != _val2) {                                                      \
        throw ::gko::ValueMismatch(__FILE__, __LINE__, __func__, _val1, _val2, \
                                   "expected equal values");                   \
    }


/**
 *Asserts that _op1 is a square matrix.
 *
 *@throw DimensionMismatch  if the number of rows of _op1 is different from the
 *                          number of columns of _op1.
 */
#define GKO_ASSERT_IS_SQUARE_MATRIX(_op1)                                \
    if (::gko::detail::get_size(_op1)[0] !=                              \
        ::gko::detail::get_size(_op1)[1]) {                              \
        throw ::gko::DimensionMismatch(                                  \
            __FILE__, __LINE__, __func__, #_op1,                         \
            ::gko::detail::get_size(_op1)[0],                            \
            ::gko::detail::get_size(_op1)[1], #_op1,                     \
            ::gko::detail::get_size(_op1)[0],                            \
            ::gko::detail::get_size(_op1)[1], "expected square matrix"); \
    }


/**
 *Asserts that _op1 is a non-empty matrix.
 *
 *@throw BadDimension if any one of the dimensions of _op1 is equal to zero.
 */
#define GKO_ASSERT_IS_NON_EMPTY_MATRIX(_op1)                           \
    if (!(::gko::detail::get_size(_op1))) {                            \
        throw ::gko::BadDimension(__FILE__, __LINE__, __func__, #_op1, \
                                  ::gko::detail::get_size(_op1)[0],    \
                                  ::gko::detail::get_size(_op1)[1],    \
                                  "expected non-empty matrix");        \
    }


/**
 *Asserts that _val is a power of two.
 *
 *@throw BadDimension  if _val is not a power of two.
 */
#define GKO_ASSERT_IS_POWER_OF_TWO(_val)                                   \
    do {                                                                   \
        if (_val == 0 || (_val & (_val - 1)) != 0) {                       \
            throw ::gko::BadDimension(__FILE__, __LINE__, __func__, #_val, \
                                      _val, _val,                          \
                                      "expected power-of-two dimension");  \
        }                                                                  \
    } while (false)


/**
 * Asserts that _op1 can be applied to _op2.
 *
 * @throw DimensionMismatch  if _op1 cannot be applied to _op2.
 */
#define GKO_ASSERT_CONFORMANT(_op1, _op2)                                     \
    if (::gko::detail::get_size(_op1)[1] !=                                   \
        ::gko::detail::get_size(_op2)[0]) {                                   \
        throw ::gko::DimensionMismatch(__FILE__, __LINE__, __func__, #_op1,   \
                                       ::gko::detail::get_size(_op1)[0],      \
                                       ::gko::detail::get_size(_op1)[1],      \
                                       #_op2,                                 \
                                       ::gko::detail::get_size(_op2)[0],      \
                                       ::gko::detail::get_size(_op2)[1],      \
                                       "expected matching inner dimensions"); \
    }


/**
 * Asserts that _op1 can be applied to _op2 from the right.
 *
 * @throw DimensionMismatch  if _op1 cannot be applied to _op2 from the right.
 */
#define GKO_ASSERT_REVERSE_CONFORMANT(_op1, _op2)                             \
    if (::gko::detail::get_size(_op1)[0] !=                                   \
        ::gko::detail::get_size(_op2)[1]) {                                   \
        throw ::gko::DimensionMismatch(__FILE__, __LINE__, __func__, #_op1,   \
                                       ::gko::detail::get_size(_op1)[0],      \
                                       ::gko::detail::get_size(_op1)[1],      \
                                       #_op2,                                 \
                                       ::gko::detail::get_size(_op2)[0],      \
                                       ::gko::detail::get_size(_op2)[1],      \
                                       "expected matching inner dimensions"); \
    }


/**
 * Asserts that `_op1` and `_op2` have the same number of rows.
 *
 * @throw DimensionMismatch  if `_op1` and `_op2` differ in the number of rows
 */
#define GKO_ASSERT_EQUAL_ROWS(_op1, _op2)                                      \
    if (::gko::detail::get_size(_op1)[0] !=                                    \
        ::gko::detail::get_size(_op2)[0]) {                                    \
        throw ::gko::DimensionMismatch(                                        \
            __FILE__, __LINE__, __func__, #_op1,                               \
            ::gko::detail::get_size(_op1)[0],                                  \
            ::gko::detail::get_size(_op1)[1], #_op2,                           \
            ::gko::detail::get_size(_op2)[0],                                  \
            ::gko::detail::get_size(_op2)[1], "expected matching row length"); \
    }


/**
 * Asserts that `_op1` and `_op2` have the same number of columns.
 *
 * @throw DimensionMismatch  if `_op1` and `_op2` differ in the number of
 *                           columns
 */
#define GKO_ASSERT_EQUAL_COLS(_op1, _op2)                                   \
    if (::gko::detail::get_size(_op1)[1] !=                                 \
        ::gko::detail::get_size(_op2)[1]) {                                 \
        throw ::gko::DimensionMismatch(__FILE__, __LINE__, __func__, #_op1, \
                                       ::gko::detail::get_size(_op1)[0],    \
                                       ::gko::detail::get_size(_op1)[1],    \
                                       #_op2,                               \
                                       ::gko::detail::get_size(_op2)[0],    \
                                       ::gko::detail::get_size(_op2)[1],    \
                                       "expected matching column length");  \
    }


/**
 * Asserts that `_op1` and `_op2` have the same number of rows and columns.
 *
 * @throw DimensionMismatch  if `_op1` and `_op2` differ in the number of
 *                           rows or columns
 */
#define GKO_ASSERT_EQUAL_DIMENSIONS(_op1, _op2)                             \
    if (::gko::detail::get_size(_op1) != ::gko::detail::get_size(_op2)) {   \
        throw ::gko::DimensionMismatch(                                     \
            __FILE__, __LINE__, __func__, #_op1,                            \
            ::gko::detail::get_size(_op1)[0],                               \
            ::gko::detail::get_size(_op1)[1], #_op2,                        \
            ::gko::detail::get_size(_op2)[0],                               \
            ::gko::detail::get_size(_op2)[1], "expected equal dimensions"); \
    }


/**
 * Asserts that _op1 and _op2 have equal number of items in the batch
 *
 * @throw ValueMismatch  if _op1 and _op2 do not have equal number of items
 */
#define GKO_ASSERT_BATCH_EQUAL_NUM_ITEMS(_op1, _op2)                       \
    {                                                                      \
        auto equal_num_items =                                             \
            ::gko::detail::get_batch_size(_op1).get_num_batch_items() ==   \
            ::gko::detail::get_batch_size(_op2).get_num_batch_items();     \
        if (!equal_num_items) {                                            \
            throw ::gko::ValueMismatch(                                    \
                __FILE__, __LINE__, __func__,                              \
                ::gko::detail::get_batch_size(_op1).get_num_batch_items(), \
                ::gko::detail::get_batch_size(_op2).get_num_batch_items(), \
                "expected equal number of batch items");                   \
        }                                                                  \
    }


/**
 * Asserts that _op1 can be applied to _op2.
 *
 * @throw DimensionMismatch  if _op1 cannot be applied to _op2.
 */
#define GKO_ASSERT_BATCH_CONFORMANT(_op1, _op2)                              \
    {                                                                        \
        GKO_ASSERT_BATCH_EQUAL_NUM_ITEMS(_op1, _op2);                        \
        auto equal_inner_size =                                              \
            ::gko::detail::get_batch_size(_op1).get_common_size()[1] ==      \
            ::gko::detail::get_batch_size(_op2).get_common_size()[0];        \
        if (!equal_inner_size) {                                             \
            throw ::gko::DimensionMismatch(                                  \
                __FILE__, __LINE__, __func__, #_op1,                         \
                ::gko::detail::get_batch_size(_op1).get_common_size()[0],    \
                ::gko::detail::get_batch_size(_op1).get_common_size()[1],    \
                #_op2,                                                       \
                ::gko::detail::get_batch_size(_op2).get_common_size()[0],    \
                ::gko::detail::get_batch_size(_op2).get_common_size()[1],    \
                "expected matching inner dimensions among all batch items"); \
        }                                                                    \
    }


/**
 * Asserts that _op1 can be applied to _op2 from the right.
 *
 * @throw DimensionMismatch  if _op1 cannot be applied to _op2 from the right.
 */
#define GKO_ASSERT_BATCH_REVERSE_CONFORMANT(_op1, _op2)                      \
    {                                                                        \
        GKO_ASSERT_BATCH_EQUAL_NUM_ITEMS(_op1, _op2);                        \
        auto equal_outer_size =                                              \
            ::gko::detail::get_batch_size(_op1).get_common_size()[0] ==      \
            ::gko::detail::get_batch_size(_op2).get_common_size()[1];        \
        if (!equal_outer_size) {                                             \
            throw ::gko::DimensionMismatch(                                  \
                __FILE__, __LINE__, __func__, #_op1,                         \
                ::gko::detail::get_batch_size(_op1).get_common_size()[0],    \
                ::gko::detail::get_batch_size(_op1).get_common_size()[1],    \
                #_op2,                                                       \
                ::gko::detail::get_batch_size(_op2).get_common_size()[0],    \
                ::gko::detail::get_batch_size(_op2).get_common_size()[1],    \
                "expected matching outer dimensions among all batch items"); \
        }                                                                    \
    }


/**
 * Asserts that `_op1` and `_op2` have the same number of rows.
 *
 * @throw DimensionMismatch  if `_op1` and `_op2` differ in the number of rows
 */
#define GKO_ASSERT_BATCH_EQUAL_ROWS(_op1, _op2)                            \
    {                                                                      \
        GKO_ASSERT_BATCH_EQUAL_NUM_ITEMS(_op1, _op2);                      \
        auto equal_rows =                                                  \
            ::gko::detail::get_batch_size(_op1).get_common_size()[0] ==    \
            ::gko::detail::get_batch_size(_op2).get_common_size()[0];      \
        if (!equal_rows) {                                                 \
            throw ::gko::DimensionMismatch(                                \
                __FILE__, __LINE__, __func__, #_op1,                       \
                ::gko::detail::get_batch_size(_op1).get_common_size()[0],  \
                ::gko::detail::get_batch_size(_op1).get_common_size()[1],  \
                #_op2,                                                     \
                ::gko::detail::get_batch_size(_op2).get_common_size()[0],  \
                ::gko::detail::get_batch_size(_op2).get_common_size()[1],  \
                "expected matching number of rows among all batch items"); \
        }                                                                  \
    }


/**
 * Asserts that `_op1` and `_op2` have the same number of columns.
 *
 * @throw DimensionMismatch  if `_op1` and `_op2` differ in the number of
 *                           columns
 */
#define GKO_ASSERT_BATCH_EQUAL_COLS(_op1, _op2)                            \
    {                                                                      \
        GKO_ASSERT_BATCH_EQUAL_NUM_ITEMS(_op1, _op2);                      \
        auto equal_cols =                                                  \
            ::gko::detail::get_batch_size(_op1).get_common_size()[1] ==    \
            ::gko::detail::get_batch_size(_op2).get_common_size()[1];      \
        if (!equal_cols) {                                                 \
            throw ::gko::DimensionMismatch(                                \
                __FILE__, __LINE__, __func__, #_op1,                       \
                ::gko::detail::get_batch_size(_op1).get_common_size()[0],  \
                ::gko::detail::get_batch_size(_op1).get_common_size()[1],  \
                #_op2,                                                     \
                ::gko::detail::get_batch_size(_op2).get_common_size()[0],  \
                ::gko::detail::get_batch_size(_op2).get_common_size()[1],  \
                "expected matching number of cols among all batch items"); \
        }                                                                  \
    }


/**
 * Asserts that `_op1` and `_op2` have the same number of rows and columns.
 *
 * @throw DimensionMismatch  if `_op1` and `_op2` differ in the number of
 *                           rows or columns
 */
#define GKO_ASSERT_BATCH_EQUAL_DIMENSIONS(_op1, _op2)                     \
    {                                                                     \
        GKO_ASSERT_BATCH_EQUAL_NUM_ITEMS(_op1, _op2);                     \
        auto equal_size =                                                 \
            ::gko::detail::get_batch_size(_op1).get_common_size() ==      \
            ::gko::detail::get_batch_size(_op2).get_common_size();        \
        if (!equal_size) {                                                \
            throw ::gko::DimensionMismatch(                               \
                __FILE__, __LINE__, __func__, #_op1,                      \
                ::gko::detail::get_batch_size(_op1).get_common_size()[0], \
                ::gko::detail::get_batch_size(_op1).get_common_size()[1], \
                #_op2,                                                    \
                ::gko::detail::get_batch_size(_op2).get_common_size()[0], \
                ::gko::detail::get_batch_size(_op2).get_common_size()[1], \
                "expected matching size among all batch items");          \
        }                                                                 \
    }


/**
 * Asserts that `_op1` and `_op2` have the same number of rows and columns.
 *
 * @throw DimensionMismatch  if `_op1` and `_op2` differ in the number of
 *                           rows or columns
 */
#define GKO_ASSERT_BATCH_HAS_SQUARE_DIMENSIONS(_op1)                      \
    {                                                                     \
        auto is_square =                                                  \
            ::gko::detail::get_batch_size(_op1).get_common_size()[0] ==   \
            ::gko::detail::get_batch_size(_op1).get_common_size()[1];     \
        if (!is_square) {                                                 \
            throw ::gko::BadDimension(                                    \
                __FILE__, __LINE__, __func__, #_op1,                      \
                ::gko::detail::get_batch_size(_op1).get_common_size()[0], \
                ::gko::detail::get_batch_size(_op1).get_common_size()[1], \
                "expected common size of matrices to be square");         \
        }                                                                 \
    }


/**
 * Instantiates a MpiError.
 *
 * @param errcode  The error code returned from the MPI routine.
 */
#define GKO_MPI_ERROR(_errcode) \
    ::gko::MpiError(__FILE__, __LINE__, __func__, _errcode)


/**
 * Instantiates a CudaError.
 *
 * @param errcode  The error code returned from a CUDA runtime API routine.
 */
#define GKO_CUDA_ERROR(_errcode) \
    ::gko::CudaError(__FILE__, __LINE__, __func__, _errcode)


/**
 * Instantiates a CublasError.
 *
 * @param errcode  The error code returned from the cuBLAS routine.
 */
#define GKO_CUBLAS_ERROR(_errcode) \
    ::gko::CublasError(__FILE__, __LINE__, __func__, _errcode)


/**
 * Instantiates a CurandError.
 *
 * @param errcode  The error code returned from the cuRAND routine.
 */
#define GKO_CURAND_ERROR(_errcode) \
    ::gko::CurandError(__FILE__, __LINE__, __func__, _errcode)


/**
 * Instantiates a CusparseError.
 *
 * @param errcode  The error code returned from the cuSPARSE routine.
 */
#define GKO_CUSPARSE_ERROR(_errcode) \
    ::gko::CusparseError(__FILE__, __LINE__, __func__, _errcode)


/**
 * Instantiates a CufftError.
 *
 * @param errcode  The error code returned from the cuFFT routine.
 */
#define GKO_CUFFT_ERROR(_errcode) \
    ::gko::CufftError(__FILE__, __LINE__, __func__, _errcode)


/**
 * Asserts that a CUDA library call completed without errors.
 *
 * @param _cuda_call  a library call expression
 */
#define GKO_ASSERT_NO_CUDA_ERRORS(_cuda_call) \
    do {                                      \
        auto _errcode = _cuda_call;           \
        if (_errcode != cudaSuccess) {        \
            throw GKO_CUDA_ERROR(_errcode);   \
        }                                     \
    } while (false)


/**
 * Asserts that a cuBLAS library call completed without errors.
 *
 * @param _cublas_call  a library call expression
 */
#define GKO_ASSERT_NO_CUBLAS_ERRORS(_cublas_call) \
    do {                                          \
        auto _errcode = _cublas_call;             \
        if (_errcode != CUBLAS_STATUS_SUCCESS) {  \
            throw GKO_CUBLAS_ERROR(_errcode);     \
        }                                         \
    } while (false)


/**
 * Asserts that a cuRAND library call completed without errors.
 *
 * @param _curand_call  a library call expression
 */
#define GKO_ASSERT_NO_CURAND_ERRORS(_curand_call) \
    do {                                          \
        auto _errcode = _curand_call;             \
        if (_errcode != CURAND_STATUS_SUCCESS) {  \
            throw GKO_CURAND_ERROR(_errcode);     \
        }                                         \
    } while (false)


/**
 * Asserts that a cuSPARSE library call completed without errors.
 *
 * @param _cusparse_call  a library call expression
 */
#define GKO_ASSERT_NO_CUSPARSE_ERRORS(_cusparse_call) \
    do {                                              \
        auto _errcode = _cusparse_call;               \
        if (_errcode != CUSPARSE_STATUS_SUCCESS) {    \
            throw GKO_CUSPARSE_ERROR(_errcode);       \
        }                                             \
    } while (false)


/**
 * Asserts that a cuFFT library call completed without errors.
 *
 * @param _cufft_call  a library call expression
 */
#define GKO_ASSERT_NO_CUFFT_ERRORS(_cufft_call) \
    do {                                        \
        auto _errcode = _cufft_call;            \
        if (_errcode != CUFFT_SUCCESS) {        \
            throw GKO_CUFFT_ERROR(_errcode);    \
        }                                       \
    } while (false)


/**
 * Instantiates a HipError.
 *
 * @param errcode  The error code returned from a HIP runtime API routine.
 */
#define GKO_HIP_ERROR(_errcode) \
    ::gko::HipError(__FILE__, __LINE__, __func__, _errcode)


/**
 * Instantiates a HipblasError.
 *
 * @param errcode  The error code returned from the HIPBLAS routine.
 */
#define GKO_HIPBLAS_ERROR(_errcode) \
    ::gko::HipblasError(__FILE__, __LINE__, __func__, _errcode)


/**
 * Instantiates a HiprandError.
 *
 * @param errcode  The error code returned from the HIPRAND routine.
 */
#define GKO_HIPRAND_ERROR(_errcode) \
    ::gko::HiprandError(__FILE__, __LINE__, __func__, _errcode)


/**
 * Instantiates a HipsparseError.
 *
 * @param errcode  The error code returned from the HIPSPARSE routine.
 */
#define GKO_HIPSPARSE_ERROR(_errcode) \
    ::gko::HipsparseError(__FILE__, __LINE__, __func__, _errcode)


/**
 * Instantiates a HipfftError.
 *
 * @param errcode  The error code returned from the hipFFT routine.
 */
#define GKO_HIPFFT_ERROR(_errcode) \
    ::gko::HipfftError(__FILE__, __LINE__, __func__, _errcode)


/**
 * Asserts that a HIP library call completed without errors.
 *
 * @param _hip_call  a library call expression
 */
#define GKO_ASSERT_NO_HIP_ERRORS(_hip_call) \
    do {                                    \
        auto _errcode = _hip_call;          \
        if (_errcode != hipSuccess) {       \
            throw GKO_HIP_ERROR(_errcode);  \
        }                                   \
    } while (false)


/**
 * Asserts that a HIPBLAS library call completed without errors.
 *
 * @param _hipblas_call  a library call expression
 */
#define GKO_ASSERT_NO_HIPBLAS_ERRORS(_hipblas_call) \
    do {                                            \
        auto _errcode = _hipblas_call;              \
        if (_errcode != HIPBLAS_STATUS_SUCCESS) {   \
            throw GKO_HIPBLAS_ERROR(_errcode);      \
        }                                           \
    } while (false)


/**
 * Asserts that a HIPRAND library call completed without errors.
 *
 * @param _hiprand_call  a library call expression
 */
#define GKO_ASSERT_NO_HIPRAND_ERRORS(_hiprand_call) \
    do {                                            \
        auto _errcode = _hiprand_call;              \
        if (_errcode != HIPRAND_STATUS_SUCCESS) {   \
            throw GKO_HIPRAND_ERROR(_errcode);      \
        }                                           \
    } while (false)


/**
 * Asserts that a HIPSPARSE library call completed without errors.
 *
 * @param _hipsparse_call  a library call expression
 */
#define GKO_ASSERT_NO_HIPSPARSE_ERRORS(_hipsparse_call) \
    do {                                                \
        auto _errcode = _hipsparse_call;                \
        if (_errcode != HIPSPARSE_STATUS_SUCCESS) {     \
            throw GKO_HIPSPARSE_ERROR(_errcode);        \
        }                                               \
    } while (false)


/**
 * Asserts that a hipFFT library call completed without errors.
 *
 * @param _hipfft_call  a library call expression
 */
#define GKO_ASSERT_NO_HIPFFT_ERRORS(_hipfft_call) \
    do {                                          \
        auto _errcode = _hipfft_call;             \
        if (_errcode != HIPFFT_SUCCESS) {         \
            throw GKO_HIPFFT_ERROR(_errcode);     \
        }                                         \
    } while (false)


/**
 * Asserts that a MPI library call completed without errors.
 *
 * @param _mpi_call  a library call expression
 */
#define GKO_ASSERT_NO_MPI_ERRORS(_mpi_call) \
    do {                                    \
        auto _errcode = _mpi_call;          \
        if (_errcode != MPI_SUCCESS) {      \
            throw GKO_MPI_ERROR(_errcode);  \
        }                                   \
    } while (false)


namespace detail {


template <typename T>
inline T ensure_allocated_impl(T ptr, const std::string& file, int line,
                               const std::string& dev, size_type size)
{
    if (ptr == nullptr) {
        throw AllocationError(file, line, dev, size);
    }
    return ptr;
}


}  // namespace detail


/**
 * Ensures the memory referenced by _ptr is allocated.
 *
 * @param _ptr  the result of the allocation, if it is a nullptr, an exception
 *              is thrown
 * @param _dev  the device where the data was allocated, used to provide
 *              additional information in the error message
 * @param _size  the size in bytes of the allocated memory, used to provide
 *               additional information in the error message
 *
 * @throw AllocationError  if the pointer equals nullptr
 *
 * @return _ptr
 */
#define GKO_ENSURE_ALLOCATED(_ptr, _dev, _size) \
    ::gko::detail::ensure_allocated_impl(_ptr, __FILE__, __LINE__, _dev, _size)


/**
 * Ensures that a memory access is in the bounds.
 *
 * @param _index  the index which is being accessed
 * @param _bound  the bound of the array being accessed
 *
 * @throw OutOfBoundsError  if `_index >= _bound`
 */
#define GKO_ENSURE_IN_BOUNDS(_index, _bound)                                 \
    if (_index >= _bound) {                                                  \
        throw ::gko::OutOfBoundsError(__FILE__, __LINE__, _index, _bound);   \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


/**
 * Ensures that two dimensions have compatible bounds, in particular before a
 * copy operation. This means the target should have at least as much elements
 * as the source.
 *
 * @param _source  the source of the expected copy operation
 * @param _target  the destination of the expected copy operation
 *
 * @throw OutOfBoundsError  if `_source > _target`
 */
#define GKO_ENSURE_COMPATIBLE_BOUNDS(_source, _target)                       \
    if (_source > _target) {                                                 \
        throw ::gko::OutOfBoundsError(__FILE__, __LINE__, _source, _target); \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


/**
 * Ensures that an access is within the specified 2D dimensions.
 *
 * @param _row  the row access
 * @param _col  the column access
 * @param _bound  the dimension bound
 *
 * @throw OutOfBoundsError  if `_row >= _bound[0] || _col >= _bound[1]`
 */
#define GKO_ENSURE_IN_DIMENSION_BOUNDS(_row, _col, _bound)          \
    GKO_ENSURE_IN_BOUNDS(_row, ::gko::detail::get_size(_bound)[0]); \
    GKO_ENSURE_IN_BOUNDS(_col, ::gko::detail::get_size(_bound)[1])


/**
 * Creates a StreamError exception.
 * This macro sets the correct information about the location of the error
 * and fills the exception with data about the stream, and the reason for the
 * error.
 *
 * @param _message  the error message describing the details of the error
 *
 * @return FileError
 */
#define GKO_STREAM_ERROR(_message) \
    ::gko::StreamError(__FILE__, __LINE__, __func__, _message)


/**
 * Marks a kernel as not eligible for any predicate.
 *
 * Attempts to call this kernel will result in a runtime error of type
 * KernelNotFound.
 */
#define GKO_KERNEL_NOT_FOUND                                                 \
    {                                                                        \
        throw ::gko::KernelNotFound(__FILE__, __LINE__, __func__);           \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


/**
 * Throws an @ref UnsupportedMatrixProperty exception in case of an unmet
 * matrix property requirement.
 *
 * @param _message  A message describing the matrix property required.
 */
#define GKO_UNSUPPORTED_MATRIX_PROPERTY(_message)                             \
    {                                                                         \
        throw ::gko::UnsupportedMatrixProperty(__FILE__, __LINE__, _message); \
    }                                                                         \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")


/**
 * Ensures that a given size, typically of a linear algebraic object,
 * is divisible by a given block size.
 *
 * @param _size  A size of a vector or matrix
 * @param _block_size  Size of small dense blocks that make up
 *                     the vector or matrix
 *
 * @throw BlockSizeError  if _block_size does not divide _size
 */
#define GKO_ASSERT_BLOCK_SIZE_CONFORMANT(_size, _block_size)                   \
    if (_size % _block_size != 0) {                                            \
        throw BlockSizeError<decltype(_size)>(__FILE__, __LINE__, _block_size, \
                                              _size);                          \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")


/**
 * Checks that the operator is a scalar, ie., has size 1x1.
 *
 * @param _op  Operator to be checked.
 *
 * @throw  BadDimension  if _op does not have size 1x1.
 */
#define GKO_ASSERT_IS_SCALAR(_op)                                            \
    {                                                                        \
        auto sz = gko::detail::get_size(_op);                                \
        if (sz[0] != 1 || sz[1] != 1) {                                      \
            throw ::gko::BadDimension(__FILE__, __LINE__, __func__, #_op,    \
                                      sz[0], sz[1], "expected scalar");      \
        }                                                                    \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


/**
 * Throws an InvalidStateError with a user-specified message
 *
 * @param _message  message to be displayed.
 *
 * @throw  InvalidStateError.
 */
#define GKO_INVALID_STATE(_message)                                          \
    {                                                                        \
        throw ::gko::InvalidStateError(__FILE__, __LINE__, __func__,         \
                                       _message);                            \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


/**
 * Throws an InvalidStateError if condition is not satisfied
 *
 * @param _condition  the condition to check.
 * @param _message  message to be displayed.
 *
 * @throw  InvalidStateError.
 */
#define GKO_THROW_IF_INVALID(_condition, _message)                           \
    {                                                                        \
        if (!(_condition)) {                                                 \
            throw ::gko::InvalidStateError(__FILE__, __LINE__, __func__,     \
                                           _message);                        \
        }                                                                    \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_EXCEPTION_HELPERS_HPP_
