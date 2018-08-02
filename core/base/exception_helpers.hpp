/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_CORE_EXCEPTION_HELPERS_HPP_
#define GKO_CORE_EXCEPTION_HELPERS_HPP_


#include "core/base/dim.hpp"
#include "core/base/exception.hpp"


#include <typeinfo>


namespace gko {


/**
 * Marks a function as not yet implemented.
 *
 * Attempts to call this function will result in a runtime error of type
 * NotImplemented.
 */
#define NOT_IMPLEMENTED                                            \
    {                                                              \
        throw ::gko::NotImplemented(__FILE__, __LINE__, __func__); \
    }


/**
 * Marks a function as not compiled.
 *
 * Attempts to call this function will result in a runtime error of type
 * NotCompiled
 *
 * @param _module  the module which should be compiled to enable the function
 */
#define NOT_COMPILED(_module)                                             \
    {                                                                     \
        throw ::gko::NotCompiled(__FILE__, __LINE__, __func__, #_module); \
    }


/**
 * Creates a NotSupported exception.
 * This macro sets the correct information about the location of the error
 * and fills the exception with data about _obj.
 *
 * @param _obj  the object referenced by NotSupported exception
 *
 * @return NotSupported
 */
#define NOT_SUPPORTED(_obj) \
    ::gko::NotSupported(__FILE__, __LINE__, __func__, typeid(_obj).name())


namespace detail {


template <typename T>
inline dim<2> get_size(const T &op)
{
    return op->get_size();
}

inline dim<2> get_size(const dim<2> &size) { return size; }


}  // namespace detail


/**
 *Asserts that _op1 is a square matrix.
 *
 *@throw DimensionMismatch  if the number of rows of _op1 is different from the
 *                          number of columns of _op1.
 */
#define ASSERT_IS_SQUARE_MATRIX(_op1)                                          \
    if (::gko::detail::get_size(_op1).num_rows !=                              \
        ::gko::detail::get_size(_op1).num_cols) {                              \
        throw ::gko::DimensionMismatch(                                        \
            __FILE__, __LINE__, __func__, #_op1,                               \
            ::gko::detail::get_size(_op1).num_rows,                            \
            ::gko::detail::get_size(_op1).num_cols, #_op1,                     \
            ::gko::detail::get_size(_op1).num_rows,                            \
            ::gko::detail::get_size(_op1).num_cols, "expected square matrix"); \
    }


/**
 * Asserts that _op1 can be applied to _op2.
 *
 * @throw DimensionMismatch  if _op1 cannot be applied to _op2.
 */
#define ASSERT_CONFORMANT(_op1, _op2)                                         \
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
 * Asserts that `_op1` and `_op2` have the same number of rows.
 *
 * @throw DimensionMismatch  if `_op1` and `_op2` differ in the number of rows
 */
#define ASSERT_EQUAL_ROWS(_op1, _op2)                                          \
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
#define ASSERT_EQUAL_COLS(_op1, _op2)                                       \
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
#define ASSERT_EQUAL_DIMENSIONS(_op1, _op2)                                 \
    if (::gko::detail::get_size(_op1) != ::gko::detail::get_size(_op2)) {   \
        throw ::gko::DimensionMismatch(                                     \
            __FILE__, __LINE__, __func__, #_op1,                            \
            ::gko::detail::get_size(_op1)[0],                               \
            ::gko::detail::get_size(_op1)[1], #_op2,                        \
            ::gko::detail::get_size(_op2)[0],                               \
            ::gko::detail::get_size(_op2)[1], "expected equal dimensions"); \
    }


/**
 * Instantiates a CudaError.
 *
 * @param errcode  The error code returned from a CUDA runtime API routine.
 */
#define CUDA_ERROR(_errcode) \
    ::gko::CudaError(__FILE__, __LINE__, __func__, _errcode)

/**
 * Instantiates a CublasError.
 *
 * @param errcode  The error code returned from the cuBLAS routine.
 */
#define CUBLAS_ERROR(_errcode) \
    ::gko::CublasError(__FILE__, __LINE__, __func__, _errcode)

/**
 * Instantiates a CusparseError.
 *
 * @param errcode  The error code returned from the cuSPARSE routine.
 */
#define CUSPARSE_ERROR(_errcode) \
    ::gko::CusparseError(__FILE__, __LINE__, __func__, _errcode)


/**
 * Asserts that a CUDA library call completed without errors.
 *
 * @param _cuda_call  a library call expression
 */
#define ASSERT_NO_CUDA_ERRORS(_cuda_call) \
    do {                                  \
        auto _errcode = _cuda_call;       \
        if (_errcode != cudaSuccess) {    \
            throw CUDA_ERROR(_errcode);   \
        }                                 \
    } while (false)


/**
 * Asserts that a cuBLAS library call completed without errors.
 *
 * @param _cuda_call  a library call expression
 */
#define ASSERT_NO_CUBLAS_ERRORS(_cublas_call)    \
    do {                                         \
        auto _errcode = _cublas_call;            \
        if (_errcode != CUBLAS_STATUS_SUCCESS) { \
            throw CUBLAS_ERROR(_errcode);        \
        }                                        \
    } while (false)


/**
 * Asserts that a cuSPARSE library call completed without errors.
 *
 * @param _cuda_call  a library call expression
 */
#define ASSERT_NO_CUSPARSE_ERRORS(_cusparse_call)  \
    do {                                           \
        auto _errcode = _cusparse_call;            \
        if (_errcode != CUSPARSE_STATUS_SUCCESS) { \
            throw CUSPARSE_ERROR(_errcode);        \
        }                                          \
    } while (false)


namespace detail {


template <typename T>
inline T ensure_allocated_impl(T ptr, const std::string &file, int line,
                               const std::string &dev, size_type size)
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
#define ENSURE_ALLOCATED(_ptr, _dev, _size) \
    ::gko::detail::ensure_allocated_impl(_ptr, __FILE__, __LINE__, _dev, _size)

/**
 * Ensures that a memory access is in the bounds.
 *
 * @param _index  the index which is being accessed
 * @param _bound  the bound of the array being accessed
 *
 * @throw OutOfBoundsError  if `_index >= _bound`
 */
#define ENSURE_IN_BOUNDS(_index, _bound)                                   \
    if (_index >= _bound) {                                                \
        throw ::gko::OutOfBoundsError(__FILE__, __LINE__, _index, _bound); \
    }


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
#define STREAM_ERROR(_message) \
    ::gko::StreamError(__FILE__, __LINE__, __func__, _message)


/**
 * Marks a kernel as not eligible for any predicate.
 *
 * Attempts to call this kernel will result in a runtime error of type
 * KernelNotFound.
 */
#define KERNEL_NOT_FOUND                                           \
    {                                                              \
        throw ::gko::KernelNotFound(__FILE__, __LINE__, __func__); \
    }


}  // namespace gko


#endif  // GKO_CORE_EXCEPTION_HELPERS_HPP_
