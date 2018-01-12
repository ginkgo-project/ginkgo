/*
 * Copyright 2017-2018
 *
 * Karlsruhe Institute of Technology
 *
 * Universitat Jaume I
 *
 * University of Tennessee
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef GKO_CORE_EXCEPTION_HELPERS_HPP_
#define GKO_CORE_EXCEPTION_HELPERS_HPP_


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


/**
 * Asserts that _operator can be applied to _vectors.
 *
 * @throw DimensionMismatch  if _operator cannot be applied to _vectors.
 */
#define ASSERT_CONFORMANT(_operator, _vectors)                         \
    if ((_operator)->get_num_cols() != (_vectors)->get_num_rows()) {   \
        throw ::gko::DimensionMismatch(                                \
            __FILE__, __LINE__, __func__, (_operator)->get_num_rows(), \
            (_operator)->get_num_cols(), (_vectors)->get_num_rows(),   \
            (_vectors)->get_num_cols());                               \
    }


/**
 * Instantiates a CudaError.
 *
 * @param errcode The error code returned from a CUDA runtime API routine.
 */
#define CUDA_ERROR(_errcode) \
    ::gko::CudaError(__FILE__, __LINE__, __func__, _errcode)


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


}  // namespace gko


#endif  // GKO_CORE_EXCEPTION_HELPERS_HPP_
