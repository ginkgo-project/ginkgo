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

#include <ginkgo/core/base/exception.hpp>


namespace gko {


Error::Error(const std::string& file, int line, const std::string& what)
    : what_(file + ":" + std::to_string(line) + ": " + what)
{}


const char* Error::what() const noexcept { return what_.c_str(); }


NotImplemented::NotImplemented(const std::string& file, int line,
                               const std::string& func)
    : Error(file, line, func + " is not implemented")
{}


NotCompiled::NotCompiled(const std::string& file, int line,
                         const std::string& func, const std::string& module)
    : Error(file, line,
            "feature " + func + " is part of the " + module +
                " module, which is not compiled on this system")
{}


NotSupported::NotSupported(const std::string& file, int line,
                           const std::string& func, const std::string& obj_type)
    : Error(file, line,
            "Operation " + func + " does not support parameters of type " +
                obj_type)
{}


MpiError::MpiError(const std::string& file, int line, const std::string& func,
                   int64 error_code)
    : Error(file, line, func + ": " + get_error(error_code))
{}


CudaError::CudaError(const std::string& file, int line, const std::string& func,
                     int64 error_code)
    : Error(file, line, func + ": " + get_error(error_code))
{}


CublasError::CublasError(const std::string& file, int line,
                         const std::string& func, int64 error_code)
    : Error(file, line, func + ": " + get_error(error_code))
{}


CurandError::CurandError(const std::string& file, int line,
                         const std::string& func, int64 error_code)
    : Error(file, line, func + ": " + get_error(error_code))
{}


CusparseError::CusparseError(const std::string& file, int line,
                             const std::string& func, int64 error_code)
    : Error(file, line, func + ": " + get_error(error_code))
{}


CufftError::CufftError(const std::string& file, int line,
                       const std::string& func, int64 error_code)
    : Error(file, line, func + ": " + get_error(error_code))
{}

HipError::HipError(const std::string& file, int line, const std::string& func,
                   int64 error_code)
    : Error(file, line, func + ": " + get_error(error_code))
{}


HipblasError::HipblasError(const std::string& file, int line,
                           const std::string& func, int64 error_code)
    : Error(file, line, func + ": " + get_error(error_code))
{}


HiprandError::HiprandError(const std::string& file, int line,
                           const std::string& func, int64 error_code)
    : Error(file, line, func + ": " + get_error(error_code))
{}


HipsparseError::HipsparseError(const std::string& file, int line,
                               const std::string& func, int64 error_code)
    : Error(file, line, func + ": " + get_error(error_code))
{}


HipfftError::HipfftError(const std::string& file, int line,
                         const std::string& func, int64 error_code)
    : Error(file, line, func + ": " + get_error(error_code))
{}


DimensionMismatch::DimensionMismatch(
    const std::string& file, int line, const std::string& func,
    const std::string& first_name, size_type first_rows, size_type first_cols,
    const std::string& second_name, size_type second_rows,
    size_type second_cols, const std::string& clarification)
    : Error(file, line,
            func + ": attempting to combine operators " + first_name + " [" +
                std::to_string(first_rows) + " x " +
                std::to_string(first_cols) + "] and " + second_name + " [" +
                std::to_string(second_rows) + " x " +
                std::to_string(second_cols) + "]: " + clarification)
{}


BadDimension::BadDimension(const std::string& file, int line,
                           const std::string& func, const std::string& op_name,
                           size_type op_num_rows, size_type op_num_cols,
                           const std::string& clarification)
    : Error(file, line,
            func + ": Object " + op_name + " has dimensions [" +
                std::to_string(op_num_rows) + " x " +
                std::to_string(op_num_cols) + "]: " + clarification)
{}


ValueMismatch::ValueMismatch(const std::string& file, int line,
                             const std::string& func, size_type val1,
                             size_type val2, const std::string& clarification)
    : Error(file, line,
            func + ": Value mismatch : " + std::to_string(val1) + " and " +
                std::to_string(val2) + " : " + clarification)
{}


AllocationError::AllocationError(const std::string& file, int line,
                                 const std::string& device, size_type bytes)
    : Error(file, line,
            device + ": failed to allocate memory block of " +
                std::to_string(bytes) + "B")
{}


OutOfBoundsError::OutOfBoundsError(const std::string& file, int line,
                                   size_type index, size_type bound)
    : Error(file, line,
            "trying to access index " + std::to_string(index) +
                " in a memory block of " + std::to_string(bound) + " elements")
{}


StreamError::StreamError(const std::string& file, int line,
                         const std::string& func, const std::string& message)
    : Error(file, line, func + ": " + message)
{}


KernelNotFound::KernelNotFound(const std::string& file, int line,
                               const std::string& func)
    : Error(file, line, func + ": unable to find an eligible kernel")
{}


UnsupportedMatrixProperty::UnsupportedMatrixProperty(const std::string& file,
                                                     const int line,
                                                     const std::string& msg)
    : Error(file, line, msg)
{}


}  // namespace gko
