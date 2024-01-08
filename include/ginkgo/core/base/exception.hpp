// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_EXCEPTION_HPP_
#define GKO_PUBLIC_CORE_BASE_EXCEPTION_HPP_


#include <exception>
#include <string>


#include <ginkgo/core/base/types.hpp>


namespace gko {


/**
 * The Error class is used to report exceptional behaviour in library
 * functions. Ginkgo uses C++ exception mechanism to this end, and the
 * Error class represents a base class for all types of errors. The exact list
 * of errors which could occur during the execution of a certain library
 * routine is provided in the documentation of that routine, along with a short
 * description of the situation when that error can occur.
 * During runtime, these errors can be detected by using standard C++ try-catch
 * blocks, and a human-readable error description can be obtained by calling
 * the Error::what() method.
 *
 * As an example, trying to compute a matrix-vector product with arguments of
 * incompatible size will result in a DimensionMismatch error, which is
 * demonstrated in the following program.
 *
 * ```cpp
 * #include <ginkgo.h>
 * #include <iostream>
 *
 * using namespace gko;
 *
 * int main()
 * {
 *     auto omp = create<OmpExecutor>();
 *     auto A = randn_fill<matrix::Csr<float>>(5, 5, 0f, 1f, omp);
 *     auto x = fill<matrix::Dense<float>>(6, 1, 1f, omp);
 *     try {
 *         auto y = apply(A, x);
 *     } catch(Error e) {
 *         // an error occurred, write the message to screen and exit
 *         std::cout << e.what() << std::endl;
 *         return -1;
 *     }
 *     return 0;
 * }
 * ```
 *
 * @ingroup error
 */
class Error : public std::exception {
public:
    /**
     * Initializes an error.
     *
     * @param file  The name of the offending source file
     * @param line  The source code line number where the error occurred
     * @param what  The error message
     */
    Error(const std::string& file, int line, const std::string& what)
        : what_(file + ":" + std::to_string(line) + ": " + what)
    {}

    /**
     * Returns a human-readable string with a more detailed description of the
     * error.
     */
    virtual const char* what() const noexcept override { return what_.c_str(); }

private:
    const std::string what_;
};


/**
 * NotImplemented is thrown in case an operation has not yet
 * been implemented (but will be implemented in the future).
 */
class NotImplemented : public Error {
public:
    /**
     * Initializes a NotImplemented error.
     *
     * @param file  The name of the offending source file
     * @param line  The source code line number where the error occurred
     * @param func  The name of the not-yet implemented function
     */
    NotImplemented(const std::string& file, int line, const std::string& func)
        : Error(file, line, func + " is not implemented")
    {}
};


/**
 * NotCompiled is thrown when attempting to call an operation which is a part of
 * a module that was not compiled on the system.
 */
class NotCompiled : public Error {
public:
    /**
     * Initializes a NotCompiled error.
     *
     * @param file  The name of the offending source file
     * @param line  The source code line number where the error occurred
     * @param func  The name of the function that has not been compiled
     * @param module  The name of the module which contains the function
     */
    NotCompiled(const std::string& file, int line, const std::string& func,
                const std::string& module)
        : Error(file, line,
                "feature " + func + " is part of the " + module +
                    " module, which is not compiled on this system")
    {}
};


/**
 * NotSupported is thrown in case it is not possible to
 * perform the requested operation on the given object type.
 */
class NotSupported : public Error {
public:
    /**
     * Initializes a NotSupported error.
     *
     * @param file  The name of the offending source file
     * @param line  The source code line number where the error occurred
     * @param func  The name of the function where the error occurred
     * @param obj_type  The object type on which the requested operation
                       cannot be performed.
     */
    NotSupported(const std::string& file, int line, const std::string& func,
                 const std::string& obj_type)
        : Error(file, line,
                "Operation " + func + " does not support parameters of type " +
                    obj_type)
    {}
};


/**
 * MpiError is thrown when a MPI routine throws a non-zero error code.
 */
class MpiError : public Error {
public:
    /**
     * Initializes a MPI error.
     * @param file The name of the offending source file
     * @param line The source code line number where the error occurred
     * @param func The name of the MPI routine that failed
     * @param error_code The resulting MPI error code
     */
    MpiError(const std::string& file, int line, const std::string& func,
             int64 error_code)
        : Error(file, line, func + ": " + get_error(error_code))
    {}

private:
    static std::string get_error(int64 error_code);
};


/**
 * CudaError is thrown when a CUDA routine throws a non-zero error code.
 */
class CudaError : public Error {
public:
    /**
     * Initializes a CUDA error.
     *
     * @param file  The name of the offending source file
     * @param line  The source code line number where the error occurred
     * @param func  The name of the CUDA routine that failed
     * @param error_code  The resulting CUDA error code
     */
    CudaError(const std::string& file, int line, const std::string& func,
              int64 error_code)
        : Error(file, line, func + ": " + get_error(error_code))
    {}

private:
    static std::string get_error(int64 error_code);
};


/**
 * CublasError is thrown when a cuBLAS routine throws a non-zero error code.
 */
class CublasError : public Error {
public:
    /**
     * Initializes a cuBLAS error.
     *
     * @param file  The name of the offending source file
     * @param line  The source code line number where the error occurred
     * @param func  The name of the cuBLAS routine that failed
     * @param error_code  The resulting cuBLAS error code
     */
    CublasError(const std::string& file, int line, const std::string& func,
                int64 error_code)
        : Error(file, line, func + ": " + get_error(error_code))
    {}

private:
    static std::string get_error(int64 error_code);
};


/**
 * CurandError is thrown when a cuRAND routine throws a non-zero error code.
 */
class CurandError : public Error {
public:
    /**
     * Initializes a cuRAND error.
     *
     * @param file  The name of the offending source file
     * @param line  The source code line number where the error occurred
     * @param func  The name of the cuRAND routine that failed
     * @param error_code  The resulting cuRAND error code
     */
    CurandError(const std::string& file, int line, const std::string& func,
                int64 error_code)
        : Error(file, line, func + ": " + get_error(error_code))
    {}

private:
    static std::string get_error(int64 error_code);
};


/**
 * CusparseError is thrown when a cuSPARSE routine throws a non-zero error code.
 */
class CusparseError : public Error {
public:
    /**
     * Initializes a cuSPARSE error.
     *
     * @param file  The name of the offending source file
     * @param line  The source code line number where the error occurred
     * @param func  The name of the cuSPARSE routine that failed
     * @param error_code  The resulting cuSPARSE error code
     */
    CusparseError(const std::string& file, int line, const std::string& func,
                  int64 error_code)
        : Error(file, line, func + ": " + get_error(error_code))
    {}

private:
    static std::string get_error(int64 error_code);
};


/**
 * CufftError is thrown when a cuFFT routine throws a non-zero error code.
 */
class CufftError : public Error {
public:
    /**
     * Initializes a cuFFT error.
     *
     * @param file  The name of the offending source file
     * @param line  The source code line number where the error occurred
     * @param func  The name of the cuFFT routine that failed
     * @param error_code  The resulting cuFFT error code
     */
    CufftError(const std::string& file, int line, const std::string& func,
               int64 error_code)
        : Error(file, line, func + ": " + get_error(error_code))
    {}

private:
    static std::string get_error(int64 error_code);
};


/**
 * HipError is thrown when a HIP routine throws a non-zero error code.
 */
class HipError : public Error {
public:
    /**
     * Initializes a HIP error.
     *
     * @param file  The name of the offending source file
     * @param line  The source code line number where the error occurred
     * @param func  The name of the HIP routine that failed
     * @param error_code  The resulting HIP error code
     */
    HipError(const std::string& file, int line, const std::string& func,
             int64 error_code)
        : Error(file, line, func + ": " + get_error(error_code))
    {}

private:
    static std::string get_error(int64 error_code);
};


/**
 * HipblasError is thrown when a hipBLAS routine throws a non-zero error code.
 */
class HipblasError : public Error {
public:
    /**
     * Initializes a hipBLAS error.
     *
     * @param file  The name of the offending source file
     * @param line  The source code line number where the error occurred
     * @param func  The name of the hipBLAS routine that failed
     * @param error_code  The resulting hipBLAS error code
     */
    HipblasError(const std::string& file, int line, const std::string& func,
                 int64 error_code)
        : Error(file, line, func + ": " + get_error(error_code))
    {}

private:
    static std::string get_error(int64 error_code);
};


/**
 * HiprandError is thrown when a hipRAND routine throws a non-zero error code.
 */
class HiprandError : public Error {
public:
    /**
     * Initializes a hipRAND error.
     *
     * @param file  The name of the offending source file
     * @param line  The source code line number where the error occurred
     * @param func  The name of the hipRAND routine that failed
     * @param error_code  The resulting hipRAND error code
     */
    HiprandError(const std::string& file, int line, const std::string& func,
                 int64 error_code)
        : Error(file, line, func + ": " + get_error(error_code))
    {}

private:
    static std::string get_error(int64 error_code);
};


/**
 * HipsparseError is thrown when a hipSPARSE routine throws a non-zero error
 * code.
 */
class HipsparseError : public Error {
public:
    /**
     * Initializes a hipSPARSE error.
     *
     * @param file  The name of the offending source file
     * @param line  The source code line number where the error occurred
     * @param func  The name of the hipSPARSE routine that failed
     * @param error_code  The resulting hipSPARSE error code
     */
    HipsparseError(const std::string& file, int line, const std::string& func,
                   int64 error_code)
        : Error(file, line, func + ": " + get_error(error_code))
    {}

private:
    static std::string get_error(int64 error_code);
};


/**
 * HipfftError is thrown when a hipFFT routine throws a non-zero error code.
 */
class HipfftError : public Error {
public:
    /**
     * Initializes a hipFFT error.
     *
     * @param file  The name of the offending source file
     * @param line  The source code line number where the error occurred
     * @param func  The name of the hipFFT routine that failed
     * @param error_code  The resulting hipFFT error code
     */
    HipfftError(const std::string& file, int line, const std::string& func,
                int64 error_code)
        : Error(file, line, func + ": " + get_error(error_code))
    {}

private:
    static std::string get_error(int64 error_code);
};


/**
 * MetisError is thrown when METIS routine throws an error code.
 */
class MetisError : public Error {
public:
    /**
     * Initializes a METIS error.
     *
     * @param file  The name of the offending source file
     * @param line  The source code line number where the error occurred
     * @param func  The name of the METIS routine that failed
     * @param error  The resulting METIS error name
     */
    MetisError(const std::string& file, int line, const std::string& func,
               const std::string& error)
        : Error(file, line, func + ": " + error)
    {}
};


/**
 * DimensionMismatch is thrown if an operation is being applied to LinOps of
 * incompatible size.
 */
class DimensionMismatch : public Error {
public:
    /**
     * Initializes a dimension mismatch error.
     *
     * @param file  The name of the offending source file
     * @param line  The source code line number where the error occurred
     * @param func  The function name where the error occurred
     * @param first_name  The name of the first operator
     * @param first_rows  The output dimension of the first operator
     * @param first_cols  The input dimension of the first operator
     * @param second_name  The name of the second operator
     * @param second_rows  The output dimension of the second operator
     * @param second_cols  The input dimension of the second operator
     * @param clarification  An additional message describing the error further
     */
    DimensionMismatch(const std::string& file, int line,
                      const std::string& func, const std::string& first_name,
                      size_type first_rows, size_type first_cols,
                      const std::string& second_name, size_type second_rows,
                      size_type second_cols, const std::string& clarification)
        : Error(file, line,
                func + ": attempting to combine operators " + first_name +
                    " [" + std::to_string(first_rows) + " x " +
                    std::to_string(first_cols) + "] and " + second_name + " [" +
                    std::to_string(second_rows) + " x " +
                    std::to_string(second_cols) + "]: " + clarification)
    {}
};


/**
 * BadDimension is thrown if an operation is being applied to a LinOp
 * with bad dimensions.
 */
class BadDimension : public Error {
public:
    /**
     * Initializes a bad dimension error.
     *
     * @param file  The name of the offending source file
     * @param line  The source code line number where the error occurred
     * @param func  The function name where the error occurred
     * @param op_name  The name of the operator
     * @param op_num_rows  The row dimension of the operator
     * @param op_num_cols  The column dimension of the operator
     * @param clarification  An additional message further describing the error
     */
    BadDimension(const std::string& file, int line, const std::string& func,
                 const std::string& op_name, size_type op_num_rows,
                 size_type op_num_cols, const std::string& clarification)
        : Error(file, line,
                func + ": Object " + op_name + " has dimensions [" +
                    std::to_string(op_num_rows) + " x " +
                    std::to_string(op_num_cols) + "]: " + clarification)
    {}
};


/**
 * Error that denotes issues between block sizes and matrix dimensions
 *
 * \tparam IndexType  Type of index used by the linear algebra object that is
 *                    incompatible with the required block size.
 */
template <typename IndexType>
class BlockSizeError : public Error {
public:
    /**
     * @param file  The name of the offending source file
     * @param line  The source code line number where the error occurred
     * @param block_size  Size of small dense blocks in a matrix
     * @param size  The size that is not exactly divided by the block size
     */
    BlockSizeError(const std::string& file, const int line,
                   const int block_size, const IndexType size)
        : Error(file, line,
                "block size = " + std::to_string(block_size) +
                    ", size = " + std::to_string(size))
    {}
};


/**
 * ValueMismatch is thrown if two values are not equal.
 */
class ValueMismatch : public Error {
public:
    /**
     * Initializes a value mismatch error.
     *
     * @param file  The name of the offending source file
     * @param line  The source code line number where the error occurred
     * @param func  The function name where the error occurred
     * @param val1  The first value to be compared.
     * @param val2  The second value to be compared.
     * @param clarification  An additional message further describing the error
     */
    ValueMismatch(const std::string& file, int line, const std::string& func,
                  size_type val1, size_type val2,
                  const std::string& clarification)
        : Error(file, line,
                func + ": Value mismatch : " + std::to_string(val1) + " and " +
                    std::to_string(val2) + " : " + clarification)
    {}
};


/**
 * AllocationError is thrown if a memory allocation fails.
 */
class AllocationError : public Error {
public:
    /**
     * Initializes an allocation error.
     *
     * @param file  The name of the offending source file
     * @param line  The source code line number where the error occurred
     * @param device  The device on which the error occurred
     * @param bytes  The size of the memory block whose allocation failed.
     */
    AllocationError(const std::string& file, int line,
                    const std::string& device, size_type bytes)
        : Error(file, line,
                device + ": failed to allocate memory block of " +
                    std::to_string(bytes) + "B")
    {}
};


/**
 * OutOfBoundsError is thrown if a memory access is detected to be
 * out-of-bounds.
 */
class OutOfBoundsError : public Error {
public:
    /**
     * Initializes an OutOfBoundsError.
     *
     * @param file  The name of the offending source file
     * @param line  The source code line number where the error occurred
     * @param index  The position that was accessed
     * @param bound  The first out-of-bound index
     */
    OutOfBoundsError(const std::string& file, int line, size_type index,
                     size_type bound)
        : Error(file, line,
                "trying to access index " + std::to_string(index) +
                    " in a memory block of " + std::to_string(bound) +
                    " elements")
    {}
};


/**
 * OverflowError is thrown when an index calculation for storage requirements
 * overflows. This most likely means that the index type is too small.
 */
class OverflowError : public Error {
public:
    /**
     * @param file  The name of the offending source file
     * @param line  The source code line number where the error occurred
     * @param index_type  The integer type that overflowed
     */
    OverflowError(const std::string& file, const int line,
                  const std::string& index_type)
        : Error(file, line, "Overflowing " + index_type)
    {}
};


/**
 * StreamError is thrown if accessing a stream failed.
 */
class StreamError : public Error {
public:
    /**
     * Initializes a file access error.
     *
     * @param file  The name of the offending source file
     * @param line  The source code line number where the error occurred
     * @param func  The name of the function that tried to access the file
     * @param message  The error message
     */
    StreamError(const std::string& file, int line, const std::string& func,
                const std::string& message)
        : Error(file, line, func + ": " + message)
    {}
};


/**
 * KernelNotFound is thrown if Ginkgo cannot find a kernel which satisfies the
 * criteria imposed by the input arguments.
 */
class KernelNotFound : public Error {
public:
    /**
     * Initializes a KernelNotFound error.
     *
     * @param file  The name of the offending source file
     * @param line  The source code line number where the error occurred
     * @param func  The name of the function where the error occurred
     */
    KernelNotFound(const std::string& file, int line, const std::string& func)
        : Error(file, line, func + ": unable to find an eligible kernel")
    {}
};


/**
 * Exception throws if a matrix does not have a property required by a
 * numerical method.
 *
 * Currently, a message is specified at the call-site manually.
 */
class UnsupportedMatrixProperty : public Error {
public:
    /**
     * Initializes the UnsupportedMatrixProperty error.
     *
     * @param file  The name of the offending source file
     * @param line  The source code line number where the error occurred
     * @param msg  A message describing the property required.
     */
    UnsupportedMatrixProperty(const std::string& file, const int line,
                              const std::string& msg)
        : Error(file, line, msg)
    {}
};


/** Exception thrown if an object is in an invalid state. */
class InvalidStateError : public Error {
public:
    /**
     * Initializes an invalid state error.
     *
     * @param file  The name of the offending source file
     * @param line  The source code line number where the error occurred
     * @param func  The function name where the error occurred
     * @param clarification  A message describing the invalid state
     */
    InvalidStateError(const std::string& file, int line,
                      const std::string& func, const std::string& clarification)
        : Error(file, line,
                func + ": Invalid state encountered : " + clarification)
    {}
};


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_EXCEPTION_HPP_
