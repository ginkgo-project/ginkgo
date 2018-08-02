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

#ifndef GKO_CORE_TYPES_HPP_
#define GKO_CORE_TYPES_HPP_


#include <climits>
#include <cstddef>
#include <cstdint>


#include <complex>
#include <type_traits>


#ifdef __CUDACC__
#define GKO_ATTRIBUTES __host__ __device__
#define GKO_INLINE __forceinline__
#else
#define GKO_ATTRIBUTES
#define GKO_INLINE inline
#endif  // __CUDACC__


namespace gko {


/**
 * Integral type used for allocation quantities.
 */
using size_type = std::size_t;


/**
 * 8-bit signed integral type.
 */
using int8 = std::int8_t;

/**
 * 16-bit signed integral type.
 */
using int16 = std::int16_t;


/**
 * 32-bit signed integral type.
 */
using int32 = std::int32_t;


/**
 * 64-bit signed integral type.
 */
using int64 = std::int64_t;


/**
 * 8-bit unsigned integral type.
 */
using uint8 = std::uint8_t;

/**
 * 16-bit unsigned integral type.
 */
using uint16 = std::uint16_t;


/**
 * 32-bit unsigned integral type.
 */
using uint32 = std::uint32_t;


/**
 * 64-bit unsigned integral type.
 */
using uint64 = std::uint64_t;


class half;


/**
 * Half precision floating point type.
 */
using float16 = half;


/**
 * Single precision floating point type.
 */
using float32 = float;


/**
 * Double precision floating point type.
 */
using float64 = double;


/**
 * The most precise floating-point type.
 */
using full_precision = double;


/**
 * Precision used if no precision is explicitly specified.
 */
using default_precision = double;


/**
 * Number of bits in a byte
 */
constexpr size_type byte_size = CHAR_BIT;


/**
 * A type representing the dimensions of a linear operator.
 */
struct dim {
    /**
     * Creates a dimension object with both rows and columns set to the same
     * value.
     *
     * @param size  the number of rows and columns
     */
    GKO_ATTRIBUTES explicit dim(size_type size = {}) : dim(size, size) {}

    /**
     * Creates a dimension object with the specified number of rows and columns.
     *
     * @param nrows  the number of rows
     * @param ncols  the number of columsn
     */
    GKO_ATTRIBUTES dim(size_type nrows, size_type ncols)
        : num_rows{nrows}, num_cols{ncols}
    {}

    /**
     * Checks if both dimensions are greater than 0.
     *
     * @return true if and only if both dimensions are greater than 9.
     */
    GKO_ATTRIBUTES operator bool() const
    {
        return num_rows > 0 && num_cols > 0;
    };

    /**
     * Number of rows of the operator.
     *
     * In other words, the dimension of its codomain.
     */
    size_type num_rows;

    /**
     * Number of columns of the operator.
     *
     * In other words, the dimension of its domain.
     */
    size_type num_cols;
};


/**
 * Checks if two dim objects are equal.
 *
 * @param x  first object
 * @param y  second object
 *
 * @return true if and only if both the number of rows and the number of columns
 *         of both objects match.
 */
GKO_ATTRIBUTES GKO_INLINE bool operator==(const dim &x, const dim &y)
{
    return x.num_rows == y.num_rows && x.num_cols == y.num_cols;
}


/**
 * Checks if two dim objects are different.
 *
 * @param x  first object
 * @param y  second object
 *
 * @return `!(x == y)`
 */
GKO_ATTRIBUTES GKO_INLINE bool operator!=(const dim &x, const dim &y)
{
    return !(x == y);
}


/**
 * Multiplies two dim objects.
 *
 * @param x  first object
 * @param y  second object
 *
 * @return a dim object representing the size of the tensor product `x * y`
 */
GKO_ATTRIBUTES GKO_INLINE dim operator*(const dim &x, const dim &y)
{
    return dim{x.num_rows * y.num_rows, x.num_cols * y.num_cols};
}

/**
 * Returns a dim object with num_rows and num_columns values swapped.
 *
 * @return a dim object with num_rows and num_columns values swapped.
 */
GKO_ATTRIBUTES GKO_INLINE dim transpose(const dim &dimensions) noexcept
{
    return {dimensions.num_cols, dimensions.num_rows};
}


/**
 * Calls a given macro for each executor type.
 *
 * The macro should take two parameters:
 *
 * -   the first one is replaced with the executor class name
 * -   the second one with the executor short name (used for namespace name)
 *
 * @param _enable_macro  macro name which will be called
 *
 * @note  the macro is not called for ReferenceExecutor
 */
#define GKO_ENABLE_FOR_ALL_EXECUTORS(_enable_macro) \
    _enable_macro(OmpExecutor, omp);                \
    _enable_macro(CudaExecutor, cuda)


/**
 * Instantiates a template for each value type compiled by Ginkgo.
 *
 * @param _macro  A macro which expands the template instantiation
 *                (not including the leading `template` specifier).
 *                Should take one argument, which is replaced by the
 *                value type.
 */
#define GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(_macro) \
    template _macro(float);                         \
    template _macro(double);                        \
    template _macro(std::complex<float>);           \
    template _macro(std::complex<double>)


/**
 * Instantiates a template for each index type compiled by Ginkgo.
 *
 * @param _macro  A macro which expands the template instantiation
 *                (not including the leading `template` specifier).
 *                Should take one argument, which is replaced by the
 *                value type.
 */
#define GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(_macro) \
    template _macro(int32);                         \
    template _macro(int64)


/**
 *define typenames for row_major accessor
 */
#define row_major_float range<accessor::row_major<float, 2>>
#define row_major_double range<accessor::row_major<double, 2>>
#define row_major_complex_float \
    range<accessor::row_major<std::complex<float>, 2>>
#define row_major_complex_double \
    range<accessor::row_major<std::complex<double>, 2>>


/**
 * Instantiates a template for each value and index type compiled by Ginkgo.
 *
 * @param _macro  A macro which expands the template instantiation
 *                (not including the leading `template` specifier).
 *                Should take two arguments, which are replaced by the
 *                value and index types.
 */
#define GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(_macro) \
    template _macro(float, int32);                            \
    template _macro(double, int32);                           \
    template _macro(std::complex<float>, int32);              \
    template _macro(std::complex<double>, int32);             \
    template _macro(float, int64);                            \
    template _macro(double, int64);                           \
    template _macro(std::complex<float>, int64);              \
    template _macro(std::complex<double>, int64)


/**
 *Instantiates a template for each value and accessor type compiled by
 Ginkgo.
 *
 *@param _macro  A macro which expands the template instantiation
 *                (not including the leading `template` specifier).
 *                Should take two arguments, which are replaced by the
 *                value and index types.
 */
#define GKO_INSTANTIATE_FOR_EACH_VALUE_AND_ACCESSOR_TYPE(_macro)   \
    template _macro(float, row_major_float);                       \
    template _macro(double, row_major_double);                     \
    template _macro(std::complex<float>, row_major_complex_float); \
    template _macro(std::complex<double>, row_major_complex_double)


}  // namespace gko


#endif  // GKO_CORE_TYPES_HPP_
