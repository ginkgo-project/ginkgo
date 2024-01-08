// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_TYPES_HPP_
#define GKO_PUBLIC_CORE_BASE_TYPES_HPP_


#include <array>
#include <cassert>
#include <climits>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>


#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#endif  // __HIPCC__


// Macros for handling different compilers / architectures uniformly
#if defined(__CUDACC__) || defined(__HIPCC__)
#define GKO_ATTRIBUTES __host__ __device__
#define GKO_INLINE __forceinline__
#define GKO_RESTRICT __restrict__
#else
#define GKO_ATTRIBUTES
#define GKO_INLINE inline
#define GKO_RESTRICT
#endif  // defined(__CUDACC__) || defined(__HIPCC__)


#if (defined(__CUDA_ARCH__) && defined(__APPLE__)) || \
    defined(__HIP_DEVICE_COMPILE__)

#ifdef NDEBUG
#define GKO_ASSERT(condition) ((void)0)
#else  // NDEBUG
// Poor man's assertions on GPUs for MACs. They won't terminate the program
// but will at least print something on the screen
#define GKO_ASSERT(condition)                                               \
    ((condition)                                                            \
         ? ((void)0)                                                        \
         : ((void)printf("%s: %d: %s: Assertion `" #condition "' failed\n", \
                         __FILE__, __LINE__, __func__)))
#endif  // NDEBUG

#else  // (defined(__CUDA_ARCH__) && defined(__APPLE__)) ||
       // defined(__HIP_DEVICE_COMPILE__)

// Handle assertions normally on other systems
#define GKO_ASSERT(condition) assert(condition)

#endif  // (defined(__CUDA_ARCH__) && defined(__APPLE__)) ||
        // defined(__HIP_DEVICE_COMPILE__)


// Handle deprecated notices correctly on different systems
// clang-format off
#define GKO_DEPRECATED(_msg) [[deprecated(_msg)]]
#ifdef __NVCOMPILER
#define GKO_BEGIN_DISABLE_DEPRECATION_WARNINGS _Pragma("diag_suppress 1445")
#define GKO_END_DISABLE_DEPRECATION_WARNINGS _Pragma("diag_warning 1445")
#elif defined(__GNUC__) || defined(__clang__)
#define GKO_BEGIN_DISABLE_DEPRECATION_WARNINGS                      \
    _Pragma("GCC diagnostic push")                                  \
    _Pragma("GCC diagnostic ignored \"-Wdeprecated-declarations\"")
#define GKO_END_DISABLE_DEPRECATION_WARNINGS _Pragma("GCC diagnostic pop")
#elif defined(_MSC_VER)
#define GKO_BEGIN_DISABLE_DEPRECATION_WARNINGS        \
    _Pragma("warning(push)")                          \
    _Pragma("warning(disable : 5211 4973 4974 4996)")
#define GKO_END_DISABLE_DEPRECATION_WARNINGS _Pragma("warning(pop)")
#else
#define GKO_BEGIN_DISABLE_DEPRECATION_WARNINGS
#define GKO_END_DISABLE_DEPRECATION_WARNINGS
#endif
// clang-format on


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


/**
 * Unsigned integer type capable of holding a pointer to void
 */
using uintptr = std::uintptr_t;


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
 * Evaluates if all template arguments Args fulfill std::is_integral. If that is
 * the case, this class inherits from `std::true_type`, otherwise, it inherits
 * from `std::false_type`.
 * If no values are passed in, `std::true_type` is inherited from.
 *
 * @tparam Args...  Arguments to test for std::is_integral
 */
template <typename... Args>
struct are_all_integral : public std::true_type {};

template <typename First, typename... Args>
struct are_all_integral<First, Args...>
    : public std::conditional<std::is_integral<std::decay_t<First>>::value,
                              are_all_integral<Args...>,
                              std::false_type>::type {};


/**
 * This class is used to encode storage precisions of low precision algorithms.
 *
 * Some algorithms in Ginkgo can improve their performance by storing parts of
 * the data in lower precision, while doing computation in full precision. This
 * class is used to encode the precisions used to store the data. From the
 * user's perspective, some algorithms can provide a parameter for fine-tuning
 * the storage precision. Commonly, the special value returned by
 * precision_reduction::autodetect() should be used to allow the algorithm to
 * automatically choose an appropriate value, though manually selected values
 * can be used for fine-tuning.
 *
 * In general, a lower precision floating point value can be obtained by either
 * dropping some of the insignificant bits of the significand (keeping the same
 * number of exponent bits, and thus preserving the range of representable
 * values) or using one of the hardware or software supported conversions
 * between IEEE formats, such as double to float or float to half (reducing both
 * the number of exponent, as well as significand bits, and thus decreasing the
 * range of representable values).
 *
 * The precision_reduction class encodes the lower precision format relative to
 * the base precision used and the algorithm in question. The encoding is
 * done by specifying the amount of range non-preserving conversions and the
 * amount of range preserving conversions that should be done on the base
 * precision to obtain the lower precision format. For example, starting with a
 * double precision value (11 exp, 52 sig. bits), the encoding specifying 1
 * non-preserving conversion and 1 preserving conversion would first use a
 * hardware-supported non-preserving conversion to obtain a single precision
 * value (8 exp, 23 sig. bits), followed by a preserving bit truncation to
 * obtain a value with 8 exponent and 7 significand bits. Note that
 * non-preserving conversion are always done first, as preserving conversions
 * usually result in datatypes that are not supported by builtin conversions
 * (thus, it is generally not possible to apply a non-preserving conversion to
 * the result of a preserving conversion).
 *
 * If the specified conversion is not supported by the algorithm, it will most
 * likely fall back to using full precision for storing the data. Refer to the
 * documentation of specific algorithms using this class for details about such
 * special cases.
 */
class precision_reduction {
public:
    /**
     * The underlying datatype used to store the encoding.
     */
    using storage_type = uint8;

private:
    static constexpr auto nonpreserving_bits = 4u;
    static constexpr auto preserving_bits =
        byte_size * sizeof(storage_type) - nonpreserving_bits;
    static constexpr auto nonpreserving_mask =
        storage_type{(0x1 << nonpreserving_bits) - 1};
    static constexpr auto preserving_mask =
        storage_type{(0x1 << preserving_bits) - 1} << nonpreserving_bits;

public:
    /**
     * Creates a default precision_reduction encoding.
     *
     * This encoding represents the case where no conversions are performed.
     */
    GKO_ATTRIBUTES constexpr precision_reduction() noexcept : data_{0x0} {}

    /**
     * Creates a precision_reduction encoding with the specified number of
     * conversions.
     *
     * @param preserving  the number of range preserving conversion
     * @param nonpreserving  the number of range non-preserving conversions
     */
    GKO_ATTRIBUTES constexpr precision_reduction(
        storage_type preserving, storage_type nonpreserving) noexcept
        : data_((GKO_ASSERT(preserving < (0x1 << preserving_bits) - 1),
                 GKO_ASSERT(nonpreserving < (0x1 << nonpreserving_bits) - 1),
                 (preserving << nonpreserving_bits) | nonpreserving))
    {}

    /**
     * Extracts the raw data of the encoding.
     *
     * @return the raw data of the encoding
     */
    GKO_ATTRIBUTES constexpr operator storage_type() const noexcept
    {
        return data_;
    }

    /**
     * Returns the number of preserving conversions in the encoding.
     *
     * @return the number of preserving conversions in the encoding.
     */
    GKO_ATTRIBUTES constexpr storage_type get_preserving() const noexcept
    {
        return (data_ & preserving_mask) >> nonpreserving_bits;
    }

    /**
     * Returns the number of non-preserving conversions in the encoding.
     *
     * @return the number of non-preserving conversions in the encoding.
     */
    GKO_ATTRIBUTES constexpr storage_type get_nonpreserving() const noexcept
    {
        return data_ & nonpreserving_mask;
    }

    /**
     * Returns a special encoding which instructs the algorithm to automatically
     * detect the best precision.
     *
     * @return  a special encoding instructing the algorithm to automatically
     *          detect the best precision.
     */
    GKO_ATTRIBUTES constexpr static precision_reduction autodetect() noexcept
    {
        return precision_reduction{preserving_mask | nonpreserving_mask};
    }

    /**
     * Returns the common encoding of input encodings.
     *
     * The common encoding is defined as the encoding that does not have more
     * preserving, nor non-preserving conversions than the input encodings.
     *
     * @param x  an encoding
     * @param y  an encoding
     *
     * @return the common encoding of `x` and `y`
     */
    GKO_ATTRIBUTES constexpr static precision_reduction common(
        precision_reduction x, precision_reduction y) noexcept
    {
        return precision_reduction(
            min(x.data_ & preserving_mask, y.data_ & preserving_mask) |
            min(x.data_ & nonpreserving_mask, y.data_ & nonpreserving_mask));
    }

private:
    GKO_ATTRIBUTES constexpr precision_reduction(storage_type data)
        : data_{data}
    {}

    GKO_ATTRIBUTES constexpr static storage_type min(storage_type x,
                                                     storage_type y) noexcept
    {
        return x < y ? x : y;
    }

    storage_type data_;
};


/**
 * Checks if two precision_reduction encodings are equal.
 *
 * @param x  an encoding
 * @param y  an encoding
 *
 * @return true if and only if `x` and `y` are the same encodings
 */
GKO_ATTRIBUTES constexpr bool operator==(precision_reduction x,
                                         precision_reduction y) noexcept
{
    using st = precision_reduction::storage_type;
    return static_cast<st>(x) == static_cast<st>(y);
}


/**
 * Checks if two precision_reduction encodings are different.
 *
 * @param x  an encoding
 * @param y  an encoding
 *
 * @return true if and only if `x` and `y` are different encodings.
 */
GKO_ATTRIBUTES constexpr bool operator!=(precision_reduction x,
                                         precision_reduction y) noexcept
{
    using st = precision_reduction::storage_type;
    return static_cast<st>(x) != static_cast<st>(y);
}


/**
 * Calls a given macro for each executor type for a given kernel.
 *
 * The macro should take two parameters:
 *
 * -   the first one is replaced with the executor class name
 * -   the second one with the name of the kernel to be bound
 *
 * @param _enable_macro  macro name which will be called
 *
 * @note  the macro is not called for ReferenceExecutor
 */
#define GKO_ENABLE_FOR_ALL_EXECUTORS(_enable_macro) \
    _enable_macro(OmpExecutor, omp);                \
    _enable_macro(HipExecutor, hip);                \
    _enable_macro(DpcppExecutor, dpcpp);            \
    _enable_macro(CudaExecutor, cuda)


/**
 * Instantiates a template for each non-complex value type compiled by Ginkgo.
 *
 * @param _macro  A macro which expands the template instantiation
 *                (not including the leading `template` specifier).
 *                Should take one argument, which is replaced by the
 *                value type.
 */
#if GINKGO_DPCPP_SINGLE_MODE
#define GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(_macro) \
    template _macro(float);                                     \
    template <>                                                 \
    _macro(double) GKO_NOT_IMPLEMENTED
#else
#define GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(_macro) \
    template _macro(float);                                     \
    template _macro(double)
#endif


/**
 * Instantiates a template for each value type compiled by Ginkgo.
 *
 * @param _macro  A macro which expands the template instantiation
 *                (not including the leading `template` specifier).
 *                Should take one argument, which is replaced by the
 *                value type.
 */
#if GINKGO_DPCPP_SINGLE_MODE
#define GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(_macro)          \
    GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(_macro); \
    template _macro(std::complex<float>);                    \
    template <>                                              \
    _macro(std::complex<double>) GKO_NOT_IMPLEMENTED
#else
#define GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(_macro)          \
    GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(_macro); \
    template _macro(std::complex<float>);                    \
    template _macro(std::complex<double>)
#endif


/**
 * Instantiates a template for each value and scalar type compiled by Ginkgo.
 * This means all value and scalar type combinations for which
 * `value = scalar * value` is well-defined.
 *
 * @param _macro  A macro which expands the template instantiation
 *                (not including the leading `template` specifier).
 *                Should take two arguments, which are replaced by the
 *                value and scalar type, respectively.
 */
#if GINKGO_DPCPP_SINGLE_MODE
#define GKO_INSTANTIATE_FOR_EACH_VALUE_AND_SCALAR_TYPE(_macro)              \
    template _macro(float, float);                                          \
    template <>                                                             \
    _macro(double, double) GKO_NOT_IMPLEMENTED;                             \
    template _macro(std::complex<float>, std::complex<float>);              \
    template <>                                                             \
    _macro(std::complex<double>, std::complex<double>) GKO_NOT_IMPLEMENTED; \
    template _macro(std::complex<float>, float);                            \
    template <>                                                             \
    _macro(std::complex<double>, double) GKO_NOT_IMPLEMENTED;
#else
#define GKO_INSTANTIATE_FOR_EACH_VALUE_AND_SCALAR_TYPE(_macro)   \
    template _macro(float, float);                               \
    template _macro(double, double);                             \
    template _macro(std::complex<float>, std::complex<float>);   \
    template _macro(std::complex<double>, std::complex<double>); \
    template _macro(std::complex<float>, float);                 \
    template _macro(std::complex<double>, double)
#endif


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
 * Instantiates a template for each non-complex value and index type compiled by
 * Ginkgo.
 *
 * @param _macro  A macro which expands the template instantiation
 *                (not including the leading `template` specifier).
 *                Should take two arguments, which are replaced by the
 *                value and index types.
 */
#if GINKGO_DPCPP_SINGLE_MODE
#define GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_INDEX_TYPE(_macro) \
    template _macro(float, int32);                                        \
    template <>                                                           \
    _macro(double, int32) GKO_NOT_IMPLEMENTED;                            \
    template _macro(float, int64);                                        \
    template <>                                                           \
    _macro(double, int64) GKO_NOT_IMPLEMENTED
#else
#define GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_INDEX_TYPE(_macro) \
    template _macro(float, int32);                                        \
    template _macro(double, int32);                                       \
    template _macro(float, int64);                                        \
    template _macro(double, int64)
#endif

#if GINKGO_DPCPP_SINGLE_MODE
#define GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INT32_TYPE(_macro) \
    template _macro(float, int32);                            \
    template <>                                               \
    _macro(double, int32) GKO_NOT_IMPLEMENTED;                \
    template _macro(std::complex<float>, int32);              \
    template <>                                               \
    _macro(std::complex<double>, int32) GKO_NOT_IMPLEMENTED
#else
#define GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INT32_TYPE(_macro) \
    template _macro(float, int32);                            \
    template _macro(double, int32);                           \
    template _macro(std::complex<float>, int32);              \
    template _macro(std::complex<double>, int32)
#endif


/**
 * Instantiates a template for each value and index type compiled by Ginkgo.
 *
 * @param _macro  A macro which expands the template instantiation
 *                (not including the leading `template` specifier).
 *                Should take two arguments, which are replaced by the
 *                value and index types.
 */
#if GINKGO_DPCPP_SINGLE_MODE
#define GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(_macro)          \
    GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_INDEX_TYPE(_macro); \
    template _macro(std::complex<float>, int32);                       \
    template <>                                                        \
    _macro(std::complex<double>, int32) GKO_NOT_IMPLEMENTED;           \
    template _macro(std::complex<float>, int64);                       \
    template <>                                                        \
    _macro(std::complex<double>, int64) GKO_NOT_IMPLEMENTED
#else
#define GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(_macro)          \
    GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_INDEX_TYPE(_macro); \
    template _macro(std::complex<float>, int32);                       \
    template _macro(std::complex<double>, int32);                      \
    template _macro(std::complex<float>, int64);                       \
    template _macro(std::complex<double>, int64)
#endif


/**
 * Instantiates a template for each non-complex value, local and global index
 * type compiled by Ginkgo.
 *
 * @param _macro  A macro which expands the template instantiation
 *                (not including the leading `template` specifier).
 *                Should take three arguments, which are replaced by the
 *                value, the local and the global index types.
 */
#if GINKGO_DPCPP_SINGLE_MODE
#define GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE( \
    _macro)                                                                     \
    template _macro(float, int32, int32);                                       \
    template _macro(float, int32, int64);                                       \
    template _macro(float, int64, int64);                                       \
    template <>                                                                 \
    _macro(double, int32, int32) GKO_NOT_IMPLEMENTED;                           \
    template <>                                                                 \
    _macro(double, int32, int64) GKO_NOT_IMPLEMENTED;                           \
    template <>                                                                 \
    _macro(double, int64, int64) GKO_NOT_IMPLEMENTED
#else
#define GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE( \
    _macro)                                                                     \
    template _macro(float, int32, int32);                                       \
    template _macro(float, int32, int64);                                       \
    template _macro(float, int64, int64);                                       \
    template _macro(double, int32, int32);                                      \
    template _macro(double, int32, int64);                                      \
    template _macro(double, int64, int64)
#endif


/**
 * Instantiates a template for each value and index type compiled by Ginkgo.
 *
 * @param _macro  A macro which expands the template instantiation
 *                (not including the leading `template` specifier).
 *                Should take two arguments, which are replaced by the
 *                value and index types.
 */
#if GINKGO_DPCPP_SINGLE_MODE
#define GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(_macro)  \
    GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE( \
        _macro);                                                            \
    template _macro(std::complex<float>, int32, int32);                     \
    template _macro(std::complex<float>, int32, int64);                     \
    template _macro(std::complex<float>, int64, int64);                     \
    template <>                                                             \
    _macro(std::complex<double>, int32, int32) GKO_NOT_IMPLEMENTED;         \
    template <>                                                             \
    _macro(std::complex<double>, int32, int64) GKO_NOT_IMPLEMENTED;         \
    template <>                                                             \
    _macro(std::complex<double>, int64, int64) GKO_NOT_IMPLEMENTED
#else
#define GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(_macro)  \
    GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE( \
        _macro);                                                            \
    template _macro(std::complex<float>, int32, int32);                     \
    template _macro(std::complex<float>, int32, int64);                     \
    template _macro(std::complex<float>, int64, int64);                     \
    template _macro(std::complex<double>, int32, int32);                    \
    template _macro(std::complex<double>, int32, int64);                    \
    template _macro(std::complex<double>, int64, int64)
#endif


#if GINKGO_DPCPP_SINGLE_MODE
#define GKO_INSTANTIATE_FOR_EACH_VALUE_CONVERSION(_macro)                  \
    template <>                                                            \
    _macro(float, double) GKO_NOT_IMPLEMENTED;                             \
    template <>                                                            \
    _macro(double, float) GKO_NOT_IMPLEMENTED;                             \
    template <>                                                            \
    _macro(std::complex<float>, std::complex<double>) GKO_NOT_IMPLEMENTED; \
    template <>                                                            \
    _macro(std::complex<double>, std::complex<float>) GKO_NOT_IMPLEMENTED


#define GKO_INSTANTIATE_FOR_EACH_VALUE_CONVERSION_OR_COPY(_macro) \
    GKO_INSTANTIATE_FOR_EACH_VALUE_CONVERSION(_macro);            \
    template _macro(float, float);                                \
    template <>                                                   \
    _macro(double, double) GKO_NOT_IMPLEMENTED;                   \
    template _macro(std::complex<float>, std::complex<float>);    \
    template <>                                                   \
    _macro(std::complex<double>, std::complex<double>) GKO_NOT_IMPLEMENTED
#else
/**
 * Instantiates a template for each value type conversion pair compiled by
 * Ginkgo.
 *
 * @param _macro  A macro which expands the template instantiation
 *                (not including the leading `template` specifier).
 *                Should take two arguments `src` and `dst`, which
 *                are replaced by the source and destination value type.
 */
#define GKO_INSTANTIATE_FOR_EACH_VALUE_CONVERSION(_macro)       \
    template _macro(float, double);                             \
    template _macro(double, float);                             \
    template _macro(std::complex<float>, std::complex<double>); \
    template _macro(std::complex<double>, std::complex<float>)


/**
 * Instantiates a template for each value type conversion or copy pair compiled
 * by Ginkgo.
 *
 * @param _macro  A macro which expands the template instantiation
 *                (not including the leading `template` specifier).
 *                Should take two arguments `src` and `dst`, which
 *                are replaced by the source and destination value type.
 */
#define GKO_INSTANTIATE_FOR_EACH_VALUE_CONVERSION_OR_COPY(_macro) \
    GKO_INSTANTIATE_FOR_EACH_VALUE_CONVERSION(_macro);            \
    template _macro(float, float);                                \
    template _macro(double, double);                              \
    template _macro(std::complex<float>, std::complex<float>);    \
    template _macro(std::complex<double>, std::complex<double>)
#endif


/**
 * Instantiates a template for each value type pair compiled by Ginkgo.
 *
 * @param _macro  A macro which expands the template instantiation
 *                (not including the leading `template` specifier).
 *                Should take two arguments, which are replaced by the
 *                value and index types.
 */
#define GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_PAIR(_macro)       \
    template _macro(float, float);                             \
    template _macro(double, double);                           \
    template _macro(std::complex<float>, float);               \
    template _macro(std::complex<double>, double);             \
    template _macro(std::complex<float>, std::complex<float>); \
    template _macro(std::complex<double>, std::complex<double>)


/**
 * Instantiates a template for each combined value and index type compiled by
 * Ginkgo.
 *
 * @param _macro  A macro which expands the template instantiation
 *                (not including the leading `template` specifier).
 *                Should take two arguments, which are replaced by the
 *                value and index types.
 */
#define GKO_INSTANTIATE_FOR_EACH_COMBINED_VALUE_AND_INDEX_TYPE(_macro) \
    template _macro(char, char);                                       \
    template _macro(int32, int32);                                     \
    template _macro(int64, int64);                                     \
    template _macro(unsigned int, unsigned int);                       \
    template _macro(unsigned long, unsigned long);                     \
    template _macro(float, float);                                     \
    template _macro(double, double);                                   \
    template _macro(long double, long double);                         \
    template _macro(std::complex<float>, std::complex<float>);         \
    template _macro(std::complex<double>, std::complex<double>)

/**
 * Instantiates a template for each value and index type compiled by Ginkgo.
 *
 * @param _macro  A macro which expands the template instantiation
 *                (not including the leading `template` specifier).
 *                Should take two arguments, which are replaced by the
 *                value and index types.
 */
#define GKO_INSTANTIATE_FOR_EACH_POD_TYPE(_macro) \
    template _macro(float);                       \
    template _macro(double);                      \
    template _macro(std::complex<float>);         \
    template _macro(std::complex<double>);        \
    template _macro(size_type);                   \
    template _macro(bool);                        \
    template _macro(int32);                       \
    template _macro(int64)


/**
 * Instantiates a template for each normal type
 *
 * @param _macro  A macro which expands the template instantiation
 *                (not including the leading `template` specifier).
 *                Should take one argument, which is replaced by the
 *                value type.
 */
#define GKO_INSTANTIATE_FOR_EACH_TEMPLATE_TYPE(_macro) \
    GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(_macro);       \
    GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(_macro);       \
    template _macro(gko::size_type)


/**
 * Value for an invalid signed index type.
 */
template <typename IndexType>
inline constexpr GKO_ATTRIBUTES IndexType invalid_index()
{
    static_assert(std::is_signed<IndexType>::value,
                  "IndexType needs to be signed");
    return static_cast<IndexType>(-1);
}


namespace experimental {
namespace distributed {


/**
 * Index type for enumerating processes in a distributed application
 *
 * Conforms to the MPI C interface of e.g. MPI rank or size
 */
using comm_index_type = int;


/**
 * Instantiates a template for each valid combination of local and global index
 * type
 *
 * @param _macro  A macro which expands the template instantiation
 *                (not including the leading `template` specifier).
 *                Should take two arguments, where the first is replaced by the
 *                local index type and the second by the global index type.
 */
#define GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(_macro) \
    template _macro(int32, int32);                               \
    template _macro(int32, int64);                               \
    template _macro(int64, int64)


}  // namespace distributed
}  // namespace experimental
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_TYPES_HPP_
