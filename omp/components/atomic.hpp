// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_OMP_COMPONENTS_ATOMIC_HPP_
#define GKO_OMP_COMPONENTS_ATOMIC_HPP_


#include <type_traits>

#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace kernels {
namespace omp {


template <typename ValueType,
          std::enable_if_t<!is_complex<ValueType>()>* = nullptr>
void atomic_add(ValueType& out, ValueType val)
{
#pragma omp atomic
    out += val;
}

template <typename ValueType,
          std::enable_if_t<is_complex<ValueType>()>* = nullptr>
void atomic_add(ValueType& out, ValueType val)
{
    // The C++ standard explicitly allows casting complex<double>* to double*
    // [complex.numbers.general]
    auto values = reinterpret_cast<gko::remove_complex<ValueType>*>(&out);
    atomic_add(values[0], real(val));
    atomic_add(values[1], imag(val));
}


template <typename ResultType, typename ValueType>
inline ResultType copy_cast(const ValueType& val)
{
    static_assert(
        sizeof(ValueType) == sizeof(ResultType) &&
            std::alignment_of_v<ResultType> == std::alignment_of_v<ValueType>,
        "only copy the same alignment and size type");
    ResultType res;
    std::memcpy(&res, &val, sizeof(ValueType));
    return res;
}


template <>
inline void atomic_add(float16& out, float16 val)
{
#ifdef __NVCOMPILER
// NVC++ uses atomic capture on uint16 leads the following error.
// use of undefined value '%L.B*' br label %L.B* !llvm.loop !*, !dbg !*
#pragma omp critical
    {
        out += val;
    }
#else
    static_assert(
        sizeof(float16) == sizeof(uint16_t) &&
            std::alignment_of_v<uint16_t> == std::alignment_of_v<float16>,
        "half does not fulfill the requirement of reinterpret_cast to half or "
        "vice versa.");
    // It is undefined behavior with reinterpret_cast, but we do not have any
    // workaround when the #omp atomic does not support custom precision
    uint16_t* address_as_converter = reinterpret_cast<uint16_t*>(&out);
    uint16_t old = *address_as_converter;
    uint16_t assumed;
    do {
        assumed = old;
        auto answer = copy_cast<uint16_t>(copy_cast<float16>(assumed) + val);
#pragma omp atomic capture
        {
            old = *address_as_converter;
            *address_as_converter = (old == assumed) ? answer : old;
        }
    } while (assumed != old);
#endif
}


// There is an error in Clang 17 which prevents us from merging the
// implementation of double and float. The compiler will throw an error if the
// templated version is implemented. GCC doesn't throw an error.
inline void store(double* addr, double val)
{
#pragma omp atomic write
    *addr = val;
}

inline void store(float* addr, float val)
{
#pragma omp atomic write
    *addr = val;
}

inline void store(int32* addr, int32 val)
{
#pragma omp atomic write
    *addr = val;
}

inline void store(int64* addr, int64 val)
{
#pragma omp atomic write
    *addr = val;
}

inline void store(float16* addr, float16 val)
{
    auto uint_addr = copy_cast<uint16_t*>(addr);
    auto uint_val = copy_cast<uint16_t>(val);
#pragma omp atomic write
    *uint_addr = uint_val;
}

template <typename T>
inline void store(std::complex<T>* addr, std::complex<T> val)
{
    auto values = reinterpret_cast<T*>(addr);
    store(values + 0, real(val));
    store(values + 1, imag(val));
}


// Same issue as with the store_helper
inline float load(float* addr)
{
    float val;
#pragma omp atomic read
    val = *addr;
    return val;
}

inline double load(double* addr)
{
    double val;
#pragma omp atomic read
    val = *addr;
    return val;
}

inline int32 load(int32* addr)
{
    float val;
#pragma omp atomic read
    val = *addr;
    return val;
}

inline int64 load(int64* addr)
{
    float val;
#pragma omp atomic read
    val = *addr;
    return val;
}

inline float16 load(float16* addr)
{
    uint16_t uint_val;
    auto uint_addr = copy_cast<uint16_t*>(addr);
#pragma omp atomic read
    uint_val = *uint_addr;
    return copy_cast<float16>(uint_val);
}

template <typename T>
inline std::complex<T> load(std::complex<T>* addr)
{
    auto values = reinterpret_cast<T*>(addr);
    return {load(values + 0), load(values + 1)};
}


}  // namespace omp
}  // namespace kernels
}  // namespace gko


#endif  // GKO_OMP_COMPONENTS_ATOMIC_HPP_
