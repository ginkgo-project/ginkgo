// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_OMP_COMPONENTS_ATOMIC_HPP_
#define GKO_OMP_COMPONENTS_ATOMIC_HPP_


#include <type_traits>

#include <ginkgo/core/base/half.hpp>
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
void atomic_add(half& out, half val)
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
        sizeof(half) == sizeof(uint16_t) &&
            std::alignment_of_v<uint16_t> == std::alignment_of_v<half>,
        "half does not fulfill the requirement of reinterpret_cast to half or "
        "vice versa.");
    // It is undefined behavior with reinterpret_cast, but we do not have any
    // workaround when the #omp atomic does not support custom precision
    uint16_t* address_as_converter = reinterpret_cast<uint16_t*>(&out);
    uint16_t old = *address_as_converter;
    uint16_t assumed;
    do {
        assumed = old;
        auto answer = copy_cast<uint16_t>(copy_cast<half>(assumed) + val);
#pragma omp atomic capture
        {
            old = *address_as_converter;
            *address_as_converter = (old == assumed) ? answer : old;
        }
    } while (assumed != old);
#endif
}


}  // namespace omp
}  // namespace kernels
}  // namespace gko


#endif  // GKO_OMP_COMPONENTS_ATOMIC_HPP_
