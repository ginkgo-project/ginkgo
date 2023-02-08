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
inline ResultType reinterpret(ValueType val)
{
    static_assert(sizeof(ValueType) == sizeof(ResultType),
                  "The type to reinterpret to must be of the same size as the "
                  "original type.");
    return reinterpret_cast<ResultType&>(val);
}


template <>
void atomic_add(half& out, half val)
{
    // UB?
    uint16_t* address_as_converter = reinterpret_cast<uint16_t*>(&out);
    uint16_t old = *address_as_converter;
    uint16_t assumed;
    do {
        assumed = old;
        auto answer = reinterpret<uint16_t>(reinterpret<half>(assumed) + val);
#pragma omp atomic capture
        {
            old = *address_as_converter;
            *address_as_converter = (old == assumed) ? answer : old;
        }
    } while (assumed != old);
}


}  // namespace omp
}  // namespace kernels
}  // namespace gko


#endif  // GKO_OMP_COMPONENTS_ATOMIC_HPP_
