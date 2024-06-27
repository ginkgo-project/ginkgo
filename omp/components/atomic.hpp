// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
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
#pragma omp atomic update relaxed
    out += val;
}

template <typename ValueType,
          std::enable_if_t<is_complex<ValueType>()>* = nullptr>
void atomic_add(ValueType& out, ValueType val)
{
    // The C++ standard explicitly allows casting complex<double>* to double*
    // [complex.numbers.general]
    auto values = reinterpret_cast<gko::remove_complex<ValueType>*>(&out);
#pragma omp atomic update relaxed
    values[0] += real(val);
#pragma omp atomic update relaxed
    values[1] += imag(val);
}


template <typename ValueType,
          std::enable_if_t<!is_complex<ValueType>()>* = nullptr>
ValueType load_relaxed(const ValueType& val)
{
    ValueType out{};
#pragma omp atomic read relaxed
    out = val;
    return out;
}

template <typename ValueType,
          std::enable_if_t<is_complex<ValueType>()>* = nullptr>
ValueType load_relaxed(const ValueType& val)
{
    remove_complex<ValueType> r{};
    remove_complex<ValueType> i{};
    // The C++ standard explicitly allows casting complex<double>* to double*
    // [complex.numbers.general]
    auto values = reinterpret_cast<const remove_complex<ValueType>*>(&val);
#pragma omp atomic read relaxed
    r = values[0];
#pragma omp atomic read relaxed
    i = values[1];
    return ValueType{r, i};
}


template <typename ValueType,
          std::enable_if_t<!is_complex<ValueType>()>* = nullptr>
void store_relaxed(ValueType& out, ValueType val)
{
#pragma omp atomic write relaxed
    out = val;
}

template <typename ValueType,
          std::enable_if_t<is_complex<ValueType>()>* = nullptr>
void store_relaxed(ValueType& out, ValueType val)
{
    // The C++ standard explicitly allows casting complex<double>* to double*
    // [complex.numbers.general]
    auto values = reinterpret_cast<remove_complex<ValueType>*>(&out);
#pragma omp atomic write relaxed
    values[0] = real(val);
#pragma omp atomic write relaxed
    values[1] = imag(val);
}


template <typename ValueType>
ValueType load_acquire(const ValueType& val)
{
    ValueType out{};
#pragma omp atomic read acquire
    out = val;
    return out;
}


template <typename ValueType, typename ValueType2>
void store_release(ValueType& out, ValueType2 val)
{
#pragma omp atomic write release
    out = ValueType(val);
}


}  // namespace omp
}  // namespace kernels
}  // namespace gko


#endif  // GKO_OMP_COMPONENTS_ATOMIC_HPP_
