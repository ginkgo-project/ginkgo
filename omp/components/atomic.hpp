/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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
