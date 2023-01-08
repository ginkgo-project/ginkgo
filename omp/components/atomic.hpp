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
