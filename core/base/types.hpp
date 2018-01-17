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


#include <complex>
#include <cstddef>
#include <cstdint>
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
 * 32-bit signed integral type.
 */
using int32 = std::int32_t;


/**
 * 64-bit signed integral type.
 */
using int64 = std::int64_t;


/**
 * The most precise floating-point type.
 */
using full_precision = double;


/**
 * Precision used if no precision is explicitly specified.
 */
using default_precision = double;


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
    _enable_macro(CpuExecutor, cpu);                \
    _enable_macro(GpuExecutor, gpu)


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

}  // namespace gko


#endif  // GKO_CORE_TYPES_HPP_
