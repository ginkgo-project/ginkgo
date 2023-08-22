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

#ifndef GKO_DPCPP_COMPONENTS_UNINITIALIZED_ARRAY_HPP_
#define GKO_DPCPP_COMPONENTS_UNINITIALIZED_ARRAY_HPP_


#include <ginkgo/core/base/types.hpp>


#include "dpcpp/base/dpct.hpp"


namespace gko {
namespace kernels {
namespace sycl {


// TODO: porting - consider directly use the array as shared memory


/**
 * Stores an array with uninitialized contents.
 *
 * This class is needed for datatypes that do have a non-empty constructor when
 * using them as shared memory, for example `thrust::complex<float>`.
 *
 * @tparam ValueType the type of values
 * @tparam size the size of the array
 */
template <typename ValueType, size_type size>
class uninitialized_array {
public:
    /**
     * Operator for casting an uninitialized_array into its constexpr value
     * pointer.
     *
     * @return the constexpr pointer to the first entry of the array.
     */
    constexpr __dpct_inline__ operator const ValueType*() const noexcept
    {
        return &(*this)[0];
    }

    /**
     * Operator for casting an uninitialized_array into its non-const value
     * pointer.
     *
     * @return the non-const pointer to the first entry of the array.
     */
    __dpct_inline__ operator ValueType*() noexcept { return &(*this)[0]; }

    /**
     * constexpr array access operator.
     *
     * @param pos The array index. Using a value outside [0, size) is undefined
     * behavior.
     *
     * @return a reference to the array entry at the given index.
     */
    constexpr __dpct_inline__ const ValueType& operator[](
        size_type pos) const noexcept
    {
        return data_[pos];
    }

    /**
     * Non-const array access operator.
     *
     * @param pos The array index. Using a value outside [0, size) is undefined
     * behavior.
     *
     * @return a reference to the array entry at the given index.
     */
    __dpct_inline__ ValueType& operator[](size_type pos) noexcept
    {
        return data_[pos];
    }

private:
    // if sycl uses char to represent data in char, compiling gives error.
    // Thanksfully, sycl support complex data allocation directly.
    ValueType data_[size];
};


}  // namespace sycl
}  // namespace kernels
}  // namespace gko


#endif  // GKO_DPCPP_COMPONENTS_UNINITIALIZED_ARRAY_HPP_
