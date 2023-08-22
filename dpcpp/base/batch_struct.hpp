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

#ifndef GKO_DPCPP_BASE_BATCH_STRUCT_HPP_
#define GKO_DPCPP_BASE_BATCH_STRUCT_HPP_


#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/math.hpp>


#include "core/base/batch_struct.hpp"
#include "dpcpp/base/config.hpp"


namespace gko {
namespace kernels {
namespace sycl {


/** @file batch_struct.hpp
 *
 * Helper functions to generate a batch struct from a batch LinOp,
 * while also shallow-casting to the required DPCPP scalar type.
 *
 * A specialization is needed for every format of every kind of linear algebra
 * object. These are intended to be called on the host.
 */


/**
 * Generates an immutable uniform batch struct from a batch of multi-vectors.
 */
template <typename ValueType>
inline batch::multi_vector::uniform_batch<const ValueType> get_batch_struct(
    const batch::MultiVector<ValueType>* const op)
{
    return {op->get_const_values(), op->get_num_batch_items(),
            static_cast<int>(op->get_common_size()[1]),
            static_cast<int>(op->get_common_size()[0]),
            static_cast<int>(op->get_common_size()[1])};
}


/**
 * Generates a uniform batch struct from a batch of multi-vectors.
 */
template <typename ValueType>
inline batch::multi_vector::uniform_batch<ValueType> get_batch_struct(
    batch::MultiVector<ValueType>* const op)
{
    return {op->get_values(), op->get_num_batch_items(),
            static_cast<int>(op->get_common_size()[1]),
            static_cast<int>(op->get_common_size()[0]),
            static_cast<int>(op->get_common_size()[1])};
}


}  // namespace sycl
}  // namespace kernels
}  // namespace gko


#endif  // GKO_DPCPP_BASE_BATCH_STRUCT_HPP_
