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

#ifndef GKO_PUBLIC_CORE_BASE_BATCH_LIN_OP_HELPERS_HPP_
#define GKO_PUBLIC_CORE_BASE_BATCH_LIN_OP_HELPERS_HPP_


#include <memory>
#include <type_traits>
#include <utility>


#include <ginkgo/core/base/abstract_factory.hpp>
#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/matrix_assembly_data.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/log/logger.hpp>


namespace gko {


/**
 * A BatchLinOp implementing this interface can read its data from a matrix_data
 * structure.
 *
 * @ingroup BatchLinOp
 */
template <typename ValueType, typename IndexType>
class BatchReadableFromMatrixData {
public:
    using value_type = ValueType;
    using index_type = IndexType;

    virtual ~BatchReadableFromMatrixData() = default;

    /**
     * Reads a batch matrix from a std::vector of matrix_data objects.
     *
     * @param data  the std::vector of matrix_data objects
     */
    virtual void read(
        const std::vector<matrix_data<ValueType, IndexType>>& data) = 0;
};


/**
 * A BatchLinOp implementing this interface can write its data to a std::vector
 * of matrix_data objects.
 *
 * @ingroup BatchLinOp
 */
template <typename ValueType, typename IndexType>
class BatchWritableToMatrixData {
public:
    using value_type = ValueType;
    using index_type = IndexType;

    virtual ~BatchWritableToMatrixData() = default;

    /**
     * Writes a matrix to a matrix_data structure.
     *
     * @param data  the matrix_data structure
     */
    virtual void write(
        std::vector<matrix_data<ValueType, IndexType>>& data) const = 0;
};


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_BATCH_LIN_OP_HELPERS_HPP_
