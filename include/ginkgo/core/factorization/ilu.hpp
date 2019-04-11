/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#ifndef GKO_CORE_FACTORIZATION_PAR_ILU_FACTORY_HPP_
#define GKO_CORE_FACTORIZATION_PAR_ILU_FACTORY_HPP_


#include <memory>


#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>


namespace gko {
/**
 * @brief The Factorization namespace.
 *
 * @ingroup factor
 */
namespace factorization {


// TODO rename to ParIluFactory
// TODO rename the file


/**
 * A Factory that generates the incomplete LU-factorization
 *
 * @tparam ValueType  Type of the values in the matrices
 * @tparam IndexType  Index type in the matrices
 *
 * @ingroup factor
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class ParIluFactors
    :  // public EnableDefaultLinOpFactory<ParIluFactory, ComposedLinOp, >
{
public:
    friend class EnablePolymorphicObject<ParIluFactory<ValueType, IndexType>,
                                         LinOpFactory>;
    using value_type = ValueType;
    using index_type = IndexType;
    using l_matrix_type = matrix::Csr<ValueType, IndexType>;
    using u_matrix_type = matrix::Csr<ValueType, IndexType>;

    using EnablePolymorphicObject<ParIluFactory<ValueType, IndexType>,
                                  LinOpFactory>::EnablePolymorphicObject;
    /**
     * Creates a factory for an incomplete LU factorization.
     * @param exec  the executor for the LU decomposition
     * @param TODO  parameters created by the `GKO_CREATE_FACTORY_PARAMETERS`
     */
    explicit ParIluFactory(
        std::shared_ptr<const Executor> exec
        /* TODO list other parameters needed for the factory */)
        : EnablePolymorphicObject(std::move(exec))
    {
        // TODO use the parameters to properly instantiate this object
    }

protected:
    /**
     * Generates the incomplete LU factors and returns a composition of the
     * lower and the upper factor.
     *
     * @param system_matrix  the source matrix used to generate the
     *                       factors
     */
    std::unique_ptr<LinOp> generate_impl(
        std::shared_ptr<const LinOp> system) const override;
};


}  // namespace factorization
}  // namespace gko


#endif  // GKO_CORE_FACTORIZATION_PAR_ILU_FACTORY_HPP_
