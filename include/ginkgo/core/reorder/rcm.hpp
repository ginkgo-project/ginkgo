/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#ifndef GKO_PUBLIC_CORE_REORDER_RCM_HPP_
#define GKO_PUBLIC_CORE_REORDER_RCM_HPP_


#include <memory>


#include <ginkgo/core/base/abstract_factory.hpp>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>
#include <ginkgo/core/reorder/reordering_base.hpp>


namespace gko {
/**
 * @brief The Reorder namespace.
 *
 * @ingroup reorder
 */
namespace reorder {


enum class starting_strategy { minimum_degree, pseudo_peripheral };


/**
 * Rcm is a reordering algorithm minimizing the bandwidth of a matrix. Such a
 * reordering typically also significantly reduces fill-in, though usually not
 * as effective as more complex algorithms, specifically AMD and nested
 * dissection schemes. The advantage of this algorithm is its low runtime.
 *
 * @note  This class is derived from polymorphic object but is not a LinOp as it
 * does not make sense for this class to implement the apply methods. The
 * objective of this class is to generate a reordering/permutation vector (in
 * the form of the Permutation matrix), which can be used to apply to reorder a
 * matrix as required.
 *
 * There are two "starting strategies" currently available: minimum degree and
 * pseudo-peripheral. These strategies control how a starting vertex for a
 * connected component is choosen, which is then renumbered as first vertex in
 * the component, starting the algorithm from there.
 * In general, the bandwidths obtained by choosing a pseudo-peripheral vertex
 * are slightly smaller than those obtained from choosing a vertex of minimum
 * degree. On the other hand, this strategy is much more expensive, relatively.
 * The algorithm for finding a pseudo-peripheral vertex as
 * described in "Computer Solution of Sparse Linear Systems" (George, Liu, Ng,
 * Oak Ridge National Laboratory, 1994) is implemented here.
 *
 * @tparam ValueType  Type of the values of all matrices used in this class
 * @tparam IndexType  Type of the indices of all matrices used in this class
 *
 * @ingroup reorder
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class Rcm
    : public EnablePolymorphicObject<Rcm<ValueType, IndexType>, ReorderingBase>,
      public EnablePolymorphicAssignment<Rcm<ValueType, IndexType>> {
    friend class EnablePolymorphicObject<Rcm, ReorderingBase>;

public:
    using SparsityMatrix = matrix::SparsityCsr<ValueType, IndexType>;
    using PermutationMatrix = matrix::Permutation<IndexType>;
    using value_type = ValueType;
    using index_type = IndexType;

    /**
     * Gets the permutation (permutation matrix, output of the algorithm) of the
     * linear operator.
     *
     * @return the permutation (permutation matrix)
     */
    std::shared_ptr<const PermutationMatrix> get_permutation() const;

    /**
     * Gets the inverse permutation (permutation matrix, output of the
     * algorithm) of the linear operator.
     *
     * @return the inverse permutation (permutation matrix)
     */
    std::shared_ptr<const PermutationMatrix> get_inverse_permutation() const;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * If this parameter is set then an inverse permutation matrix is also
         * constructed along with the normal permutation matrix.
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(construct_inverse_permutation, false);

        /**
         * This parameter controls the strategy used to determine a starting
         * vertex.
         */
        starting_strategy GKO_FACTORY_PARAMETER_SCALAR(
            strategy, starting_strategy::pseudo_peripheral);
    };
    GKO_ENABLE_REORDERING_BASE_FACTORY(Rcm, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    /**
     * Generates the permutation matrix and if required the inverse permutation
     * matrix.
     */
    void generate(std::shared_ptr<const Executor>& exec,
                  std::unique_ptr<SparsityMatrix> adjacency_matrix) const;

    explicit Rcm(std::shared_ptr<const Executor> exec);

    explicit Rcm(const Factory* factory, const ReorderingBaseArgs& args);

private:
    std::shared_ptr<PermutationMatrix> permutation_;
    std::shared_ptr<PermutationMatrix> inv_permutation_;
};


}  // namespace reorder
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_REORDER_RCM_HPP_
