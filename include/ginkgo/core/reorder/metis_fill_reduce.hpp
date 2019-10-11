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

#ifndef GKO_CORE_REORDER_METIS_FILL_REDUCE_HPP_
#define GKO_CORE_REORDER_METIS_FILL_REDUCE_HPP_


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


/**
 * MetisFillReduce is the reordering algorithm which uses METIS to compute a
 * fill reducing re-ordering of the given sparse matrix. The METIS
 * `Metis_NodeND` which is implements the multilevel nested dissection algorithm
 * as in the METIS documentation.
 *
 * @note  This class is derives from polymorphic object but is not a LinOp as it
 * does not make sense for this class to implement the apply methods. The
 * objective of this class is to generate a reordering/permutation vector (in
 * the form of the Permutation matrix), which can be used to apply to reorder a
 * matrix as required.
 *
 * @tparam ValueType  Type of the values of all matrices used in this class
 * @tparam IndexType  Type of the indices of all matrices used in this class
 *
 * @ingroup reorder
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class MetisFillReduce
    : public EnablePolymorphicObject<MetisFillReduce<ValueType, IndexType>,
                                     ReorderingBase> {
    friend class EnablePolymorphicObject<MetisFillReduce, ReorderingBase>;

public:
    using SparsityMatrix = matrix::SparsityCsr<ValueType, IndexType>;
    using PermutationMatrix = matrix::Permutation<IndexType>;

    /**
     * Gets the system operator (input matrix) of the linear operator.
     *
     * @return the system operator (matrix)
     */
    std::shared_ptr<const SparsityMatrix> get_adjacency_matrix() const
    {
        return adjacency_matrix_;
    }

    /**
     * Gets the permutation (permutation matrix, output of the algorithm) of the
     * linear operator.
     *
     * @return the permutation (permutation matrix)
     */
    std::shared_ptr<const PermutationMatrix> get_permutation() const
    {
        return permutation_;
    }

    /**
     * Gets the inverse permutation (permutation matrix, output of the
     * algorithm) of the linear operator.
     *
     * @return the inverse permutation (permutation matrix)
     */
    std::shared_ptr<const PermutationMatrix> get_inverse_permutation() const
    {
        return inv_permutation_;
    }

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * If this parameter is set then an inverse permutation matrix is also
         * constructed along with the normal permutation matrix.
         */
        bool GKO_FACTORY_PARAMETER(construct_inverse_permutation, false);

        /**
         * The weights associated with the vertices of the adjacency. Within
         * METIS, this is either a nullptr or it **has** to be an array of
         * non-zeros. Any array with equal non-zero weights is equivalent to
         * when a nullptr is passed.
         */
        Array<IndexType> GKO_FACTORY_PARAMETER(vertex_weights, nullptr);
    };
    GKO_ENABLE_REORDERING_BASE_FACTORY(MetisFillReduce, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    /**
     * Generates the permutation matrix and if required the inverse permutation
     * matrix.
     */
    void generate() const;

    explicit MetisFillReduce(std::shared_ptr<const Executor> exec)
        : EnablePolymorphicObject<MetisFillReduce, ReorderingBase>(
              std::move(exec))
    {}

    explicit MetisFillReduce(const Factory *factory,
                             const ReorderingBaseArgs &args)
        : EnablePolymorphicObject<MetisFillReduce, ReorderingBase>(
              factory->get_executor()),
          parameters_{factory->get_parameters()}
    {
        const auto exec = this->get_executor();
        // The adjacency matrix has to be square.
        GKO_ASSERT_IS_SQUARE_MATRIX(args.adjacency_matrix);
        // This is needed because it does not make sense to call the copy and
        // convert if the existing matrix is empty.
        if (!args.adjacency_matrix->get_size()) {
            adjacency_matrix_ = SparsityMatrix::create(exec);
        } else {
            auto tmp = copy_and_convert_to<SparsityMatrix>(
                exec, args.adjacency_matrix);
            // This function provided within the Sparsity matrix format removes
            // the diagonal elements and outputs an adjacency matrix.
            adjacency_matrix_ = tmp->to_adjacency_matrix();
        }
        if (vertex_weights_ != nullptr) {
            vertex_weights_ =
                std::make_shared<Array<IndexType>>(parameters_.vertex_weights);
        } else {
            vertex_weights_ =
                std::shared_ptr<Array<IndexType>>(new Array<IndexType>(exec));
        }
        permutation_ =
            PermutationMatrix::create(exec, adjacency_matrix_->get_size());
        inv_permutation_ =
            PermutationMatrix::create(exec, adjacency_matrix_->get_size());

        this->generate();
    }

private:
    std::shared_ptr<SparsityMatrix> adjacency_matrix_{};
    std::shared_ptr<Array<IndexType>> vertex_weights_{};
    std::shared_ptr<PermutationMatrix> permutation_{};
    std::shared_ptr<PermutationMatrix> inv_permutation_{};
};


}  // namespace reorder
}  // namespace gko


#endif  // GKO_CORE_REORDER_METIS_FILL_REDUCE_HPP_
