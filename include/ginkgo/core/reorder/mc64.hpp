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

#ifndef GKO_PUBLIC_CORE_REORDER_MC64_HPP_
#define GKO_PUBLIC_CORE_REORDER_MC64_HPP_


#include <memory>


#include <ginkgo/core/base/abstract_factory.hpp>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
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
 * Strategy defining the goal of the MC64 reordering.
 * max_diagonal_product aims at maximizing the product of
 * absolute diagonal entries.
 * max_diag_sum aims at maximizing the sum of absolute values
 * for the diagonal entries.
 */
enum class reordering_strategy { max_diagonal_product, max_diagonal_sum };


/**
 * MC64 is an algorithm for permuting large entries to the diagonal of a
 * sparse matrix. This approach can increase numerical stability of e.g.
 * an LU factorization without pivoting. Under the assumption of working
 * on a nonsingular square matrix, the algorithm computes a minimum weight
 * extreme matching on a weighted edge bipartite graph of the matrix. It is
 * described in detail in "On Algorithms for Permuting Large Entries to the
 * Diagonal of a Sparse Matrix" (Duff, Koster, 2001). There are two strategies
 * for choosing the weights supported:
 *  - Maximizing the product of the absolute values on the diagonal.
 *    For this strategy, the weights are computed as
 *      c(i, j) = log2(a_i) - log2(abs(a(i, j))) if a(i, j) is nonzero and
 * infinity otherwise Here, a_i is the maximum absolute value in row i of the
 * matrix A. In this case, the implementation computes a row permutation P and
 * row and column scaling coefficients L and R such that the matrix P*L*A*R has
 * values with unity absolute value on the diagonal and smaller or equal entries
 * everywhere else.
 *  - Maximizing the sum of the absolute values on the diagonal.
 *    For this strategy, the weights are computed as
 *      c(i, j) = a_i - abs(a(i, j)) if a(i, j) is nonzero and infinity
 * otherwise In this case, no scaling coefficients are computed.
 *
 * @note  This class is derived from polymorphic object but is not a LinOp as it
 * does not make sense for this class to implement the apply methods. The
 * objective of this class is to generate a reordering/permutation vector (in
 * the form of the Permutation matrix), which can be used to apply to reorder a
 * matrix as required.
 *
 * @tparam ValueType  Type of the values of all matrices used in this class
 * @tparam IndexType  Type of the indices of all matrices used in this class
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class Mc64 : public EnablePolymorphicObject<Mc64<ValueType, IndexType>,
                                            ReorderingBase>,
             public EnablePolymorphicAssignment<Mc64<ValueType, IndexType>> {
    friend class EnablePolymorphicObject<Mc64, ReorderingBase>;

public:
    using matrix_type = matrix::Csr<ValueType, IndexType>;
    using PermutationMatrix = matrix::Permutation<IndexType>;
    using DiagonalMatrix = matrix::Diagonal<ValueType>;
    using value_type = ValueType;
    using index_type = IndexType;


    /**
     * Gets the permutation (permutation matrix, output of the algorithm) of the
     * linear operator.
     *
     * @return the permutation (permutation matrix)
     */
    std::shared_ptr<const LinOp> get_permutation() const override
    {
        return permutation_;
    }

    /**
     * Gets the inverse permutation (permutation matrix, output of the
     * algorithm) of the linear operator.
     *
     * @return the inverse permutation (permutation matrix)
     */
    std::shared_ptr<const LinOp> get_inverse_permutation() const override
    {
        return inv_permutation_;
    }

    /**
     * Gets the row scaling coefficients. If the strategy is max_diagonal_sum,
     * these are all 1.
     *
     * @return the row scaling coefficients (diagonal matrix)
     */
    std::shared_ptr<const DiagonalMatrix> get_row_scaling() const
    {
        return row_scaling_;
    }

    /**
     * Gets the column sclaing coefficients. If the strategy is
     * max_diagonal_sum, these are all 1.
     *
     * @return the column scaling coefficients (diagonal matrix)
     */
    std::shared_ptr<const DiagonalMatrix> get_col_scaling() const
    {
        return col_scaling_;
    }

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * This parameter controls the goal of the permutation.
         */
        reordering_strategy GKO_FACTORY_PARAMETER_SCALAR(
            strategy, reordering_strategy::max_diagonal_product);

        /**
         * This parameter controls the tolerance below which a weight is
         * considered to be zero.
         */
        remove_complex<ValueType> GKO_FACTORY_PARAMETER_SCALAR(tolerance,
                                                               1e-14);

        /**
         * This parameter controls the binary logarithm of the heap arity
         * for the addressable priority queue used in generating the
         * minimum weight perfect matching.
         */
        int GKO_FACTORY_PARAMETER_SCALAR(log2_degree, 4);
    };
    GKO_ENABLE_REORDERING_BASE_FACTORY(Mc64, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    /**
     * Generates the permutation matrix and the inverse permutation
     * matrix.
     */
    void generate(std::shared_ptr<const Executor>& exec,
                  std::shared_ptr<LinOp> system_matrix);

    explicit Mc64(std::shared_ptr<const Executor> exec)
        : EnablePolymorphicObject<Mc64, ReorderingBase>(std::move(exec))
    {}

    explicit Mc64(const Factory* factory, const ReorderingBaseArgs& args)
        : EnablePolymorphicObject<Mc64, ReorderingBase>(
              factory->get_executor()),
          parameters_{factory->get_parameters()}
    {
        auto exec = this->get_executor();
        // Always execute the reordering on a reference executor as the
        // algorithm is only implemented sequentially.
        const auto is_gpu_executor = exec != exec->get_master();
        const auto host_is_ref =
            dynamic_cast<const ReferenceExecutor*>(exec->get_master().get());
        auto ref =
            host_is_ref ? exec->get_master() : ReferenceExecutor::create();

        auto system_matrix = share(matrix_type::create(ref));

        // The system matrix has to be square.
        GKO_ASSERT_IS_SQUARE_MATRIX(args.system_matrix);
        if (args.system_matrix->get_size()) {
            system_matrix =
                copy_and_convert_to<matrix_type>(ref, args.system_matrix);
        }

        auto const dim = system_matrix->get_size();

        this->generate(ref, system_matrix);

        // Copy back results to original executor if necessary.
        if (ref != exec) {
            auto perm = share(PermutationMatrix::create(exec, dim));
            perm->copy_from(permutation_.get());
            permutation_ = perm;
            auto inv_perm = share(PermutationMatrix::create(exec, dim));
            inv_perm->copy_from(inv_permutation_.get());
            inv_permutation_ = inv_perm;
            auto row_scaling = share(DiagonalMatrix::create(exec, dim[0]));
            row_scaling->copy_from(row_scaling_.get());
            row_scaling_ = row_scaling;
            auto col_scaling = share(DiagonalMatrix::create(exec, dim[0]));
            col_scaling->copy_from(col_scaling_.get());
            col_scaling_ = col_scaling;
        }
    }

private:
    std::shared_ptr<PermutationMatrix> permutation_;
    std::shared_ptr<PermutationMatrix> inv_permutation_;
    std::shared_ptr<DiagonalMatrix> row_scaling_;
    std::shared_ptr<DiagonalMatrix> col_scaling_;
};


}  // namespace reorder
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_REORDER_MC64_HPP_
