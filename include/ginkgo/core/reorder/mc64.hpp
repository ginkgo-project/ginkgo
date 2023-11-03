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

#ifndef GKO_PUBLIC_CORE_REORDER_MC64_HPP_
#define GKO_PUBLIC_CORE_REORDER_MC64_HPP_


#include <memory>


#include <ginkgo/core/base/abstract_factory.hpp>
#include <ginkgo/core/base/composition.hpp>
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
namespace experimental {
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
enum class mc64_strategy { max_diagonal_product, max_diagonal_sum };


/**
 * MC64 is an algorithm for permuting large entries to the diagonal of a
 * sparse matrix. This approach can increase numerical stability of e.g.
 * an LU factorization without pivoting. Under the assumption of working
 * on a nonsingular square matrix, the algorithm computes a minimum weight
 * perfect matching on a weighted edge bipartite graph of the matrix. It is
 * described in detail in "On Algorithms for Permuting Large Entries to the
 * Diagonal of a Sparse Matrix" (Duff, Koster, 2001,
 * DOI: 10.1137/S0895479899358443). There are two strategies for choosing the
 * weights supported:
 *  - Maximizing the product of the absolute values on the diagonal.
 *    For this strategy, the weights are computed as
 *      $c(i, j) = \log_2(a_i) - \log_2(|a(i, j)|)$ if $a(i, j) \neq 0 $ and
 * $c(i, j) = \infty$ otherwise. Here, a_i is the maximum absolute value in row
 * i of the matrix A. In this case, the implementation computes a row
 * permutation P and row and column scaling coefficients L and R such that the
 * matrix P*L*A*R has values with unity absolute value on the diagonal and
 * smaller or equal entries everywhere else.
 *  - Maximizing the sum of the absolute values on the diagonal.
 *    For this strategy, the weights are computed as
 *      $c(i, j) = a_i - |a(i, j)|$ if $a(i, j) \neq 0$ and $c(i, j) =
 * \infty$ otherwise. In this case, no scaling coefficients are computed.
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
class Mc64
    : public EnablePolymorphicObject<Mc64<ValueType, IndexType>, LinOpFactory>,
      public EnablePolymorphicAssignment<Mc64<ValueType, IndexType>> {
public:
    struct parameters_type;
    friend class EnablePolymorphicObject<Mc64<ValueType, IndexType>,
                                         LinOpFactory>;
    friend class enable_parameters_type<parameters_type,
                                        Mc64<ValueType, IndexType>>;

    using value_type = ValueType;
    using index_type = IndexType;
    using result_type = Composition<value_type>;
    using matrix_type = matrix::Csr<value_type, index_type>;

    struct parameters_type
        : public enable_parameters_type<parameters_type, Mc64> {
        /**
         * This parameter controls the goal of the permutation.
         */
        mc64_strategy GKO_FACTORY_PARAMETER_SCALAR(
            strategy, mc64_strategy::max_diagonal_product);

        /**
         * This parameter controls the tolerance below which a weight is
         * considered to be zero.
         */
        remove_complex<ValueType> GKO_FACTORY_PARAMETER_SCALAR(tolerance,
                                                               1e-14);
    };

    /**
     * Returns the parameters used to construct the factory.
     *
     * @return the parameters used to construct the factory.
     */
    const parameters_type& get_parameters() const { return parameters_; }

    /**
     * @copydoc LinOpFactory::generate
     * @note This function overrides the default LinOpFactory::generate to
     *       return a Permutation instead of a generic LinOp, which would
     *       need to be cast to ScaledPermutation again to access its indices.
     *       It is only necessary because smart pointers aren't covariant.
     */
    std::unique_ptr<result_type> generate(
        std::shared_ptr<const LinOp> system_matrix) const;

    /** Creates a new parameter_type to set up the factory. */
    static parameters_type build() { return {}; }

protected:
    explicit Mc64(std::shared_ptr<const Executor> exec,
                  const parameters_type& params = {});

    std::unique_ptr<LinOp> generate_impl(
        std::shared_ptr<const LinOp> system_matrix) const override;

    parameters_type parameters_;
};


}  // namespace reorder
}  // namespace experimental
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_REORDER_MC64_HPP_
