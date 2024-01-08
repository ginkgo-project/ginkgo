// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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
 * Rcm (Reverse Cuthill-McKee) is a reordering algorithm minimizing the
 * bandwidth of a matrix. Such a reordering typically also significantly reduces
 * fill-in, though usually not as effective as more complex algorithms,
 * specifically AMD and nested dissection schemes. The advantage of this
 * algorithm is its low runtime.
 * It requires the input matrix to be structurally symmetric.
 *
 * @note  This class is derived from polymorphic object but is not a LinOp as it
 * does not make sense for this class to implement the apply methods. The
 * objective of this class is to generate a reordering/permutation vector (in
 * the form of the Permutation matrix), which can be used to apply to reorder a
 * matrix as required.
 * @deprecated  This class is deprecated and should be replaced by
 * gko::experimental::reorder::Rcm, which integrates more cleanly with the other
 * reordering-related functionality of Ginkgo.
 *
 * There are two "starting strategies" currently available: minimum degree and
 * pseudo-peripheral. These strategies control how a starting vertex for a
 * connected component is chosen, which is then renumbered as first vertex in
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
class Rcm : public EnablePolymorphicObject<Rcm<ValueType, IndexType>,
                                           ReorderingBase<IndexType>>,
            public EnablePolymorphicAssignment<Rcm<ValueType, IndexType>> {
    friend class EnablePolymorphicObject<Rcm, ReorderingBase<IndexType>>;

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

    /*const array<index_type>& get_permutation_array() const override
    {
        return permutation_array_;
    }*/

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
    explicit Rcm(std::shared_ptr<const Executor> exec);

    explicit Rcm(const Factory* factory, const ReorderingBaseArgs& args);

private:
    std::shared_ptr<PermutationMatrix> permutation_;
    std::shared_ptr<PermutationMatrix> inv_permutation_;
};


}  // namespace reorder


namespace experimental {
namespace reorder {


using rcm_starting_strategy = gko::reorder::starting_strategy;


/**
 * Rcm (Reverse Cuthill-McKee) is a reordering algorithm minimizing the
 * bandwidth of a matrix. Such a reordering typically also significantly reduces
 * fill-in, though usually not as effective as more complex algorithms,
 * specifically AMD and nested dissection schemes. The advantage of this
 * algorithm is its low runtime.
 *
 * The class is a LinOpFactory generating a Permutation matrix out of a Csr
 * system matrix, to be used with `Csr::permute(...)`.
 *
 * There are two "starting strategies" currently available: minimum degree and
 * pseudo-peripheral. These strategies control how a starting vertex for a
 * connected component is chosen, which is then renumbered as first vertex in
 * the component, starting the algorithm from there.
 * In general, the bandwidths obtained by choosing a pseudo-peripheral vertex
 * are slightly smaller than those obtained from choosing a vertex of minimum
 * degree. On the other hand, this strategy is much more expensive, relatively.
 * The algorithm for finding a pseudo-peripheral vertex as
 * described in "Computer Solution of Sparse Linear Systems" (George, Liu, Ng,
 * Oak Ridge National Laboratory, 1994) is implemented here.
 *
 * @tparam IndexType  Type of the indices of all matrices used in this class
 *
 * @ingroup reorder
 */
template <typename IndexType = int32>
class Rcm : public EnablePolymorphicObject<Rcm<IndexType>, LinOpFactory>,
            public EnablePolymorphicAssignment<Rcm<IndexType>> {
public:
    struct parameters_type;
    friend class EnablePolymorphicObject<Rcm<IndexType>, LinOpFactory>;
    friend class enable_parameters_type<parameters_type, Rcm<IndexType>>;

    using index_type = IndexType;
    using permutation_type = matrix::Permutation<index_type>;

    struct parameters_type
        : public enable_parameters_type<parameters_type, Rcm<IndexType>> {
        /**
         * If set to false, computes the RCM reordering on A + A^T, otherwise
         * assumes that A is symmetric and uses it directly.
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(skip_symmetrize, false);

        /**
         * This parameter controls the strategy used to determine a starting
         * vertex.
         */
        rcm_starting_strategy GKO_FACTORY_PARAMETER_SCALAR(
            strategy, rcm_starting_strategy::pseudo_peripheral);
    };

    /**
     * Returns the parameters used to construct the factory.
     *
     * @return the parameters used to construct the factory.
     */
    const parameters_type& get_parameters() { return parameters_; }

    /**
     * @copydoc LinOpFactory::generate
     * @note This function overrides the default LinOpFactory::generate to
     *       return a Permutation instead of a generic LinOp, which would
     *       need to be cast to Permutation again to access its indices.
     *       It is only necessary because smart pointers aren't covariant.
     */
    std::unique_ptr<permutation_type> generate(
        std::shared_ptr<const LinOp> system_matrix) const;

    /** Creates a new parameter_type to set up the factory. */
    static parameters_type build() { return {}; }

protected:
    explicit Rcm(std::shared_ptr<const Executor> exec,
                 const parameters_type& params = {});

    std::unique_ptr<LinOp> generate_impl(
        std::shared_ptr<const LinOp> system_matrix) const override;

    parameters_type parameters_;
};


}  // namespace reorder
}  // namespace experimental
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_REORDER_RCM_HPP_
