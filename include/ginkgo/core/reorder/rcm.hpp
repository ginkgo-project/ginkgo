// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
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
    /**
     * Generates the permutation matrix and if required the inverse permutation
     * matrix.
     */
    void generate(std::shared_ptr<const Executor>& exec,
                  std::unique_ptr<SparsityMatrix> adjacency_matrix) const;

    explicit Rcm(std::shared_ptr<const Executor> exec)
        : EnablePolymorphicObject<Rcm, ReorderingBase<IndexType>>(
              std::move(exec))
    {}

    explicit Rcm(const Factory* factory, const ReorderingBaseArgs& args)
        : EnablePolymorphicObject<Rcm, ReorderingBase<IndexType>>(
              factory->get_executor()),
          parameters_{factory->get_parameters()}
    {
        // Always execute the reordering on the cpu.
        const auto is_gpu_executor =
            this->get_executor() != this->get_executor()->get_master();
        auto cpu_exec = is_gpu_executor ? this->get_executor()->get_master()
                                        : this->get_executor();

        auto adjacency_matrix = SparsityMatrix::create(cpu_exec);
        array<IndexType> degrees;

        // The adjacency matrix has to be square.
        GKO_ASSERT_IS_SQUARE_MATRIX(args.system_matrix);
        // This is needed because it does not make sense to call the copy and
        // convert if the existing matrix is empty.
        if (args.system_matrix->get_size()) {
            auto tmp = copy_and_convert_to<SparsityMatrix>(cpu_exec,
                                                           args.system_matrix);
            // This function provided within the Sparsity matrix format removes
            // the diagonal elements and outputs an adjacency matrix.
            adjacency_matrix = tmp->to_adjacency_matrix();
        }

        auto const dim = adjacency_matrix->get_size();
        permutation_ = PermutationMatrix::create(cpu_exec, dim);

        // To make it explicit.
        inv_permutation_ = nullptr;
        if (parameters_.construct_inverse_permutation) {
            inv_permutation_ = PermutationMatrix::create(cpu_exec, dim);
        }

        this->generate(cpu_exec, std::move(adjacency_matrix));

        // Copy back results to gpu if necessary.
        if (is_gpu_executor) {
            const auto gpu_exec = this->get_executor();
            auto gpu_perm = share(PermutationMatrix::create(gpu_exec, dim));
            gpu_perm->copy_from(permutation_);
            permutation_ = gpu_perm;
            if (inv_permutation_) {
                auto gpu_inv_perm =
                    share(PermutationMatrix::create(gpu_exec, dim));
                gpu_inv_perm->copy_from(inv_permutation_);
                inv_permutation_ = gpu_inv_perm;
            }
        }
        auto permutation_array =
            make_array_view(this->get_executor(), permutation_->get_size()[0],
                            permutation_->get_permutation());
        this->set_permutation_array(permutation_array);
    }

private:
    std::shared_ptr<PermutationMatrix> permutation_;
    std::shared_ptr<PermutationMatrix> inv_permutation_;
};


}  // namespace reorder
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_REORDER_RCM_HPP_
