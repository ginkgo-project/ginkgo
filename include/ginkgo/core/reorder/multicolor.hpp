// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_REORDER_MULTICOLOR_HPP_
#define GKO_PUBLIC_CORE_REORDER_MULTICOLOR_HPP_


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
 * A multicolor reordering, also known as independent set reordering.
 *
 * This is intended for reordering an arbitrary sparse matrix with symmetric
 * structure so that preconditioners/smoothers such as Gauss-Seidel and ILU
 * can be applied in parallel.
 *
 * The reference implementation is based on a simple greedy approach, while
 * the parallel implementations use the Jones-Plassman-Luby (JPL) algorithm
 * described in: "A Parallel Graph Coloring Heuristic" by Mark T. Jones and
 * Paul E. Plassmann, SIAM Journal on Scientific Computing 1993 14:3, 654-669,
 * doi:10.1137/0914041.
 *
 * @tparam ValueType  Type of the values of all matrices used in this class
 * @tparam IndexType  Type of the indices of all matrices used in this class
 *
 * @ingroup reorder
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class Multicolor
    : public EnablePolymorphicObject<Multicolor<ValueType, IndexType>,
                                     ReorderingBase<IndexType>>,
      public EnablePolymorphicAssignment<Multicolor<ValueType, IndexType>> {
    friend class EnablePolymorphicObject<Multicolor, ReorderingBase<IndexType>>;
    GKO_ASSERT_SUPPORTED_VALUE_AND_INDEX_TYPE;

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

    /**
     * Get a copy of the row indices at which each independent set (color)
     * begins.
     *
     * If the number of colors is n_c, the size if n_c + 1.
     * The first entry is always 0, since the first color always starts at 0.
     * The last entry stores the total number of rows.
     * The underlying storage is always on the master (host) executor.
     */
    gko::array<index_type> get_color_pointers() const { return color_ptrs_; }

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory){};
    GKO_ENABLE_REORDERING_BASE_FACTORY(Multicolor, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    explicit Multicolor(std::shared_ptr<const Executor> exec);

    explicit Multicolor(const Factory* factory, const ReorderingBaseArgs& args);

private:
    std::shared_ptr<PermutationMatrix> permutation_;
    std::shared_ptr<PermutationMatrix> inv_permutation_;
    gko::array<index_type> color_ptrs_;
};


}  // namespace reorder
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_REORDER_RCM_HPP_
