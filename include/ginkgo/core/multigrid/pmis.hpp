// SPDX-FileCopyrightText: 2025 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_MULTIGRID_PMIS_HPP_
#define GKO_PUBLIC_CORE_MULTIGRID_PMIS_HPP_


#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/config/type_descriptor.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/multigrid/multigrid_level.hpp>


namespace gko {
namespace multigrid {


/**
 * Selects how a Pmis level builds its prolongation once the C/F splitting is
 * known. The coarsening itself is identical in every case.
 *
 * - `direct`: each F-point interpolates from its strongly connected C-points
 *   only, using row-local information. Available for distributed matrices.
 *
 * - `classical`: the Ruge-Stueben formula, which additionally distributes
 *   each strongly connected F-neighbour's coupling over that neighbour's own
 *   C-points. A row whose denominator would be numerically unsafe falls back
 *   to the direct formula, and a near-singular row interpolates nothing. It
 *   reads the rows of neighbouring points, so it is unavailable for
 *   distributed matrices.
 *
 * - `ext_i`: extended+i interpolation. Like `classical` it distributes each
 *   strong F-neighbour's coupling, but from a distance-two set: the row's
 *   strong C-neighbours plus those of each of its strong F-neighbours. A fine
 *   row can therefore draw from a C-point it has no matrix entry for, which
 *   frees it from the common-C-point assumption `classical` needs and PMIS
 *   coarsening does not guarantee. The prolongation reproduces the constant
 *   vector only for an operator with zero row sums. On a distributed matrix
 *   the interpolatory set is restricted to the nodes this rank can address,
 *   since a second-hop C-point may live on a rank it never sees.
 */
enum class interpolation_type { direct, classical, ext_i };


/**
 * Parallel modified independent set (Pmis) is the classical coarsening method
 * introduced in the paper H. De Sterck et al., "Reducing complexity in parallel
 * algebraic multigrid preconditioners".
 *
 * Pmis creates the coarse- and fine-points group according to the matrix value
 * not the structure.
 *
 * @note The restriction is \f$P^T\f$ and the coarse operator the Galerkin
 *       product \f$P^T A P\f$, which assumes a symmetric positive definite
 *       system. Non-symmetric systems still coarsen, but without a
 *       convergence guarantee.
 *
 * @note `generate()` may return a level that does not reduce the dimension.
 *       A level whose every row is fully summed has no coarse point to
 *       select and legitimately is the coarsest one, so `Pmis` reports the
 *       identity as prolongation and restriction rather than raising.
 *       `Multigrid` stops on that condition and discards the level, so a
 *       hierarchy may end earlier than `max_levels` and `min_coarse_rows`
 *       suggest.
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  precision of matrix indices
 *
 * @ingroup MultigridLevel
 * @ingroup Multigrid
 * @ingroup LinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class Pmis : public LinOp, public EnableMultigridLevel<ValueType> {
    GKO_ASSERT_SUPPORTED_VALUE_AND_INDEX_TYPE;

public:
    using value_type = ValueType;
    using index_type = IndexType;

    /**
     * Returns the system operator (matrix) of the linear system.
     *
     * @return the system operator (matrix)
     */
    std::shared_ptr<const LinOp> get_system_matrix() const
    {
        return system_matrix_;
    }


    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * strength_threshold gives the threshold to decide which elements
         * strongly depend on another element. i strongly depends on j
         * when
         * -real(mtx(i, j)) > strength_threshold * max(0, max (k!=i)
         * -real(mtx(i, k))). The test is sign-aware, so a positive
         * off-diagonal is never strong. This value is usually chosen between
         * 0.25 and 0.5.
         */
        remove_complex<ValueType> GKO_FACTORY_PARAMETER_SCALAR(
            strength_threshold, remove_complex<ValueType>{0.25});

        /**
         * max_row_sum strips every dependency from a row that is nearly fully
         * summed: if |row_sum| > max_row_sum * |mtx(i, i)|, where row_sum is
         * mtx(i, i) plus every off-diagonal entry of row i, row i is left with
         * no strong dependency at all. Setting it to 1 disables the filter,
         * which is how the parameter spells "off". Values outside (0, 1] are
         * rejected by an assertion. Stripping every row leaves no coarse point
         * to select, which generate() reports as terminal coarsening rather
         * than raising; see the class-level note.
         */
        remove_complex<ValueType> GKO_FACTORY_PARAMETER_SCALAR(
            max_row_sum, remove_complex<ValueType>{0.9});

        /**
         * p_max_elmts bounds the number of entries kept per row of the
         * assembled prolongation: a row with more than p_max_elmts entries
         * keeps only the p_max_elmts of largest magnitude, and the survivors
         * are rescaled so that their sum reproduces the row's original sum,
         * except when the kept entries themselves sum to zero, where rescaling
         * is skipped rather than dividing by zero. A row with p_max_elmts or
         * fewer entries is left untouched.
         *
         * Truncation is not confined to the level it acts on: a narrower P
         * changes the Galerkin product, hence the coarse operator and how far
         * coarsening proceeds below it, so expect a different hierarchy rather
         * than merely narrower rows. The default of 0 disables it, and it
         * matters most for `ext_i`, which deliberately produces wide rows.
         */
        size_type GKO_FACTORY_PARAMETER_SCALAR(p_max_elmts, size_type{0});


        /**
         * The `system_matrix`, which will be given to this factory, must be
         * sorted (first by row, then by column) in order for the algorithm
         * to work. If it is known that the matrix will be sorted, this
         * parameter can be set to `true` to skip the sorting (therefore,
         * shortening the runtime).
         * However, if it is unknown or if the matrix is known to be not sorted,
         * it must remain `false`, otherwise, this multigrid_level might be
         * incorrect.
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(skip_sorting, false);

        /** The interpolation scheme used to build the prolongation. */
        interpolation_type GKO_FACTORY_PARAMETER_SCALAR(
            interpolation, interpolation_type::direct);
    };
    GKO_ENABLE_LIN_OP_FACTORY(Pmis, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

    /**
     * Create the parameters from the property_tree.
     * Because this is directly tied to the specific type, the value/index type
     * settings within config are ignored and type_descriptor is only used
     * for children configs.
     *
     * @param config  the property tree for setting
     * @param context  the registry
     * @param td_for_child  the type descriptor for children configs. The
     *                      default uses the value/index type of this class.
     *
     * @return parameters
     */
    static parameters_type parse(
        const config::pnode& config, const config::registry& context,
        const config::type_descriptor& td_for_child =
            config::make_type_descriptor<ValueType, IndexType>());

protected:
    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

    explicit Pmis(std::shared_ptr<const Executor> exec);

    explicit Pmis(const Factory* factory,
                  std::shared_ptr<const LinOp> system_matrix);

    void generate();

private:
    std::shared_ptr<const LinOp> system_matrix_{};
};


}  // namespace multigrid
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MULTIGRID_PMIS_HPP_
