// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_MULTIGRID_HMIS_HPP_
#define GKO_PUBLIC_CORE_MULTIGRID_HMIS_HPP_


#include <vector>

#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/config/type_descriptor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/multigrid/multigrid_level.hpp>


namespace gko {
namespace multigrid {


/**
 * HMIS (Hybrid Modified Independent Set) is a classical AMG coarsening
 * algorithm, as used by default in hypre's BoomerAMG.
 *
 * It performs a C/F (coarse/fine) splitting using a two-phase approach:
 * first a sequential Ruge-Stueben first pass, then PMIS (Parallel Modified
 * Independent Set) to finalize any remaining undecided points.
 * Prolongation is built using classical interpolation (Ruge-Stueben 1987).
 * The coarse operator is computed via the Galerkin triple product R*A*P.
 *
 * References:
 * - De Sterck, Yang, Heys: "Reducing Complexity in Parallel Algebraic
 *   Multigrid Preconditioners", SIAM JMAA 27(4), 2006.
 * - Ruge, Stueben: "Algebraic Multigrid" in Multigrid Methods,
 *   Frontiers in Applied Mathematics vol. 3, 1987.
 * - hypre source: par_coarsen.c, par_interp.c, par_strength.c
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  precision of matrix indexes
 *
 * @ingroup MultigridLevel
 * @ingroup Multigrid
 * @ingroup LinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class Hmis : public EnableLinOp<Hmis<ValueType, IndexType>>,
             public EnableMultigridLevel<ValueType> {
    friend class EnableLinOp<Hmis>;
    friend class EnablePolymorphicObject<Hmis, LinOp>;

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

    /**
     * Returns the C/F marker array.
     *
     * The marker array has the same size as the number of rows.
     * cf_marker[i] = 1 means point i is a C-point (coarse),
     * cf_marker[i] = 0 means point i is an F-point (fine).
     *
     * @return the C/F marker array.
     */
    const IndexType* get_const_cf_marker() const noexcept
    {
        return cf_marker_.get_const_data();
    }

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Strength threshold (theta) for the strength-of-connection.
         * A connection a_ij is strong if -a_ij >= theta * max_{k!=i}(-a_ik).
         * Default 0.25 matches hypre's BoomerAMG default.
         */
        double GKO_FACTORY_PARAMETER_SCALAR(strength_threshold, 0.25);

        /**
         * Maximum number of PMIS iterations for the C/F splitting.
         */
        unsigned GKO_FACTORY_PARAMETER_SCALAR(max_iterations, 100u);

        /**
         * The `system_matrix`, which will be given to this factory, must be
         * sorted (first by row, then by column) in order for the algorithm
         * to work. If it is known that the matrix will be sorted, this
         * parameter can be set to `true` to skip the sorting.
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(skip_sorting, false);
    };
    GKO_ENABLE_LIN_OP_FACTORY(Hmis, parameters, Factory);
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
    void apply_impl(const LinOp* b, LinOp* x) const override
    {
        this->get_composition()->apply(b, x);
    }

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override
    {
        this->get_composition()->apply(alpha, b, beta, x);
    }

    explicit Hmis(std::shared_ptr<const Executor> exec)
        : EnableLinOp<Hmis>(std::move(exec))
    {}

    explicit Hmis(const Factory* factory,
                  std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<Hmis>(factory->get_executor(), system_matrix->get_size()),
          EnableMultigridLevel<ValueType>(system_matrix),
          parameters_{factory->get_parameters()},
          system_matrix_{system_matrix},
          cf_marker_(factory->get_executor(), system_matrix_->get_size()[0])
    {
        GKO_ASSERT(parameters_.strength_threshold >= 0.0);
        GKO_ASSERT(parameters_.strength_threshold <= 1.0);
        if (system_matrix_->get_size()[0] != 0) {
            this->generate();
        }
    }

    void generate();

    /**
     * Generates the local multigrid coarsening operators.
     *
     * @return a tuple with prolongation, coarse, and restriction linop
     */
    std::tuple<std::shared_ptr<LinOp>, std::shared_ptr<LinOp>,
               std::shared_ptr<LinOp>>
    generate_local(
        std::shared_ptr<const matrix::Csr<ValueType, IndexType>> local_matrix);

private:
    std::shared_ptr<const LinOp> system_matrix_{};
    array<IndexType> cf_marker_;
};


}  // namespace multigrid
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MULTIGRID_HMIS_HPP_
