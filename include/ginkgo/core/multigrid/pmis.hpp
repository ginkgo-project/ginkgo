// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_MULTIGRID_PMIS_HPP_
#define GKO_PUBLIC_CORE_MULTIGRID_PMIS_HPP_


#include <vector>

#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/config/type_descriptor.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/multigrid/multigrid_level.hpp>


namespace gko {
namespace multigrid {


/**
 * Parallel modified independent set (Pmis) is the classical coarsening method
 * introduced in the paper H. De Sterck et al., "Reducing complexity in parallel
 * algebraic multigrid preconditioners".
 *
 * Pmis creates the coarse- and fine-points group according to the matrix value
 * not the structure.
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  precision of matrix indexes
 *
 * @ingroup MultigridLevel
 * @ingroup Multigrid
 * @ingroup LinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class Pmis : public EnableLinOp<Pmis<ValueType, IndexType>>,
             public EnableMultigridLevel<ValueType> {
    friend class EnableLinOp<Pmis>;
    friend class EnablePolymorphicObject<Pmis, LinOp>;

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
         * strongly depends on another element. if |mtx(i, j)| >=
         * strength_threshold * max (k!=i) abs(mtx(i, k)), i strongly depends on
         * j. This value is usually chosen between 0.25 and 0.5. We use the same
         * default value 0.25 as HYPRE.
         */
        remove_complex<ValueType> GKO_FACTORY_PARAMETER_SCALAR(
            strength_threshold, remove_complex<ValueType>{0.25});


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
    void apply_impl(const LinOp* b, LinOp* x) const override
    {
        this->get_composition()->apply(b, x);
    }

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override
    {
        this->get_composition()->apply(alpha, b, beta, x);
    }

    explicit Pmis(std::shared_ptr<const Executor> exec)
        : EnableLinOp<Pmis>(std::move(exec))
    {}

    explicit Pmis(const Factory* factory,
                  std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<Pmis>(factory->get_executor(), system_matrix->get_size()),
          EnableMultigridLevel<ValueType>(system_matrix),
          parameters_{factory->get_parameters()},
          system_matrix_{system_matrix},
          weight_(factory->get_executor(), system_matrix_->get_size()[0]),
          status_(factory->get_executor(), system_matrix_->get_size()[0])
    {
        GKO_ASSERT(parameters_.strength_threshold <= 1.0);
        GKO_ASSERT(parameters_.strength_threshold >= 0.0);
        if (system_matrix_->get_size()[0] != 0) {
            // generate on the existed matrix
            this->generate();
        }
    }

    void generate();

    /**
     * This function generates the local matrix coarsening operators.
     *
     * @return a tuple with prolongation, coarse, and restriction linop
     */
    std::tuple<std::shared_ptr<LinOp>, std::shared_ptr<LinOp>,
               std::shared_ptr<LinOp>>
    generate_local(
        std::shared_ptr<const matrix::Csr<ValueType, IndexType>> local_matrix);

private:
    std::shared_ptr<const LinOp> system_matrix_{};
    // weight = the number of strong dependence + rand[0, 1]
    gko::array<remove_complex<ValueType>> weight_;
    // status records the nodes set status. 0: not assigned, 1: fine group 2:
    // coarse group
    gko::array<int> status_;
};


}  // namespace multigrid
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MULTIGRID_PMIS_HPP_
