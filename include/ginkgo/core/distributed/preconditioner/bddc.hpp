// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_DISTRIBUTED_PRECONDITIONER_BDDC_HPP_
#define GKO_PUBLIC_CORE_DISTRIBUTED_PRECONDITIONER_BDDC_HPP_


#include <ginkgo/config.hpp>


#if GINKGO_BUILD_MPI


#include <ginkgo/core/base/abstract_factory.hpp>
#include <ginkgo/core/base/intrinsics.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/config/type_descriptor.hpp>
#include <ginkgo/core/distributed/dd_matrix.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/permutation.hpp>


namespace gko {
namespace experimental {
namespace distributed {
/**
 * @brief The Preconditioner namespace.
 *
 * @ingroup precond
 */
namespace preconditioner {


enum class dof_type { inner, vertex, edge, face };


/**
 * BDDC is a two-level, non-overlapping substructuring method.
 *
 * @tparam ValueType  precision of matrix element
 * @tparam LocalIndexType  local integer type of the matrix
 * @tparam GlobalIndexType  global integer type of the matrix
 *
 * @ingroup bddc
 * @ingroup precond
 * @ingroup LinOp
 */
template <typename ValueType = default_precision,
          typename LocalIndexType = int32, typename GlobalIndexType = int64>
class Bddc
    : public EnableLinOp<Bddc<ValueType, LocalIndexType, GlobalIndexType>> {
    friend class EnableLinOp<Bddc>;
    friend class EnablePolymorphicObject<Bddc, LinOp>;

public:
    using EnableLinOp<Bddc>::convert_to;
    using EnableLinOp<Bddc>::move_to;
    using value_type = ValueType;
    using index_type = GlobalIndexType;
    using local_index_type = LocalIndexType;
    using global_index_type = GlobalIndexType;
    using perm_type = matrix::Permutation<local_index_type>;
    using local_mtx = matrix::Csr<value_type, local_index_type>;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Local solver factory.
         */
        std::shared_ptr<const LinOpFactory> GKO_DEFERRED_FACTORY_PARAMETER(
            local_solver);
    };
    GKO_ENABLE_LIN_OP_FACTORY(Bddc, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

    /**
     * Create the parameters from the property_tree.
     * Because this is directly tied to the specific type, the value/index type
     * settings within config are ignored and type_descriptor is only used
     * for children objects.
     *
     * @param config  the property tree for setting
     * @param context  the registry
     * @param td_for_child  the type descriptor for children objects. The
     *                      default uses the value/local/global index type of
     *                      this class.
     *
     * @return parameters
     */
    static parameters_type parse(
        const config::pnode& config, const config::registry& context,
        const config::type_descriptor& td_for_child =
            config::make_type_descriptor<ValueType, LocalIndexType,
                                         GlobalIndexType>());

protected:
    /**
     * Creates an empty Bddc preconditioner.
     *
     * @param exec  the executor this object is assigned to
     */
    explicit Bddc(std::shared_ptr<const Executor> exec)
        : EnableLinOp<Bddc>(std::move(exec))
    {}

    /**
     * Creates a Bddc preconditioner from a matrix using a Bddc::Factory.
     *
     * @param factory  the factory to use to create the preconditioner
     * @param system_matrix  the matrix this preconditioner should be created
     *                       from
     */
    explicit Bddc(const Factory* factory,
                  std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<Bddc>(factory->get_executor(),
                            gko::transpose(system_matrix->get_size())),
          parameters_{factory->get_parameters()}
    {
        this->generate(system_matrix);
    }

    /**
     * Generates the preconditioner.
     */
    void generate(std::shared_ptr<const LinOp> system_matrix);

    void apply_impl(const LinOp* b, LinOp* x) const override;

    template <typename VectorType>
    void apply_dense_impl(const VectorType* b, VectorType* x) const;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

private:
    /**
     * Sets the solver operator used as the local solver.
     *
     * @param new_solver  the new local solver
     */
    void set_solver(std::shared_ptr<const LinOp> new_solver);

    std::shared_ptr<const LinOp> inner_solver_;
    std::shared_ptr<const LinOp> local_solver_;
    std::shared_ptr<const perm_type> permutation_;
};


}  // namespace preconditioner
}  // namespace distributed
}  // namespace experimental
}  // namespace gko


#endif  // GINKGO_BUILD_MPI
#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_PRECONDITIONER_BDDC_HPP_
