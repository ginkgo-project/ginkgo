// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_DISTRIBUTED_PRECONDITIONER_BDDC_HPP_
#define GKO_PUBLIC_CORE_DISTRIBUTED_PRECONDITIONER_BDDC_HPP_


#include <memory>

#include <ginkgo/config.hpp>

#include "ginkgo/core/base/types.hpp"


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
#include <ginkgo/core/matrix/diagonal.hpp>
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


enum class dof_type { inner, inactive, face, edge, vertex };


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
    using real_type = remove_complex<value_type>;
    using index_type = GlobalIndexType;
    using local_index_type = LocalIndexType;
    using global_index_type = GlobalIndexType;
    using perm_type = matrix::Permutation<local_index_type>;
    using local_mtx = matrix::Csr<value_type, local_index_type>;
    using local_real_mtx = matrix::Csr<real_type, local_index_type>;
    using local_vec = matrix::Dense<value_type>;
    using local_real_vec = matrix::Dense<real_type>;
    using vec = Vector<value_type>;
    using diag = matrix::Diagonal<value_type>;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Local solver factory.
         */
        std::shared_ptr<const LinOpFactory> GKO_DEFERRED_FACTORY_PARAMETER(
            local_solver);

        /**
         * Coarse solver factory.
         */
        std::shared_ptr<const LinOpFactory> GKO_DEFERRED_FACTORY_PARAMETER(
            coarse_solver);

        /**
         * Reordering factory to use on the local matrices for reducing fill-in.
         */
        std::shared_ptr<const LinOpFactory> GKO_DEFERRED_FACTORY_PARAMETER(
            reordering);

        /**
         * Use of Vertex constraints.
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(vertices, true);

        /**
         * Use of Edge constraints.
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(edges, true);

        /**
         * Use of Face constraints.
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(faces, true);

        std::shared_ptr<const stop::CriterionFactory>
            GKO_DEFERRED_FACTORY_PARAMETER(local_criterion);

        bool GKO_FACTORY_PARAMETER_SCALAR(repartition_coarse, false);
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

    std::shared_ptr<const LinOp> restriction_;
    std::shared_ptr<const LinOp> prolongation_;
    std::shared_ptr<const LinOp> coarse_restriction_;
    std::shared_ptr<const LinOp> coarse_prolongation_;
    std::shared_ptr<const LinOp> inner_solver_;
    std::shared_ptr<const LinOp> local_solver_;
    std::shared_ptr<const LinOp> schur_solver_;
    std::shared_ptr<const LinOp> coarse_solver_;
    std::shared_ptr<const perm_type> permutation_;
    std::shared_ptr<local_real_mtx> constraints_;
    std::shared_ptr<local_real_mtx> constraints_t_;
    std::shared_ptr<local_vec> phi_;
    std::shared_ptr<local_vec> phi_t_;
    std::shared_ptr<LinOp> weights_;
    std::shared_ptr<vec> buf_1_;
    std::shared_ptr<vec> buf_2_;
    std::shared_ptr<vec> coarse_buf_1_;
    std::shared_ptr<vec> coarse_buf_2_;
    std::shared_ptr<local_vec> local_buf_1_;
    std::shared_ptr<local_vec> interior_1_;
    std::shared_ptr<local_vec> bndry_1_;
    std::shared_ptr<local_vec> dual_1_;
    std::shared_ptr<local_vec> primal_1_;
    std::shared_ptr<local_vec> local_buf_2_;
    std::shared_ptr<local_vec> interior_2_;
    std::shared_ptr<local_vec> bndry_2_;
    std::shared_ptr<local_vec> dual_2_;
    std::shared_ptr<local_vec> primal_2_;
    std::shared_ptr<local_vec> local_buf_3_;
    std::shared_ptr<local_vec> interior_3_;
    std::shared_ptr<local_vec> bndry_3_;
    std::shared_ptr<local_vec> dual_3_;
    std::shared_ptr<local_vec> dual_4_;
    std::shared_ptr<local_vec> primal_3_;
    std::shared_ptr<local_vec> local_buf_4_;
    std::shared_ptr<local_vec> schur_buf_1_;
    std::shared_ptr<local_vec> schur_buf_2_;
    std::shared_ptr<vec> broken_coarse_buf_1_;
    std::shared_ptr<vec> broken_coarse_buf_2_;
    std::shared_ptr<local_vec> local_coarse_buf_1_;
    std::shared_ptr<local_vec> local_coarse_buf_2_;
    std::shared_ptr<local_mtx> A_LL;
    std::shared_ptr<const local_mtx> A_LP;
    std::shared_ptr<const local_mtx> A_PL;
    std::shared_ptr<const local_mtx> A_PP;
    std::shared_ptr<const local_mtx> A_IB;
    std::shared_ptr<const local_mtx> A_BI;
    std::shared_ptr<const local_mtx> A_II_;
    std::shared_ptr<const LinOp> one_;
    std::shared_ptr<const LinOp> neg_one_;
    std::shared_ptr<const LinOp> zero_;
    std::shared_ptr<const matrix::Permutation<LocalIndexType>> reorder_LL_;
    std::shared_ptr<const matrix::Permutation<LocalIndexType>> reorder_II_;
    std::shared_ptr<local_vec> schur_interm_;
};


}  // namespace preconditioner
}  // namespace distributed
}  // namespace experimental
}  // namespace gko


#endif  // GINKGO_BUILD_MPI
#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_PRECONDITIONER_BDDC_HPP_
