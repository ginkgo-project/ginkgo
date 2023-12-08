// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_DISTRIBUTED_PRECONDITIONER_BDDC_HPP_
#define GKO_PUBLIC_CORE_DISTRIBUTED_PRECONDITIONER_BDDC_HPP_


#include <map>
#include <set>


#include <ginkgo/config.hpp>


#if GINKGO_BUILD_MPI


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/combination.hpp>
#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/preconditioner/schwarz.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/factorization/factorization.hpp>
#include <ginkgo/core/matrix/csr.hpp>


namespace gko {
namespace experimental {
namespace distributed {
/**
 * @brief The Preconditioner namespace.
 *
 * @ingroup precond
 */
namespace preconditioner {


template <typename ValueType = default_precision, typename IndexType = int32>
class Bddc : public EnableLinOp<Bddc<ValueType, IndexType>> {
    friend class EnableLinOp<Bddc>;
    friend class EnablePolymorphicObject<Bddc, LinOp>;

public:
    using EnableLinOp<Bddc>::convert_to;
    using EnableLinOp<Bddc>::move_to;
    using value_type = ValueType;
    using index_type = IndexType;
    using global_matrix_type =
        experimental::distributed::Matrix<ValueType, IndexType, IndexType>;
    using global_vec_type = experimental::distributed::Vector<ValueType>;
    using matrix_type = matrix::Csr<ValueType, IndexType>;
    using diag_type = matrix::Diagonal<ValueType>;
    using vec_type = matrix::Dense<ValueType>;
    using compo_type = Composition<ValueType>;
    using combi_type = Combination<ValueType>;
    using schwarz = gko::experimental::distributed::preconditioner::Schwarz<
        ValueType, IndexType, IndexType>;
    using fact_type =
        gko::experimental::factorization::Factorization<ValueType, IndexType>;


    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Local system matrix data.
         */
        std::shared_ptr<const LinOp> GKO_FACTORY_PARAMETER_SCALAR(
            local_system_matrix, nullptr);

        /**
         * Local solver factory.
         */
        std::shared_ptr<const LinOpFactory> GKO_FACTORY_PARAMETER_SCALAR(
            local_factorization_factory, nullptr);

        /**
         * Schur complement solver factory.
         */
        std::shared_ptr<const LinOpFactory> GKO_FACTORY_PARAMETER_SCALAR(
            schur_complement_solver_factory, nullptr);

        /**
         * Inner solver factory.
         */
        std::shared_ptr<const LinOpFactory> GKO_FACTORY_PARAMETER_SCALAR(
            inner_solver_factory, nullptr);

        /**
         * Coarse solver factory.
         */
        std::shared_ptr<const LinOpFactory> GKO_FACTORY_PARAMETER_SCALAR(
            coarse_solver_factory, nullptr);

        bool GKO_FACTORY_PARAMETER_SCALAR(static_condensation, true);

        /**
         * A list of all interfaces in the global matrix. An interface can
         * have one (corner) or multiple (edges / faces) entries.
         */
        std::vector<std::vector<index_type>> GKO_FACTORY_PARAMETER_VECTOR(
            interface_dofs, 0);

        /**
         * A list of lists containing all ranks sharing each interface dof.
         */
        std::vector<std::vector<index_type>> GKO_FACTORY_PARAMETER_VECTOR(
            interface_dof_ranks, 0);

        /**
         * A rank local list of interior dofs.
         */
        std::vector<index_type> GKO_FACTORY_PARAMETER_VECTOR(interior_dofs, 0);

        std::set<index_type> GKO_FACTORY_PARAMETER_VECTOR(boundary_idxs, {});

        bool GKO_FACTORY_PARAMETER_SCALAR(skip_sorting_interfaces, false);
    };
    GKO_ENABLE_LIN_OP_FACTORY(Bddc, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

    std::shared_ptr<const LinOp> get_schur() const { return global_schur; }

    void pre_solve(const LinOp* b, LinOp* x) const;

    void post_solve(const LinOp* b, LinOp* x) const;

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
     * Creates a Bddc preconditioner using a Bddc::Factory and its parameters.
     *
     * @param factory  the factory to use to create the preconditoner
     * @param system_matrix  the matrix this preconditioner should be created
     *                       from
     */
    explicit Bddc(const Factory* factory,
                  std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<Bddc>(factory->get_executor(),
                            gko::transpose(system_matrix->get_size())),
          parameters_{factory->get_parameters()},
          global_system_matrix_{
              as<experimental::distributed::Matrix<ValueType, IndexType,
                                                   IndexType>>(system_matrix)}
    {
        this->generate();
    }

    /**
     * Generates the preconditoner.
     *
     * @param system_matrix  the source matrix used to generate the
     *                       preconditioner
     */
    void generate();

    void generate_interfaces();

    void apply_impl(const LinOp* b, LinOp* x) const override;

    template <typename VectorType>
    void apply_dense_impl(const VectorType* b, VectorType* x) const;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

private:
    std::shared_ptr<matrix_type> A_ig;
    std::shared_ptr<matrix_type> A_gi;
    std::shared_ptr<matrix_type> c;
    std::shared_ptr<matrix_type> cT;
    std::shared_ptr<const global_matrix_type> global_system_matrix_;
    std::shared_ptr<global_matrix_type> global_coarse_matrix_;
    std::shared_ptr<global_matrix_type> R;
    std::shared_ptr<global_matrix_type> RT;
    std::shared_ptr<global_matrix_type> RG;
    std::shared_ptr<global_matrix_type> RGT;
    std::shared_ptr<global_matrix_type> RC;
    std::shared_ptr<global_matrix_type> RCT;
    std::shared_ptr<const LinOp> inner_solver;
    std::shared_ptr<const LinOp> edge_solver;
    std::shared_ptr<const LinOp> local_schur_solver;
    std::shared_ptr<const LinOp> coarse_solver;
    std::shared_ptr<const diag_type> weights;
    std::shared_ptr<vec_type> phi;
    std::shared_ptr<vec_type> phi_t;
    std::vector<index_type> interfaces_;
    std::vector<index_type> inner_idxs_;
    std::vector<index_type> edge_idxs;
    std::shared_ptr<vec_type> one_op;
    std::shared_ptr<vec_type> neg_one_op;
    std::vector<std::vector<index_type>> interface_dofs_;
    std::vector<std::vector<index_type>> interface_dof_ranks_;
    std::shared_ptr<global_vec_type> restricted_residual;
    std::shared_ptr<global_vec_type> restricted_solution;
    std::shared_ptr<global_vec_type> schur_residual;
    std::shared_ptr<global_vec_type> schur_solution;
    std::shared_ptr<global_vec_type> coarse_residual;
    std::shared_ptr<global_vec_type> coarse_solution;
    std::shared_ptr<global_vec_type> coarse_b;
    std::shared_ptr<global_vec_type> coarse_x;
    std::shared_ptr<vec_type> inner_intermediate;
    std::shared_ptr<const LinOp> global_schur;
    std::shared_ptr<vec_type> coarse_1;
    std::shared_ptr<vec_type> coarse_2;
    std::shared_ptr<vec_type> coarse_3;
    std::shared_ptr<vec_type> local_1;
    std::shared_ptr<vec_type> local_2;
    std::shared_ptr<vec_type> local_3;
};


}  // namespace preconditioner
}  // namespace distributed
}  // namespace experimental
}  // namespace gko


#endif  // GINKGO_BUILD_MPI
#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_PRECONDITIONER_BDDC_HPP_
