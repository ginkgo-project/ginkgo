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

#ifndef GKO_PUBLIC_CORE_DISTRIBUTED_PRECONDITIONER_BDDC_HPP_
#define GKO_PUBLIC_CORE_DISTRIBUTED_PRECONDITIONER_BDDC_HPP_


#include <ginkgo/config.hpp>


#include <map>
#include <set>


#if GINKGO_BUILD_MPI


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/vector.hpp>
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
    using mat_data = matrix_data<ValueType, IndexType>;
    using diag_type = matrix::Diagonal<ValueType>;
    using vec_type = matrix::Dense<ValueType>;

    /**
     * Returns the number of blocks of the operator.
     *
     * @return the number of blocks of the operator
     */
    std::shared_ptr<const matrix_type> get_local_system_matrix() const
    {
        return local_system_matrix_;
    }


    std::shared_ptr<const vec_type> get_phi() const { return phi_; }


    std::shared_ptr<const vec_type> get_local_coarse_matrix() const
    {
        return local_coarse_matrix_;
    }


    std::shared_ptr<const diag_type> get_weights() const { return weights_; }


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
            local_solver_factory, nullptr);

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

        std::set<index_type> GKO_FACTORY_PARAMETER_VECTOR(boundary_idxs, {});
    };
    GKO_ENABLE_LIN_OP_FACTORY(Bddc, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

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

    void generate_constraints();

    void schur_complement_solve();

    void generate_coarse_system();

    void restrict_residual(const LinOp* global_residual) const;

    void coarsen_residual() const;

    void prolong_coarse_solution() const;

    void apply_impl(const LinOp* b, LinOp* x) const override;

    template <typename VectorType>
    void apply_dense_impl(const VectorType* b, VectorType* x) const;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

private:
    std::shared_ptr<const global_matrix_type> global_system_matrix_;
    std::shared_ptr<global_matrix_type> global_coarse_matrix_;
    std::shared_ptr<matrix_type> local_system_matrix_;
    std::shared_ptr<matrix_type> inner_system_matrix_;
    std::shared_ptr<matrix_type> outer_system_matrix_;
    std::shared_ptr<const LinOp> local_solver_;
    std::shared_ptr<const LinOp> schur_complement_solver_;
    std::shared_ptr<const LinOp> coarse_solver_;
    std::shared_ptr<const LinOp> inner_solver_;
    std::shared_ptr<matrix_type> constraints_;
    std::shared_ptr<vec_type> local_coarse_matrix_;
    std::shared_ptr<diag_type> weights_;
    std::shared_ptr<vec_type> phi_;
    std::shared_ptr<vec_type> phi_t_;
    std::vector<index_type> local_rows_;
    mutable std::map<index_type, index_type> global_idx_to_recv_buffer_;
    mutable array<index_type> local_idx_to_send_buffer_;
    mutable array<index_type> recv_buffer_to_global_;
    array<index_type> non_local_to_local_;
    array<index_type> coarse_non_local_to_local_;
    array<index_type> non_local_idxs_;
    array<index_type> local_to_local_;
    array<index_type> coarse_local_to_local_;
    array<index_type> inner_to_local_;
    std::vector<index_type> local_idxs_;
    std::vector<index_type> inner_idxs_;
    array<index_type> local_to_inner_;
    std::vector<index_type> interfaces_;
    std::shared_ptr<vec_type> local_residual_;
    std::shared_ptr<vec_type> local_residual_large_;
    std::shared_ptr<vec_type> local_solution_;
    std::shared_ptr<vec_type> local_solution_large_;
    std::shared_ptr<vec_type> local_coarse_residual_;
    std::shared_ptr<vec_type> local_coarse_solution_;
    std::shared_ptr<vec_type> local_intermediate_;
    std::shared_ptr<vec_type> inner_residual_;
    std::shared_ptr<vec_type> inner_solution_;
    std::shared_ptr<vec_type> one_op_;
    std::shared_ptr<vec_type> neg_one_op_;
    std::shared_ptr<vec_type> host_residual_;
    mutable array<comm_index_type> send_sizes_;
    mutable array<comm_index_type> send_offsets_;
    mutable array<comm_index_type> recv_sizes_;
    mutable array<comm_index_type> recv_offsets_;
    mutable std::vector<comm_index_type> coarse_send_sizes_;
    mutable std::vector<comm_index_type> coarse_send_offsets_;
    mutable std::vector<comm_index_type> coarse_recv_sizes_;
    mutable std::vector<comm_index_type> coarse_recv_offsets_;
    std::vector<comm_index_type> coarse_owners_;
    mutable array<value_type> send_buffer_;
    mutable array<value_type> recv_buffer_;
    mutable std::vector<value_type> coarse_send_buffer_;
    mutable std::vector<value_type> coarse_recv_buffer_;
    mutable array<value_type> coarsening_send_buffer_;
    mutable array<value_type> coarsening_recv_buffer_;
    mutable array<comm_index_type> coarsening_send_sizes_;
    mutable array<comm_index_type> coarsening_send_offsets_;
    mutable array<comm_index_type> coarsening_recv_sizes_;
    mutable array<comm_index_type> coarsening_recv_offsets_;
    array<index_type> coarsening_local_to_send_;
    array<index_type> coarsening_recv_to_local_;
    std::shared_ptr<global_vec_type> coarse_residual_;
    std::shared_ptr<global_vec_type> coarse_solution_;
    std::vector<std::vector<index_type>> interface_dofs_;
    std::vector<std::vector<index_type>> interface_dof_ranks_;
    array<index_type> coarse_non_local_to_global_;
    std::shared_ptr<vec_type> nonlocal_;
    array<IndexType> coarse_local_to_non_local_;
    std::vector<IndexType> dbcs_;
};


}  // namespace preconditioner
}  // namespace distributed
}  // namespace experimental
}  // namespace gko


#endif  // GINKGO_BUILD_MPI
#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_PRECONDITIONER_BDDC_HPP_
