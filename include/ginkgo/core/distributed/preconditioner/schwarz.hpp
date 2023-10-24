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

#ifndef GKO_PUBLIC_CORE_DISTRIBUTED_PRECONDITIONER_SCHWARZ_HPP_
#define GKO_PUBLIC_CORE_DISTRIBUTED_PRECONDITIONER_SCHWARZ_HPP_


#include <ginkgo/config.hpp>


#if GINKGO_BUILD_MPI


#include <ginkgo/core/base/abstract_factory.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/vector.hpp>


namespace gko {
namespace experimental {
namespace distributed {
/**
 * @brief The Preconditioner namespace.
 *
 * @ingroup precond
 */
namespace preconditioner {


/**
 * A Schwarz preconditioner is a simple domain decomposition preconditioner that
 * generalizes the Block Jacobi preconditioner, incorporating options for
 * different local subdomain solvers and overlaps between the subdomains.
 *
 * See Iterative Methods for Sparse Linear Systems (Y. Saad) for a general
 * treatment and variations of the method.
 *
 * @note Currently overlap and coarse grid correction are not supported (TODO).
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  integral type of the preconditioner
 *
 * @ingroup schwarz
 * @ingroup precond
 * @ingroup LinOp
 */
template <typename ValueType = default_precision,
          typename LocalIndexType = int32, typename GlobalIndexType = int64>
class Schwarz
    : public EnableLinOp<Schwarz<ValueType, LocalIndexType, GlobalIndexType>> {
    friend class EnableLinOp<Schwarz>;
    friend class EnablePolymorphicObject<Schwarz, LinOp>;

public:
    using EnableLinOp<Schwarz>::convert_to;
    using EnableLinOp<Schwarz>::move_to;
    using value_type = ValueType;
    using index_type = GlobalIndexType;
    using local_index_type = LocalIndexType;
    using global_index_type = GlobalIndexType;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Local solver factory.
         */
        std::shared_ptr<const LinOpFactory> GKO_DEFERRED_FACTORY_PARAMETER(
            local_solver);

        /**
         * Generated Inner solvers.
         */
        std::shared_ptr<const LinOp> GKO_FACTORY_PARAMETER_SCALAR(
            generated_local_solver, nullptr);
    };
    GKO_ENABLE_LIN_OP_FACTORY(Schwarz, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    /**
     * Creates an empty Schwarz preconditioner.
     *
     * @param exec  the executor this object is assigned to
     */
    explicit Schwarz(std::shared_ptr<const Executor> exec)
        : EnableLinOp<Schwarz>(std::move(exec))
    {}

    /**
     * Creates a Schwarz preconditioner from a matrix using a Schwarz::Factory.
     *
     * @param factory  the factory to use to create the preconditioner
     * @param system_matrix  the matrix this preconditioner should be created
     *                       from
     */
    explicit Schwarz(const Factory* factory,
                     std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<Schwarz>(factory->get_executor(),
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

    std::shared_ptr<const LinOp> local_solver_;
};


}  // namespace preconditioner
}  // namespace distributed
}  // namespace experimental
}  // namespace gko


#endif  // GINKGO_BUILD_MPI
#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_PRECONDITIONER_SCHWARZ_HPP_
