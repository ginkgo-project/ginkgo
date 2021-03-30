/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#ifndef GKO_PUBLIC_CORE_SOLVER_CB_GMRES_HPP_
#define GKO_PUBLIC_CORE_SOLVER_CB_GMRES_HPP_


#include <memory>
#include <vector>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/criterion.hpp>


namespace gko {
namespace solver {


namespace cb_gmres {


/**
 * Describes the storage precision that is used in CB-GMRES.
 *
 * The storage precision is described relative to the ValueType:
 * - keep: The storage precision is the same as the ValueType.
 * - reduce1: The storage type is the ValueType reduced in precision once,
 *            for example, ValueType == double -> storage precision == float
 * - reduce2: ValueType precision is reduced twice
 * - integer: The storage precision is an integer of the same size of
 *            ValueType. Note that complex values are not supported.
 * - ireduce1: The storage precision is an integer of the same size as
 *             a reduced ValueType.
 * - ireduce2: The storage precision is an integer of the same size as
 *             a twice reduced ValueType.
 *
 * Precision reduction works as follows:
 * - double -> float -> half -> half -> ... (half is the lowest supported
 *   precision)
 * - std::complex<double> -> std::complex<float> -> std::complex<half>
 *   -> std::complex<half> ... (std::complex<half> is the lowest supported
 *   precision)
 *
 * To integer conversions:
 * - double -> int64
 * - float -> int32
 * - half -> int16
 */
enum class storage_precision {
    keep,
    reduce1,
    reduce2,
    integer,
    ireduce1,
    ireduce2
};


}  // namespace cb_gmres


/**
 * CB-GMRES or the compressed basis generalized minimal residual method is an
 * iterative type Krylov subspace method which is suitable for nonsymmetric
 * linear systems.
 *
 * The implementation in Ginkgo makes use of the merged kernel to make the best
 * use of data locality. The inner operations in one iteration of CB-GMRES
 * are merged into 2 separate steps. Classical Gram-Schmidt with
 * reorthogonalization is used.
 *
 * The krylov basis can be stored in reduced precision (compressed) to reduce
 * memory accesses, while all computations (including krylov basis operations)
 * are performed in the same arithmetic precision ValueType. By default, the
 * krylov basis are stored in one precision lower than ValueType.
 *
 * @tparam ValueType  the arithmetic precision and the precision of matrix
 *                    elements
 *
 * @ingroup solvers
 * @ingroup LinOp
 */
template <typename ValueType = default_precision>
class CbGmres : public EnableLinOp<CbGmres<ValueType>>,
                public Preconditionable {
    friend class EnableLinOp<CbGmres>;
    friend class EnablePolymorphicObject<CbGmres, LinOp>;

public:
    using value_type = ValueType;

    /**
     * Gets the system operator (matrix) of the linear system.
     *
     * @return the system operator (matrix)
     */
    std::shared_ptr<const LinOp> get_system_matrix() const
    {
        return system_matrix_;
    }

    /**
     * Returns the krylov dimension.
     *
     * @return the krylov dimension
     */
    size_type get_krylov_dim() const { return krylov_dim_; }

    /**
     * Returns the storage precision used internally.
     *
     * @return the storage precision used internally
     */
    cb_gmres::storage_precision get_storage_precision() const
    {
        return storage_precision_;
    }

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Determines which storage type is used.
         */
        cb_gmres::storage_precision GKO_FACTORY_PARAMETER_SCALAR(
            storage_precision, cb_gmres::storage_precision::reduce1);

        /**
         * Criterion factories.
         */
        std::vector<std::shared_ptr<const stop::CriterionFactory>>
            GKO_FACTORY_PARAMETER_VECTOR(criteria, nullptr);

        /**
         * Preconditioner factory.
         */
        std::shared_ptr<const LinOpFactory> GKO_FACTORY_PARAMETER_SCALAR(
            preconditioner, nullptr);

        /**
         * Already generated preconditioner. If one is provided, the factory
         * `preconditioner` will be ignored.
         */
        std::shared_ptr<const LinOp> GKO_FACTORY_PARAMETER_SCALAR(
            generated_preconditioner, nullptr);

        /**
         * krylov dimension factory.
         */
        size_type GKO_FACTORY_PARAMETER_SCALAR(krylov_dim, 100u);
    };
    GKO_ENABLE_LIN_OP_FACTORY(CbGmres, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    void apply_impl(const LinOp *b, LinOp *x) const override;

    void apply_dense_impl(const matrix::Dense<ValueType> *b,
                          matrix::Dense<ValueType> *x) const;

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override;

    explicit CbGmres(std::shared_ptr<const Executor> exec)
        : EnableLinOp<CbGmres>(std::move(exec))
    {}

    explicit CbGmres(const Factory *factory,
                     std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<CbGmres>(factory->get_executor(),
                               transpose(system_matrix->get_size())),
          parameters_{factory->get_parameters()},
          system_matrix_{std::move(system_matrix)}
    {
        if (parameters_.generated_preconditioner) {
            GKO_ASSERT_EQUAL_DIMENSIONS(parameters_.generated_preconditioner,
                                        this);
            set_preconditioner(parameters_.generated_preconditioner);
        } else if (parameters_.preconditioner) {
            set_preconditioner(
                parameters_.preconditioner->generate(system_matrix_));
        } else {
            set_preconditioner(matrix::Identity<ValueType>::create(
                this->get_executor(), this->get_size()[0]));
        }
        krylov_dim_ = parameters_.krylov_dim;
        stop_criterion_factory_ =
            stop::combine(std::move(parameters_.criteria));
        storage_precision_ = parameters_.storage_precision;
    }

private:
    std::shared_ptr<const LinOp> system_matrix_{};
    std::shared_ptr<const stop::CriterionFactory> stop_criterion_factory_{};
    size_type krylov_dim_;
    cb_gmres::storage_precision storage_precision_;
};


}  // namespace solver
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_SOLVER_CB_GMRES_HPP_
