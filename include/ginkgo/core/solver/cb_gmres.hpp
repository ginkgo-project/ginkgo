// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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
#include <ginkgo/core/solver/solver_base.hpp>
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
 * The Krylov basis can be stored in reduced precision (compressed) to reduce
 * memory accesses, while all computations (including Krylov basis operations)
 * are performed in the same arithmetic precision ValueType. By default, the
 * Krylov basis are stored in one precision lower than ValueType.
 *
 * @tparam ValueType  the arithmetic precision and the precision of matrix
 *                    elements
 *
 * @ingroup solvers
 * @ingroup LinOp
 */
template <typename ValueType = default_precision>
class CbGmres : public EnableLinOp<CbGmres<ValueType>>,
                public EnablePreconditionedIterativeSolver<ValueType,
                                                           CbGmres<ValueType>> {
    friend class EnableLinOp<CbGmres>;
    friend class EnablePolymorphicObject<CbGmres, LinOp>;

public:
    using value_type = ValueType;

    /**
     * Returns the Krylov dimension.
     *
     * @return the Krylov dimension
     */
    size_type get_krylov_dim() const { return parameters_.krylov_dim; }

    /**
     * Sets the Krylov dimension
     *
     * @param other  the new Krylov dimension
     */
    void set_krylov_dim(size_type other) { parameters_.krylov_dim = other; }

    /**
     * Returns the storage precision used internally.
     *
     * @return the storage precision used internally
     */
    cb_gmres::storage_precision get_storage_precision() const
    {
        return parameters_.storage_precision;
    }

    class Factory;

    struct parameters_type
        : enable_preconditioned_iterative_solver_factory_parameters<
              parameters_type, Factory> {
        /**
         * Determines which storage type is used.
         */
        cb_gmres::storage_precision GKO_FACTORY_PARAMETER_SCALAR(
            storage_precision, cb_gmres::storage_precision::reduce1);

        /**
         * Krylov dimension factory.
         */
        size_type GKO_FACTORY_PARAMETER_SCALAR(krylov_dim, 100u);
    };

    GKO_ENABLE_LIN_OP_FACTORY(CbGmres, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_dense_impl(const matrix::Dense<ValueType>* b,
                          matrix::Dense<ValueType>* x) const;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

    explicit CbGmres(std::shared_ptr<const Executor> exec)
        : EnableLinOp<CbGmres>(std::move(exec))
    {}

    explicit CbGmres(const Factory* factory,
                     std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<CbGmres>(factory->get_executor(),
                               transpose(system_matrix->get_size())),
          EnablePreconditionedIterativeSolver<ValueType, CbGmres<ValueType>>{
              std::move(system_matrix), factory->get_parameters()},
          parameters_{factory->get_parameters()}
    {}
};


}  // namespace solver
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_SOLVER_CB_GMRES_HPP_
