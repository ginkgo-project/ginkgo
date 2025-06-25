// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_SOLVER_PIPE_CG_HPP_
#define GKO_PUBLIC_CORE_SOLVER_PIPE_CG_HPP_


#include <vector>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/config/type_descriptor.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/solver/solver_base.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/criterion.hpp>


namespace gko {
namespace solver {


/**
 * PIPE_CG or the pipelined conjugate gradient method is an iterative type
 * Krylov subspace method which is suitable for symmetric positive definite
 * methods. It improves upon the CG method by allowing computation of inner
 * products and norms to be overlapped with operator and preconditioner
 * application. The pipelined method scales up to the 10^6 nodes of the assumed
 * exascale machine, while its standard counterpart level off about one order of
 * magnitude earlier, as suggested in the referenced paper (see below).
 *
 * Possible issues:
 * 1. Numerical instability: Due to the rearrangement of the operations, the
 * method is known to be less stable than standard PCG.
 * 2. The method is suitable for cases where a large number of iterations need
 * to be performed and when the solver is distributed over a large number of
 * distributed nodes. The advantage of lesser number of reductions that need to
 * be performed comes at the cost of increased vector operations, and the cost
 * of increased storage of vectors.
 * 3. As the CG itself, this method performs very well for symmetric positive
 * definite matrices but it is in general not suitable for general matrices.
 *
 * The implementation in Ginkgo is based on the following paper:
 * Pipelined, Flexible Krylov Subspace Methods, P. Sanan et. al, SISC, 2016,
 * doi: 10.1137/15M1049130
 *
 * @tparam ValueType  precision of matrix elements
 *
 * @ingroup solvers
 * @ingroup LinOp
 */
template <typename ValueType = default_precision>
class PipeCg
    : public EnableLinOp<PipeCg<ValueType>>,
      public EnablePreconditionedIterativeSolver<ValueType, PipeCg<ValueType>>,
      public Transposable {
    friend class EnableLinOp<PipeCg>;
    friend class EnablePolymorphicObject<PipeCg, LinOp>;

public:
    using value_type = ValueType;
    using transposed_type = PipeCg<ValueType>;

    std::unique_ptr<LinOp> transpose() const override;

    std::unique_ptr<LinOp> conj_transpose() const override;

    /**
     * Return true as iterative solvers use the data in x as an initial guess.
     *
     * @return true as iterative solvers use the data in x as an initial guess.
     */
    bool apply_uses_initial_guess() const override { return true; }

    class Factory;

    struct parameters_type
        : enable_preconditioned_iterative_solver_factory_parameters<
              parameters_type, Factory> {};

    GKO_ENABLE_LIN_OP_FACTORY(PipeCg, parameters, Factory);
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
     *                      default uses the value type of this class.
     *
     * @return parameters
     */
    static parameters_type parse(const config::pnode& config,
                                 const config::registry& context,
                                 const config::type_descriptor& td_for_child =
                                     config::make_type_descriptor<ValueType>());

protected:
    void apply_impl(const LinOp* b, LinOp* x) const override;

    template <typename VectorType>
    void apply_dense_impl(const VectorType* b, VectorType* x) const;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

    explicit PipeCg(std::shared_ptr<const Executor> exec)
        : EnableLinOp<PipeCg>(std::move(exec))
    {}

    explicit PipeCg(const Factory* factory,
                    std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<PipeCg>(factory->get_executor(),
                              gko::transpose(system_matrix->get_size())),
          EnablePreconditionedIterativeSolver<ValueType, PipeCg<ValueType>>{
              std::move(system_matrix), factory->get_parameters()},
          parameters_{factory->get_parameters()}
    {}
};


template <typename ValueType>
struct workspace_traits<PipeCg<ValueType>> {
    using Solver = PipeCg<ValueType>;
    // number of vectors used by this workspace
    static int num_vectors(const Solver&);
    // number of arrays used by this workspace
    static int num_arrays(const Solver&);
    // array containing the num_vectors names for the workspace vectors
    static std::vector<std::string> op_names(const Solver&);
    // array containing the num_arrays names for the workspace vectors
    static std::vector<std::string> array_names(const Solver&);
    // array containing all varying scalar vectors (independent of problem size)
    static std::vector<int> scalars(const Solver&);
    // array containing all varying vectors (dependent on problem size)
    static std::vector<int> vectors(const Solver&);

    // residual vector
    constexpr static int r = 0;
    // preconditioned residual vector
    constexpr static int z = 1;
    // p vector
    constexpr static int p = 2;
    // w vector
    constexpr static int w = 3;
    // m vector
    constexpr static int m = 4;
    // n vector
    constexpr static int n = 5;
    // q vector
    constexpr static int q = 6;
    // f vector
    constexpr static int f = 7;
    // g vector
    constexpr static int g = 8;
    // beta scalar
    constexpr static int beta = 9;
    // delta scalar
    constexpr static int delta = 10;
    // previous rho scalar
    constexpr static int prev_rho = 11;
    // current rho scalar
    constexpr static int rho = 12;
    // constant 1.0 scalar
    constexpr static int one = 13;
    // constant -1.0 scalar
    constexpr static int minus_one = 14;

    // stopping status array
    constexpr static int stop = 0;
    // reduction tmp array
    constexpr static int tmp = 1;
};


}  // namespace solver
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_SOLVER_PIPE_CG_HPP_
