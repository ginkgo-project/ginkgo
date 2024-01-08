// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_SOLVER_BICGSTAB_HPP_
#define GKO_PUBLIC_CORE_SOLVER_BICGSTAB_HPP_


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
/**
 * @brief The ginkgo Solve namespace.
 *
 * @ingroup solvers
 */
namespace solver {


/**
 * BiCGSTAB or the Bi-Conjugate Gradient-Stabilized is a Krylov subspace solver.
 *
 * Being a generic solver, it is capable of solving general matrices, including
 * non-s.p.d matrices. Though, the memory and the computational requirement of
 * the BiCGSTAB solver are higher than of its s.p.d solver counterpart, it has
 * the capability to solve generic systems. It was developed by stabilizing the
 * BiCG method.
 *
 * @tparam ValueType precision of the elements of the system matrix.
 *
 * @ingroup bicgstab
 * @ingroup solvers
 * @ingroup LinOp
 */
template <typename ValueType = default_precision>
class Bicgstab
    : public EnableLinOp<Bicgstab<ValueType>>,
      public EnablePreconditionedIterativeSolver<ValueType,
                                                 Bicgstab<ValueType>>,
      public Transposable {
    friend class EnableLinOp<Bicgstab>;
    friend class EnablePolymorphicObject<Bicgstab, LinOp>;

public:
    using value_type = ValueType;
    using transposed_type = Bicgstab<ValueType>;

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

    GKO_ENABLE_LIN_OP_FACTORY(Bicgstab, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    void apply_impl(const LinOp* b, LinOp* x) const override;

    template <typename VectorType>
    void apply_dense_impl(const VectorType* b, VectorType* x) const;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

    explicit Bicgstab(std::shared_ptr<const Executor> exec)
        : EnableLinOp<Bicgstab>(std::move(exec))
    {}

    explicit Bicgstab(const Factory* factory,
                      std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<Bicgstab>(factory->get_executor(),
                                gko::transpose(system_matrix->get_size())),
          EnablePreconditionedIterativeSolver<ValueType, Bicgstab<ValueType>>{
              std::move(system_matrix), factory->get_parameters()},
          parameters_{factory->get_parameters()}
    {}
};


template <typename ValueType>
struct workspace_traits<Bicgstab<ValueType>> {
    using Solver = Bicgstab<ValueType>;
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
    // y vector
    constexpr static int y = 2;
    // v vector
    constexpr static int v = 3;
    // s vector
    constexpr static int s = 4;
    // t vector
    constexpr static int t = 5;
    // p vector
    constexpr static int p = 6;
    // rr vector
    constexpr static int rr = 7;
    // alpha scalar
    constexpr static int alpha = 8;
    // beta scalar
    constexpr static int beta = 9;
    // gamma scalar
    constexpr static int gamma = 10;
    // previous rho scalar
    constexpr static int prev_rho = 11;
    // current rho scalar
    constexpr static int rho = 12;
    // omega scalar
    constexpr static int omega = 13;
    // constant 1.0 scalar
    constexpr static int one = 14;
    // constant -1.0 scalar
    constexpr static int minus_one = 15;

    // stopping status array
    constexpr static int stop = 0;
    // reduction tmp array
    constexpr static int tmp = 1;
};


}  // namespace solver
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_SOLVER_BICGSTAB_HPP_
