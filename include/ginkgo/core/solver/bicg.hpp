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

#ifndef GKO_PUBLIC_CORE_SOLVER_BICG_HPP_
#define GKO_PUBLIC_CORE_SOLVER_BICG_HPP_


#include <vector>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/half.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/solver/solver_base.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/criterion.hpp>


namespace gko {
namespace solver {


/**
 * BICG or the Biconjugate gradient method is a Krylov subspace solver.
 *
 * Being a generic solver, it is capable of solving general matrices, including
 * non-s.p.d matrices. Though, the memory and the computational requirement of
 * the BiCG solver are higher than of its s.p.d solver counterpart, it has
 * the capability to solve generic systems.
 *
 * BiCG is based on the bi-Lanczos tridiagonalization method and in exact
 * arithmetic should terminate in at most N iterations (2N MV's, with A and
 * A^H). It forms the basis of many of the cheaper methods such as BiCGSTAB and
 * CGS.
 *
 * Reference: R.Fletcher, Conjugate gradient methods for indefinite systems,
 * doi: 10.1007/BFb0080116
 *
 * @tparam ValueType  precision of matrix elements
 *
 * @ingroup solvers
 * @ingroup LinOp
 */
template <typename ValueType = default_precision>
class Bicg
    : public EnableLinOp<Bicg<ValueType>>,
      public EnablePreconditionedIterativeSolver<ValueType, Bicg<ValueType>>,
      public Transposable {
    friend class EnableLinOp<Bicg>;
    friend class EnablePolymorphicObject<Bicg, LinOp>;

public:
    using value_type = ValueType;
    using transposed_type = Bicg<ValueType>;

    std::unique_ptr<LinOp> transpose() const override;

    std::unique_ptr<LinOp> conj_transpose() const override;

    /**
     * Return true as iterative solvers use the data in x as an initial guess.
     *
     * @return true as iterative solvers use the data in x as an initial guess.
     */
    bool apply_uses_initial_guess() const override { return true; }

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
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
    };
    GKO_ENABLE_LIN_OP_FACTORY(Bicg, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_dense_impl(const matrix::Dense<ValueType>* b,
                          matrix::Dense<ValueType>* x) const;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

    explicit Bicg(std::shared_ptr<const Executor> exec)
        : EnableLinOp<Bicg>(std::move(exec))
    {}

    explicit Bicg(const Factory* factory,
                  std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<Bicg>(factory->get_executor(),
                            gko::transpose(system_matrix->get_size())),
          EnablePreconditionedIterativeSolver<ValueType, Bicg<ValueType>>{
              std::move(system_matrix), factory->get_parameters()},
          parameters_{factory->get_parameters()}
    {}
};


template <typename ValueType>
struct workspace_traits<Bicg<ValueType>> {
    using Solver = Bicg<ValueType>;
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
    // q vector
    constexpr static int q = 3;
    // "transposed" residual vector
    constexpr static int r2 = 4;
    // "transposed" preconditioned residual vector
    constexpr static int z2 = 5;
    // "transposed" p vector
    constexpr static int p2 = 6;
    // "transposed" q vector
    constexpr static int q2 = 7;
    // alpha scalar
    constexpr static int alpha = 8;
    // beta scalar
    constexpr static int beta = 9;
    // previous rho scalar
    constexpr static int prev_rho = 10;
    // current rho scalar
    constexpr static int rho = 11;
    // constant 1.0 scalar
    constexpr static int one = 12;
    // constant -1.0 scalar
    constexpr static int minus_one = 13;

    // stopping status array
    constexpr static int stop = 0;
    // reduction tmp array
    constexpr static int tmp = 1;
};


}  // namespace solver
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_SOLVER_BICG_HPP_
