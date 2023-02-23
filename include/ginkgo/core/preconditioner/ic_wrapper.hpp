/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#ifndef GKO_PUBLIC_CORE_PRECONDITIONER_IC_WRAPPER_HPP_
#define GKO_PUBLIC_CORE_PRECONDITIONER_IC_WRAPPER_HPP_


#include <memory>
#include <type_traits>


#include <ginkgo/core/base/abstract_factory.hpp>
#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/std_extensions.hpp>
#include <ginkgo/core/factorization/par_ic.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/solver_traits.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


namespace gko {
namespace preconditioner {


/**
 * The Incomplete Cholesky (IC) preconditioner solves the equation $LL^H*x = b$
 * for a given lower triangular matrix L and the right hand side b (can contain
 * multiple right hand sides).
 *
 * It allows to set both the solver for L defaulting to solver::LowerTrs, which
 * is a direct triangular solvers. The solver for L^H is the
 * conjugate-transposed solver for L, ensuring that the preconditioner is
 * symmetric and positive-definite. For this L solver, a factory can be provided
 * (using `with_l_solver_factory`) to have more control over their behavior. In
 * particular, it is possible to use an iterative method for solving the
 * triangular systems. The default parameters for an iterative triangluar solver
 * are:
 * - reduction factor = 1e-4
 * - max iteration = <number of rows of the matrix given to the solver>
 * Solvers without such criteria can also be used, in which case none are set.
 *
 * An object of this class can be created with a matrix or a gko::Composition
 * containing two matrices. If created with a matrix, it is factorized before
 * creating the solver. If a gko::Composition (containing two matrices) is
 * used, the first operand will be taken as the L matrix, the second will be
 * considered the L^H matrix, which helps to avoid the otherwise necessary
 * transposition of L inside the solver. ParIc can be directly used, since it
 * orders the factors in the correct way.
 *
 * @note When providing a gko::Composition, the first matrix must be the lower
 *       matrix ($L$), and the second matrix must be its conjugate-transpose
 * ($L^H$). If they are swapped, solving might crash or return the wrong result.
 *
 * @note Do not use symmetric solvers (like CG) for the L solver since both
 *       matrices (L and L^H) are, by design, not symmetric.
 *
 * @note This class is not thread safe (even a const object is not) because it
 *       uses an internal cache to accelerate multiple (sequential) applies.
 *       Using it in parallel can lead to segmentation faults, wrong results
 *       and other unwanted behavior.
 *
 *
 * @ingroup precond
 * @ingroup LinOp
 */
class IcWrapper : public EnableLinOp<IcWrapper>, public Transposable {
    friend class EnableLinOp<IcWrapper>;
    friend class EnablePolymorphicObject<IcWrapper, LinOp>;

public:
    using transposed_type = IcWrapper;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Factory for the L solver
         */
        std::shared_ptr<LinOpFactory> GKO_FACTORY_PARAMETER_SCALAR(
            l_solver_factory, nullptr);

        /**
         * Factory for the factorization
         */
        std::shared_ptr<LinOpFactory> GKO_FACTORY_PARAMETER_SCALAR(
            factorization_factory, nullptr);
    };

    GKO_ENABLE_LIN_OP_FACTORY(IcWrapper, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

    /**
     * Returns the solver which is used for the provided L matrix.
     *
     * @returns  the solver which is used for the provided L matrix
     */
    std::shared_ptr<const LinOp> get_l_solver() const { return l_solver_; }

    /**
     * Returns the solver which is used for the L^H matrix.
     *
     * @returns  the solver which is used for the L^H matrix
     */
    std::shared_ptr<const LinOp> get_lh_solver() const { return lh_solver_; }

    std::unique_ptr<LinOp> transpose() const override
    {
        std::unique_ptr<transposed_type> transposed{
            new transposed_type{this->get_executor()}};
        transposed->set_size(gko::transpose(this->get_size()));
        transposed->l_solver_ =
            share(as<Transposable>(this->get_lh_solver())->transpose());
        transposed->lh_solver_ =
            share(as<Transposable>(this->get_l_solver())->transpose());

        return std::move(transposed);
    }

    std::unique_ptr<LinOp> conj_transpose() const override
    {
        std::unique_ptr<transposed_type> transposed{
            new transposed_type{this->get_executor()}};
        transposed->set_size(gko::transpose(this->get_size()));
        transposed->l_solver_ =
            share(as<Transposable>(this->get_lh_solver())->conj_transpose());
        transposed->lh_solver_ =
            share(as<Transposable>(this->get_l_solver())->conj_transpose());

        return std::move(transposed);
    }

    /**
     * Generates a default solver of type SolverType.
     *
     * Also checks whether SolverType can be assigned a criteria, and if it
     * can, it is assigned default values which should be well suited for a
     * preconditioner.
     */
    template <typename SolverType>
    static std::enable_if_t<solver::has_with_criteria<SolverType>::value,
                            std::unique_ptr<typename SolverType::Factory>>
    generate_default_solver(const std::shared_ptr<const Executor>& exec)
    {
        using value_type = typename SolverType::value_type;
        constexpr gko::remove_complex<value_type> default_reduce_residual{1e-4};

        return SolverType::build()
            .with_criteria(gko::stop::ResidualNorm<value_type>::build()
                               .with_reduction_factor(default_reduce_residual)
                               .on(exec))
            .on(exec);
    }

    /**
     * @copydoc generate_default_solver
     */
    template <typename SolverType>
    static std::enable_if_t<!solver::has_with_criteria<SolverType>::value,
                            std::unique_ptr<typename SolverType::Factory>>
    generate_default_solver(const std::shared_ptr<const Executor>& exec)
    {
        return SolverType::build().on(exec);
    }

protected:
    void apply_impl(const LinOp* b, LinOp* x) const override
    {
        this->set_cache_to(b);
        l_solver_->apply(b, cache_.intermediate.get());
        if (lh_solver_->apply_uses_initial_guess()) {
            x->copy_from(cache_.intermediate.get());
        }
        lh_solver_->apply(cache_.intermediate.get(), x);
    }

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override
    {
        this->set_cache_to(b);
        l_solver_->apply(b, cache_.intermediate.get());
        lh_solver_->apply(alpha, cache_.intermediate.get(), beta, x);
    }

    explicit IcWrapper(std::shared_ptr<const Executor> exec)
        : EnableLinOp<IcWrapper>(std::move(exec))
    {}

    explicit IcWrapper(const Factory* factory,
                       std::shared_ptr<const LinOp> lin_op)
        : EnableLinOp<IcWrapper>(factory->get_executor(), lin_op->get_size()),
          parameters_{factory->get_parameters()}
    {
        // TODO: Composition should not need
        auto comp = std::dynamic_pointer_cast<const CompositionBase>(lin_op);
        std::shared_ptr<const LinOp> l_factor;

        // build factorization if we weren't passed a composition
        if (!comp) {
            auto exec = lin_op->get_executor();
            if (!parameters_.factorization_factory) {
                GKO_NOT_SUPPORTED(lin_op);
            }
            auto fact = std::shared_ptr<const LinOp>(
                parameters_.factorization_factory->generate(lin_op));
            // ensure that the result is a composition
            comp = std::dynamic_pointer_cast<const CompositionBase>(fact);
            if (!comp) {
                GKO_NOT_SUPPORTED(comp);
            }
        }
        // comp must contain one or two factors
        if (comp->get_operators().size() > 2 || comp->get_operators().empty()) {
            GKO_NOT_SUPPORTED(comp);
        }
        l_factor = comp->get_operators()[0];
        GKO_ASSERT_IS_SQUARE_MATRIX(l_factor);

        auto exec = this->get_executor();
        if (comp->get_operators().size() == 2) {
            // If comp contains both factors: use the transposed factor to avoid
            // transposing twice
            auto lh_factor = comp->get_operators()[1];
            GKO_ASSERT_EQUAL_DIMENSIONS(l_factor, lh_factor);
        }

        if (!parameters_.l_solver_factory) {
            GKO_NOT_SUPPORTED(parameters_.l_solver_factory);
        } else {
            l_solver_ = parameters_.l_solver_factory->generate(l_factor);
            lh_solver_ = as<Transposable>(l_solver_)->conj_transpose();
        }
    }

    /**
     * Prepares the intermediate vector for the solve by creating it and
     * by copying the values from `b`, so `b` acts as the initial guess.
     *
     * @param b  Right hand side of the first solve. Also acts as the
     * initial guess, meaning the intermediate value will be a copy of b
     */
    void set_cache_to(const LinOp* b) const
    {
        if (cache_.intermediate == nullptr) {
            // TODO: create a create_empty from b
            cache_.intermediate = b->clone(this->get_executor());
        }
        // Use b as the initial guess for the first triangular solve
        cache_.intermediate->copy_from(b);
    }

private:
    std::shared_ptr<const LinOp> l_solver_{};
    std::shared_ptr<const LinOp> lh_solver_{};
    /**
     * Manages a vector as a cache, so there is no need to allocate one
     * every time an intermediate vector is required. Copying an instance
     * will only yield an empty object since copying the cached vector would
     * not make sense.
     *
     * @internal  The struct is present so the whole class can be copyable
     *            (could also be done with writing `operator=` and copy
     *            constructor of the enclosing class by hand)
     */
    mutable struct cache_struct {
        cache_struct() = default;
        ~cache_struct() = default;
        cache_struct(const cache_struct&) {}
        cache_struct(cache_struct&&) {}
        cache_struct& operator=(const cache_struct&) { return *this; }
        cache_struct& operator=(cache_struct&&) { return *this; }
        std::unique_ptr<LinOp> intermediate{};
    } cache_;
};


}  // namespace preconditioner
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_PRECONDITIONER_IC_WRAPPER_HPP_
