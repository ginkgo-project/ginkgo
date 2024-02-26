// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_PRECONDITIONER_IC_HPP_
#define GKO_PUBLIC_CORE_PRECONDITIONER_IC_HPP_


#include <memory>
#include <type_traits>


#include <ginkgo/core/base/abstract_factory.hpp>
#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/std_extensions.hpp>
#include <ginkgo/core/factorization/par_ic.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/solver_traits.hpp>
#include <ginkgo/core/solver/triangular.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
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
 * (using `with_l_solver`) to have more control over their behavior. In
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
 * @tparam LSolverType  type of the solver used for the L matrix.
 *                      Defaults to solver::LowerTrs
 * @tparam IndexType  type of the indices when ParIc is used to generate
 *                    the L and L^H factors. Irrelevant otherwise.
 *
 * @ingroup precond
 * @ingroup LinOp
 */
template <typename LSolverType = solver::LowerTrs<>, typename IndexType = int32>
class Ic : public EnableLinOp<Ic<LSolverType, IndexType>>, public Transposable {
    friend class EnableLinOp<Ic>;
    friend class EnablePolymorphicObject<Ic, LinOp>;

public:
    static_assert(
        std::is_same<typename LSolverType::transposed_type::transposed_type,
                     LSolverType>::value,
        "LSolverType::transposed_type must be symmetric");
    using value_type = typename LSolverType::value_type;
    using l_solver_type = LSolverType;
    using lh_solver_type = typename LSolverType::transposed_type;
    using index_type = IndexType;
    using transposed_type = Ic<LSolverType, IndexType>;

    class Factory;

    struct parameters_type
        : public enable_parameters_type<parameters_type, Factory> {
        /**
         * Factory for the L solver
         */
        std::shared_ptr<const typename l_solver_type::Factory>
            l_solver_factory{};

        /**
         * Factory for the factorization
         */
        std::shared_ptr<const LinOpFactory> factorization_factory{};

        GKO_DEPRECATED("use with_l_solver instead")
        parameters_type& with_l_solver_factory(
            deferred_factory_parameter<const typename l_solver_type::Factory>
                solver)
        {
            return with_l_solver(std::move(solver));
        }

        parameters_type& with_l_solver(
            deferred_factory_parameter<const typename l_solver_type::Factory>
                solver)
        {
            this->l_solver_generator = std::move(solver);
            this->deferred_factories["l_solver"] = [](const auto& exec,
                                                      auto& params) {
                if (!params.l_solver_generator.is_empty()) {
                    params.l_solver_factory =
                        params.l_solver_generator.on(exec);
                }
            };
            return *this;
        }

        GKO_DEPRECATED("use with_factorization instead")
        parameters_type& with_factorization_factory(
            deferred_factory_parameter<const LinOpFactory> factorization)
        {
            return with_factorization(std::move(factorization));
        }

        parameters_type& with_factorization(
            deferred_factory_parameter<const LinOpFactory> factorization)
        {
            this->factorization_generator = std::move(factorization);
            this->deferred_factories["factorization"] = [](const auto& exec,
                                                           auto& params) {
                if (!params.factorization_generator.is_empty()) {
                    params.factorization_factory =
                        params.factorization_generator.on(exec);
                }
            };
            return *this;
        }

    private:
        deferred_factory_parameter<const typename l_solver_type::Factory>
            l_solver_generator;

        deferred_factory_parameter<const LinOpFactory> factorization_generator;
    };

    GKO_ENABLE_LIN_OP_FACTORY(Ic, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

    /**
     * Returns the solver which is used for the provided L matrix.
     *
     * @returns  the solver which is used for the provided L matrix
     */
    std::shared_ptr<const l_solver_type> get_l_solver() const
    {
        return l_solver_;
    }

    /**
     * Returns the solver which is used for the L^H matrix.
     *
     * @returns  the solver which is used for the L^H matrix
     */
    std::shared_ptr<const lh_solver_type> get_lh_solver() const
    {
        return lh_solver_;
    }

    std::unique_ptr<LinOp> transpose() const override
    {
        std::unique_ptr<transposed_type> transposed{
            new transposed_type{this->get_executor()}};
        transposed->set_size(gko::transpose(this->get_size()));
        transposed->l_solver_ =
            share(as<typename lh_solver_type::transposed_type>(
                this->get_lh_solver()->transpose()));
        transposed->lh_solver_ =
            share(as<typename l_solver_type::transposed_type>(
                this->get_l_solver()->transpose()));

        return std::move(transposed);
    }

    std::unique_ptr<LinOp> conj_transpose() const override
    {
        std::unique_ptr<transposed_type> transposed{
            new transposed_type{this->get_executor()}};
        transposed->set_size(gko::transpose(this->get_size()));
        transposed->l_solver_ =
            share(as<typename lh_solver_type::transposed_type>(
                this->get_lh_solver()->conj_transpose()));
        transposed->lh_solver_ =
            share(as<typename l_solver_type::transposed_type>(
                this->get_l_solver()->conj_transpose()));

        return std::move(transposed);
    }

    /**
     * Copy-assigns an IC preconditioner. Preserves the executor,
     * shallow-copies the solvers and parameters. Creates a clone of the solvers
     * if they are on the wrong executor.
     */
    Ic& operator=(const Ic& other)
    {
        if (&other != this) {
            EnableLinOp<Ic>::operator=(other);
            auto exec = this->get_executor();
            l_solver_ = other.l_solver_;
            lh_solver_ = other.lh_solver_;
            parameters_ = other.parameters_;
            if (other.get_executor() != exec) {
                l_solver_ = gko::clone(exec, l_solver_);
                lh_solver_ = gko::clone(exec, lh_solver_);
            }
        }
        return *this;
    }

    /**
     * Move-assigns an IC preconditioner. Preserves the executor,
     * moves the solvers and parameters. Creates a clone of the solvers
     * if they are on the wrong executor. The moved-from object is empty (0x0
     * with nullptr solvers and default parameters)
     */
    Ic& operator=(Ic&& other)
    {
        if (&other != this) {
            EnableLinOp<Ic>::operator=(other);
            auto exec = this->get_executor();
            l_solver_ = std::move(other.l_solver_);
            lh_solver_ = std::move(other.lh_solver_);
            parameters_ = std::exchange(other.parameters_, parameters_type{});
            if (other.get_executor() != exec) {
                l_solver_ = gko::clone(exec, l_solver_);
                lh_solver_ = gko::clone(exec, lh_solver_);
            }
        }
        return *this;
    }

    /**
     * Copy-constructs an IC preconditioner. Inherits the executor,
     * shallow-copies the solvers and parameters.
     */
    Ic(const Ic& other) : Ic{other.get_executor()} { *this = other; }

    /**
     * Move-constructs an IC preconditioner. Inherits the executor,
     * moves the solvers and parameters. The moved-from object is empty (0x0
     * with nullptr solvers and default parameters)
     */
    Ic(Ic&& other) : Ic{other.get_executor()} { *this = std::move(other); }

protected:
    void apply_impl(const LinOp* b, LinOp* x) const override
    {
        // take care of real-to-complex apply
        precision_dispatch_real_complex<value_type>(
            [&](auto dense_b, auto dense_x) {
                this->set_cache_to(dense_b);
                l_solver_->apply(dense_b, cache_.intermediate);
                if (lh_solver_->apply_uses_initial_guess()) {
                    dense_x->copy_from(cache_.intermediate);
                }
                lh_solver_->apply(cache_.intermediate, dense_x);
            },
            b, x);
    }

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override
    {
        precision_dispatch_real_complex<value_type>(
            [&](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
                this->set_cache_to(dense_b);
                l_solver_->apply(dense_b, cache_.intermediate);
                lh_solver_->apply(dense_alpha, cache_.intermediate, dense_beta,
                                  dense_x);
            },
            alpha, b, beta, x);
    }

    explicit Ic(std::shared_ptr<const Executor> exec)
        : EnableLinOp<Ic>(std::move(exec))
    {}

    explicit Ic(const Factory* factory, std::shared_ptr<const LinOp> lin_op)
        : EnableLinOp<Ic>(factory->get_executor(), lin_op->get_size()),
          parameters_{factory->get_parameters()}
    {
        auto comp =
            std::dynamic_pointer_cast<const Composition<value_type>>(lin_op);
        std::shared_ptr<const LinOp> l_factor;

        // build factorization if we weren't passed a composition
        if (!comp) {
            auto exec = lin_op->get_executor();
            if (!parameters_.factorization_factory) {
                parameters_.factorization_factory =
                    factorization::ParIc<value_type, index_type>::build()
                        .with_both_factors(false)
                        .on(exec);
            }
            auto fact = std::shared_ptr<const LinOp>(
                parameters_.factorization_factory->generate(lin_op));
            // ensure that the result is a composition
            comp =
                std::dynamic_pointer_cast<const Composition<value_type>>(fact);
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

        // If no factories are provided, generate default ones
        if (!parameters_.l_solver_factory) {
            l_solver_ = generate_default_solver<l_solver_type>(exec, l_factor);
            // If comp contains both factors: use the transposed factor to avoid
            // transposing twice
            if (comp->get_operators().size() == 2) {
                auto lh_factor = comp->get_operators()[1];
                GKO_ASSERT_EQUAL_DIMENSIONS(l_factor, lh_factor);
                lh_solver_ = as<lh_solver_type>(l_solver_->conj_transpose());
            } else {
                lh_solver_ = as<lh_solver_type>(l_solver_->conj_transpose());
            }
        } else {
            l_solver_ = parameters_.l_solver_factory->generate(l_factor);
            lh_solver_ = as<lh_solver_type>(l_solver_->conj_transpose());
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
            cache_.intermediate =
                matrix::Dense<value_type>::create(this->get_executor());
        }
        // Use b as the initial guess for the first triangular solve
        cache_.intermediate->copy_from(b);
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
                            std::unique_ptr<SolverType>>
    generate_default_solver(const std::shared_ptr<const Executor>& exec,
                            const std::shared_ptr<const LinOp>& mtx)
    {
        constexpr gko::remove_complex<value_type> default_reduce_residual{1e-4};
        const unsigned int default_max_iters{
            static_cast<unsigned int>(mtx->get_size()[0])};

        return SolverType::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(default_max_iters),
                gko::stop::ResidualNorm<value_type>::build()
                    .with_reduction_factor(default_reduce_residual))
            .on(exec)
            ->generate(mtx);
    }

    /**
     * @copydoc generate_default_solver
     */
    template <typename SolverType>
    static std::enable_if_t<!solver::has_with_criteria<SolverType>::value,
                            std::unique_ptr<SolverType>>
    generate_default_solver(const std::shared_ptr<const Executor>& exec,
                            const std::shared_ptr<const LinOp>& mtx)
    {
        return SolverType::build().on(exec)->generate(mtx);
    }

private:
    std::shared_ptr<const l_solver_type> l_solver_{};
    std::shared_ptr<const lh_solver_type> lh_solver_{};
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


#endif  // GKO_PUBLIC_CORE_PRECONDITIONER_IC_HPP_
