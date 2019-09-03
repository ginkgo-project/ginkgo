/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#ifndef GKO_CORE_PRECONDITIONER_ILU_HPP_
#define GKO_CORE_PRECONDITIONER_ILU_HPP_


#include <memory>
#include <type_traits>


#include <ginkgo/core/base/abstract_factory.hpp>
#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/std_extensions.hpp>
#include <ginkgo/core/matrix/dense.hpp>
// #include <ginkgo/core/solver/lower_trs.hpp>
#include <ginkgo/core/solver/bicgstab.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm_reduction.hpp>


namespace gko {
namespace preconditioner {


// TODO: Replace Bicgstab with Upper/LowerTrs when available!
/**
 * The incomplete LU (ILU) preconditioner solves the equation LUx = b for a
 * given lower triangular matrix L, an upper triangular matrix U and the right
 * hand side b (can contain multiple right hand sides).
 *
 * It allows to set both the solver for L and the solver for U independently,
 * while providing the defaults solver::LowerTrs and solver::UpperTrs.
 *
 * @note This class is not thread safe (even a const object is not) because it
 *       uses an internal cache to accelerate multiple (sequential) applies
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam LSolverType  type of the solver used for the L matrix.
 *                      Defaults to solver::LowerTrs
 * @tparam USolverType  type of the solver used for the U matrix
 *                      Defaults to solver::UpperTrs
 * @tparam ReverseApply  default behavior (ReverseApply = false) is first to
 *                       solve with L (Ly = b) and then with U (Ux = y).
 *                       When set to true, it will solve first with U, and then
 *                       with L.
 *
 * @ingroup precond
 * @ingroup LinOp
 */
template <typename ValueType,
          typename LSolverType = solver::Bicgstab<ValueType>,
          typename USolverType = solver::Bicgstab<ValueType>,
          bool ReverseApply = false>
class Ilu : public EnableLinOp<
                Ilu<ValueType, LSolverType, USolverType, ReverseApply>> {
    friend class EnableLinOp<Ilu>;
    friend class EnablePolymorphicObject<Ilu, LinOp>;

public:
    using value_type = ValueType;
    using l_solver_type = LSolverType;
    using u_solver_type = USolverType;
    static constexpr bool performs_reverse_apply = ReverseApply;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Factory for the L solver
         */
        std::shared_ptr<typename LSolverType::Factory> GKO_FACTORY_PARAMETER(
            l_solver_factory, nullptr);

        /**
         * Factory for the U solver
         */
        std::shared_ptr<typename USolverType::Factory> GKO_FACTORY_PARAMETER(
            u_solver_factory, nullptr);
    };

protected:
    /**
     * Manages the `generate` arguments for the parent class to allow multiple
     * versions to initialize both L and U. Three constructors are provided:
     * - a Composition, containing the L matrix as the first operand, and the
     *   U matrix as the second and last
     * - one ParIlu, containing both L and U (since it is equal to the first)
     * - both L and U matrix as separate parameters
     */
    struct LuArgs {
        LuArgs(std::shared_ptr<const LinOp> composition)
        {
            auto comp_cast =
                as<const Composition<ValueType>>(composition.get());
            if (comp_cast->get_operators().size() != 2) {
                throw GKO_NOT_SUPPORTED(comp_cast);
            }
            l_factor = comp_cast->get_operators()[0];
            u_factor = comp_cast->get_operators()[1];
        }

        LuArgs(std::shared_ptr<const LinOp> l_fac,
               std::shared_ptr<const LinOp> u_fac)
            : l_factor{std::move(l_fac)}, u_factor{std::move(u_fac)}
        {}

        /**
         * Returns the size that the solver using L and U would return
         *
         * @param inverse_apply  determines if the solver solves for U first
         *                       and then for L (inverse_apply = true), or
         *                       first with L, then with U
         *                       (inverse_apply = false)
         *
         * @returns the size that the solver using L and U would return
         */
        dim<2> get_solver_size(bool inverse_apply = false) const
        {
            return (inverse_apply) ? dim<2>{l_factor->get_size()[0],
                                            u_factor->get_size()[1]}
                                   : dim<2>{u_factor->get_size()[0],
                                            l_factor->get_size()[1]};
        }

        std::shared_ptr<const LinOp> l_factor;
        std::shared_ptr<const LinOp> u_factor;
    };

    // The following code is used to replace the default
    // `GKO_ENABLE_LIN_OP_FACTORY` macro
    using PolymorphicBaseFactory = AbstractFactory<LinOp, LuArgs>;
    template <typename ConcreteFactory>
    using EnableIluFactory =
        EnableDefaultFactory<ConcreteFactory, Ilu, parameters_type,
                             PolymorphicBaseFactory>;

public:
    /**
     * Returns the parameters used to build the initial object.
     *
     * @returns the parameters used to build the initial object.
     */
    const parameters_type &get_parameters() const { return parameters_; }

    /**
     * Used to replace the `GKO_ENABLE_LIN_OP_FACTORY` macro to allow for
     * more variety in arguments for the `generate` function.
     */
    class Factory : public EnableIluFactory<Factory> {
        friend class ::gko::EnablePolymorphicObject<Factory,
                                                    PolymorphicBaseFactory>;
        friend class ::gko::enable_parameters_type<parameters_type, Factory>;
        using EnableIluFactory<Factory>::EnableIluFactory;
    };

    friend EnableIluFactory<Factory>;

private:
    parameters_type parameters_;

    // End of the code to replace the `GKO_ENABLE_LIN_OP_FACTORY` macro
public:
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    void apply_impl(const LinOp *b, LinOp *x) const override
    {
        ensure_cache_support_for(b);
        if (!ReverseApply) {
            l_solver_->apply(b, cache_.intermediate.get());
            u_solver_->apply(cache_.intermediate.get(), x);
        } else {
            u_solver_->apply(b, cache_.intermediate.get());
            l_solver_->apply(cache_.intermediate.get(), x);
        }
    }

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override
    {
        ensure_cache_support_for(b);
        if (!ReverseApply) {
            l_solver_->apply(b, cache_.intermediate.get());
            u_solver_->apply(alpha, cache_.intermediate.get(), beta, x);
        } else {
            u_solver_->apply(b, cache_.intermediate.get());
            l_solver_->apply(alpha, cache_.intermediate.get(), beta, x);
        }
    }

    explicit Ilu(std::shared_ptr<const Executor> exec)
        : EnableLinOp<Ilu>(std::move(exec))
    {}

    explicit Ilu(const Factory *factory, LuArgs lu_args)
        : EnableLinOp<Ilu>(factory->get_executor(),
                           lu_args.get_solver_size(ReverseApply)),
          parameters_{factory->get_parameters()},
          l_factor_{std::move(lu_args.l_factor)},
          u_factor_{std::move(lu_args.u_factor)}
    {
        GKO_ASSERT_EQUAL_ROWS(l_factor_, u_factor_);
        GKO_ASSERT_EQUAL_COLS(l_factor_, u_factor_);

        constexpr ValueType default_reduce_precision{1e-4};
        const unsigned int default_max_iters{
            static_cast<unsigned int>(this->get_size()[0])};
        auto exec = this->get_executor();

        // If no factories are provided, generate default ones
        if (!parameters_.l_solver_factory) {
            l_solver_ = generate_default_solver<LSolverType>(exec, l_factor_);
        } else {
            l_solver_ = parameters_.l_solver_factory->generate(l_factor_);
        }
        if (!parameters_.u_solver_factory) {
            u_solver_ = generate_default_solver<USolverType>(exec, u_factor_);
        } else {
            u_solver_ = parameters_.u_solver_factory->generate(u_factor_);
        }
    }

    void ensure_cache_support_for(const LinOp *b) const
    {
        dim<2> expected_size =
            ReverseApply ? dim<2>{u_solver_->get_size()[0], b->get_size()[1]}
                         : dim<2>{l_solver_->get_size()[0], b->get_size()[1]};
        if (cache_.intermediate == nullptr ||
            cache_.intermediate->get_size() != expected_size) {
            cache_.intermediate = matrix::Dense<ValueType>::create(
                this->get_executor(), expected_size);
        }
    }

    /**
     * @internal  Looks at the build() method to determine the type of the
     *            factory.
     */
    template <typename T>
    using factory_type_t = decltype(T::build());

    // Parameter type of function `with_criteria`.
    using with_criteria_param_type =
        std::shared_ptr<const stop::CriterionFactory>;

    /**
     * Helper structure to test if the Factory of SolverType has a function
     * `with_criteria`.
     *
     * Contains a constexpr boolean `value`, which is true if the Factory class
     * of SolverType has a `with_criteria`, and false otherwise.
     *
     * @tparam SolverType   Solver to test if its factory has a with_criteria
     *                      function.
     *
     */
    template <typename SolverType, typename = void>
    struct has_with_criteria : std::false_type {};

    /**
     * @copydoc has_with_criteria
     *
     * @internal  The second template parameter (which uses SFINAE) must match
     *            the default value of the general case in order to be accepted
     *            as a specialization.
     */
    template <typename SolverType>
    struct has_with_criteria<
        SolverType,
        xstd::void_t<decltype(std::declval<factory_type_t<SolverType>>()
                                  .with_criteria(with_criteria_param_type()))>>
        : std::true_type {};


    /**
     * Generates a default solver of type SolverType.
     *
     * Also checks wheather SolverType can be assigned a criteria, and if it
     * can, it is assigned default values which should be well suited for a
     * preconditioner.
     */
    template <typename SolverType>
    static xstd::enable_if_t<has_with_criteria<SolverType>::value,
                             std::unique_ptr<SolverType>>
    generate_default_solver(const std::shared_ptr<const Executor> &exec,
                            const std::shared_ptr<const LinOp> &mtx)
    {
        constexpr ValueType default_reduce_precision{1e-4};
        const unsigned int default_max_iters{
            static_cast<unsigned int>(mtx->get_size()[0])};

        return SolverType::build()
            .with_criteria(gko::stop::Iteration::build()
                               .with_max_iters(default_max_iters)
                               .on(exec),
                           gko::stop::ResidualNormReduction<>::build()
                               .with_reduction_factor(default_reduce_precision)
                               .on(exec))
            .on(exec)
            ->generate(mtx);
    }

    /**
     * @copydoc generate_default_solver
     */
    template <typename SolverType>
    static xstd::enable_if_t<!has_with_criteria<SolverType>::value,
                             std::unique_ptr<SolverType>>
    generate_default_solver(const std::shared_ptr<const Executor> &exec,
                            const std::shared_ptr<const LinOp> &mtx)
    {
        return SolverType::build().on(exec)->generate(mtx);
    }

private:
    std::shared_ptr<const LinOp> l_factor_{};
    std::shared_ptr<const LinOp> u_factor_{};
    /**
     * Manages a vector as a cache, so there is no need to allocate one every
     * time an intermediate vector is required.
     * Copying an instance will only yield an empty object since copying the
     * cached vector would not make sense.
     *
     * @internal  The struct is present so the whole class can be copyable
     *            (could also be done with writing `operator=` and copy
     *            constructor of the enclosing class by hand)
     */
    mutable struct cache_struct {
        cache_struct() = default;
        ~cache_struct() = default;
        cache_struct(const cache_struct &other) {}
        cache_struct &operator=(const cache_struct &) { return *this; }
        std::unique_ptr<LinOp> intermediate{};
    } cache_;
    std::shared_ptr<const LSolverType> l_solver_{};
    std::shared_ptr<const USolverType> u_solver_{};
};


}  // namespace preconditioner
}  // namespace gko


#endif  // GKO_CORE_PRECONDITIONER_ILU_HPP_
