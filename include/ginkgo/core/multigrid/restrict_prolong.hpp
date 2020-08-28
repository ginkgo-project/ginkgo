/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#ifndef GKO_CORE_MULTIGRID_RESTRICT_PROLONG_HPP_
#define GKO_CORE_MULTIGRID_RESTRICT_PROLONG_HPP_


#include <functional>
#include <memory>


#include <ginkgo/core/base/abstract_factory.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace multigrid {


/**
 * The RestrictProlong class can be used to construct restrict_apply and
 * prolong_applyadd. R is the restrict operator (fine level -> coarse
 * level) and P is the prolong operator (coarse level -> fine level).
 * restrict_apply(b, x) -> x = R(b)
 * prolong_applyadd(b, x) -> x = P(b) + x
 *
 * @ingroup RestrictProlong
 * @ingroup Multigrid
 */
class RestrictProlong
    : public EnableAbstractPolymorphicObject<RestrictProlong> {
public:
    /**
     * Returns the coarse operator (matrix) which is R * matrix * P
     *
     * @return the coarse operator (matrix)
     */
    std::shared_ptr<const LinOp> get_coarse_operator() const { return coarse_; }

    /**
     * Applies a restrict operator to a vector (or a sequence of vectors).
     *
     * Performs the operation x = restrict(b).
     *
     * @param b  the input vector(s) on which the operator is applied
     * @param x  the output vector(s) where the result is stored
     *
     * @return this
     */
    RestrictProlong *restrict_apply(const LinOp *b, LinOp *x)
    {
        this->validate_restrict_parameters(b, x);
        auto exec = this->get_executor();
        this->restrict_apply_impl(make_temporary_clone(exec, b).get(),
                                  make_temporary_clone(exec, x).get());
        return this;
    }

    /**
     * @copydoc restrict_apply(const LinOp *, LinOp *)
     */
    const RestrictProlong *restrict_apply(const LinOp *b, LinOp *x) const
    {
        this->validate_restrict_parameters(b, x);
        auto exec = this->get_executor();
        this->restrict_apply_impl(make_temporary_clone(exec, b).get(),
                                  make_temporary_clone(exec, x).get());
        return this;
    }

    /**
     * Applies a prolong operator to a vector (or a sequence of vectors).
     *
     * Performs the operation x = prolong(b) + x.
     *
     * @param b  the input vector(s) on which the operator is applied
     * @param x  the output vector(s) where the result is stored
     *
     * @return this
     */
    RestrictProlong *prolong_applyadd(const LinOp *b, LinOp *x)
    {
        this->validate_prolong_parameters(b, x);
        auto exec = this->get_executor();
        this->prolong_applyadd_impl(make_temporary_clone(exec, b).get(),
                                    make_temporary_clone(exec, x).get());
        return this;
    }

    /**
     * @copydoc prolong_applyadd(const LinOp *, LinOp *)
     */
    const RestrictProlong *prolong_applyadd(const LinOp *b, LinOp *x) const
    {
        this->validate_prolong_parameters(b, x);
        auto exec = this->get_executor();
        this->prolong_applyadd_impl(make_temporary_clone(exec, b).get(),
                                    make_temporary_clone(exec, x).get());
        return this;
    }

protected:
    /**
     * Throws a DimensionMismatch exception if the parameters to
     * `restrict_apply` are of the wrong size.
     *
     * @param b  vector(s) on which the operator is applied
     * @param x  output vector(s)
     */
    void validate_restrict_parameters(const LinOp *b, LinOp *x) const
    {
        auto restrict_dim = gko::dim<2>(coarse_dim_, fine_dim_);
        GKO_ASSERT_CONFORMANT(restrict_dim, b);
        GKO_ASSERT_EQUAL_ROWS(restrict_dim, x);
        GKO_ASSERT_EQUAL_COLS(b, x);
    }

    /**
     * Throws a DimensionMismatch exception if the parameters to
     * `prolong_applyadd` are of the wrong size.
     *
     * @param b  vector(s) on which the operator is applied
     * @param x  output vector(s)
     */
    void validate_prolong_parameters(const LinOp *b, LinOp *x) const
    {
        auto prolong_dim = gko::dim<2>(fine_dim_, coarse_dim_);
        GKO_ASSERT_CONFORMANT(prolong_dim, b);
        GKO_ASSERT_EQUAL_ROWS(prolong_dim, x);
        GKO_ASSERT_EQUAL_COLS(b, x);
    }

    /**
     * Implementers of RestrictProlong should override this function instead
     * of restrict_apply(const LinOp *, LinOp *).
     *
     * Performs the operation x = restrict(b).
     *
     * @param b  the input vector(s) on which the operator is applied
     * @param x  the output vector(s) where the result is stored
     */
    virtual void restrict_apply_impl(const LinOp *b, LinOp *x) const = 0;

    /**
     * Implementers of RestrictProlong should override this function instead
     * of prolong_applyadd(const LinOp *, LinOp *).
     *
     * Performs the operation x += prolong(b).
     *
     * @param b  the input vector(s) on which the operator is applied
     * @param x  the output vector(s) where the result is stored
     */
    virtual void prolong_applyadd_impl(const LinOp *b, LinOp *x) const = 0;

    /**
     * Sets the components of RestrictProlong
     *
     * @param coarse  the coarse matrix
     * @param fine_dim  the fine_level size
     */
    void set_coarse_fine(std::shared_ptr<const LinOp> coarse,
                         size_type fine_dim)
    {
        coarse_ = coarse;
        fine_dim_ = fine_dim;
        coarse_dim_ = coarse->get_size()[0];
    }

    /**
     * Creates a RestrictProlong.
     *
     * @param exec  the executor where all the operations are performed
     */
    explicit RestrictProlong(std::shared_ptr<const Executor> exec)
        : EnableAbstractPolymorphicObject<RestrictProlong>(exec)
    {}

private:
    std::shared_ptr<const LinOp> coarse_{};
    size_type fine_dim_;
    size_type coarse_dim_;
};


/**
 * A RestrictProlongFactory represents a higher order mapping which transforms
 * one linear operator into the behavior between multigrid level.
 *
 * A RestrictProlongFactory transforms a linear operator $F$ (fine level) to
 * $R$ (restrict), $P$ (prolong), and $C=RFP$ (coarse level), which defines the
 * mapping between fine level and coarse level. R: F -> C and P: C -> F.
 *
 * @ingroup RestrictProlong
 * @ingroup Multigrid
 */
class RestrictProlongFactory
    : public AbstractFactory<RestrictProlong, std::shared_ptr<const LinOp>> {
public:
    using AbstractFactory<RestrictProlong,
                          std::shared_ptr<const LinOp>>::AbstractFactory;

    std::unique_ptr<RestrictProlong> generate(
        std::shared_ptr<const LinOp> input) const
    {
        return AbstractFactory::generate(input);
    }
};


/**
 * This is an alias for the EnableDefaultFactory mixin, which correctly sets the
 * template parameters to enable a subclass of RestrictProlongFactory.
 *
 * @tparam ConcreteFactory  the concrete factory which is being implemented
 *                          [CRTP parmeter]
 * @tparam ConcreteRstrPrlg  the concrete RestrictProlong type which this
 * factory produces, needs to have a constructor which takes a const
 * ConcreteFactory *, and an std::shared_ptr<const LinOp> as parameters.
 * @tparam ParametersType  a subclass of enable_parameters_type template which
 *                         defines all of the parameters of the factory
 * @tparam PolymorphicBase  parent of ConcreteFactory in the polymorphic
 *                          hierarchy, has to be a subclass of
 * RestrictProlongFactory
 *
 * @ingroup LinOp
 */
template <typename ConcreteFactory, typename ConcreteRstrPrlg,
          typename ParametersType,
          typename PolymorphicBase = RestrictProlongFactory>
using EnableDefaultRestrictProlongFactory =
    EnableDefaultFactory<ConcreteFactory, ConcreteRstrPrlg, ParametersType,
                         PolymorphicBase>;


/**
 * The EnableRestrictProlong mixin can be used to provide sensible default
 * implementations of the majority of the RestrictProlong and PolymorphicObject
 * interface.
 *
 * *
 * The goal of the mixin is to facilitate the development of new
 * RestrictProlong, by enabling the implementers to focus on the important parts
 * of their operator, while the library takes care of generating the trivial
 * utility functions. The mixin will provide default implementations for the
 * entire PolymorphicObject interface, including a default implementation of
 * `copy_from` between objects of the new LinOp type. It will also hide the
 * default RestrictPrlong::restrict_apply() and
 * RestrictPrlong::prolong_applyadd() methods with versions that preserve the
 * static type of the object.
 *
 * Implementers of new RestrictProlong are required to specify only the
 * following aspects:
 *
 * 1.  Creation of the RestrictProlong: This can be facilitated via
 *     GKO_ENABLE_RESTRICT_PROLONG_FACTORY macro.
 * 2.  Application of the RestrictProlong: Implementers have to override
 *     RestrictPrlong::restrict_apply_impl() and
 *     RestrictPrlong::prolong_applyadd_impl() virtual methods and use
 *     set_coarse_fine() to set the generated results.
 *
 * @tparam ConcreteRstrPrlg  the concrete RestrictProlong which is being
 *                           implemented [CRTP parameter]
 * @tparam PolymorphicBase  parent of ConcreteRstrPrlg in the polymorphic
 *                          hierarchy, has to be a subclass of RestrictProlong
 *
 * @ingroup RestrictProlong
 * @ingroup Multigrid
 */
template <typename ConcreteRstrPrlg, typename PolymorphicBase = RestrictProlong>
class EnableRestrictProlong
    : public EnablePolymorphicObject<ConcreteRstrPrlg, PolymorphicBase>,
      public EnablePolymorphicAssignment<ConcreteRstrPrlg> {
public:
    using EnablePolymorphicObject<ConcreteRstrPrlg,
                                  PolymorphicBase>::EnablePolymorphicObject;

    ConcreteRstrPrlg *restrict_apply(const LinOp *b, LinOp *x)
    {
        this->validate_restrict_parameters(b, x);
        auto exec = this->get_executor();
        this->restrict_apply_impl(make_temporary_clone(exec, b).get(),
                                  make_temporary_clone(exec, x).get());
        return self();
    }

    const ConcreteRstrPrlg *restrict_apply(const LinOp *b, LinOp *x) const
    {
        this->validate_restrict_parameters(b, x);
        auto exec = this->get_executor();
        this->restrict_apply_impl(make_temporary_clone(exec, b).get(),
                                  make_temporary_clone(exec, x).get());
        return self();
    }

    ConcreteRstrPrlg *prolong_applyadd(const LinOp *b, LinOp *x)
    {
        this->validate_prolong_parameters(b, x);
        auto exec = this->get_executor();
        this->prolong_applyadd_impl(make_temporary_clone(exec, b).get(),
                                    make_temporary_clone(exec, x).get());
        return self();
    }

    const ConcreteRstrPrlg *prolong_applyadd(const LinOp *b, LinOp *x) const
    {
        this->validate_prolong_parameters(b, x);
        auto exec = this->get_executor();
        this->prolong_applyadd_impl(make_temporary_clone(exec, b).get(),
                                    make_temporary_clone(exec, x).get());
        return self();
    }

protected:
    GKO_ENABLE_SELF(ConcreteRstrPrlg);
};


#define GKO_ENABLE_RESTRICT_PROLONG_FACTORY(_restrict_prolong,                \
                                            _parameters_name, _factory_name)  \
public:                                                                       \
    const _parameters_name##_type &get_##_parameters_name() const             \
    {                                                                         \
        return _parameters_name##_;                                           \
    }                                                                         \
                                                                              \
    class _factory_name                                                       \
        : public ::gko::multigrid::EnableDefaultRestrictProlongFactory<       \
              _factory_name, _restrict_prolong, _parameters_name##_type> {    \
        friend class ::gko::EnablePolymorphicObject<                          \
            _factory_name, ::gko::multigrid::RestrictProlongFactory>;         \
        friend class ::gko::enable_parameters_type<_parameters_name##_type,   \
                                                   _factory_name>;            \
        explicit _factory_name(std::shared_ptr<const ::gko::Executor> exec)   \
            : ::gko::multigrid::EnableDefaultRestrictProlongFactory<          \
                  _factory_name, _restrict_prolong, _parameters_name##_type>( \
                  std::move(exec))                                            \
        {}                                                                    \
        explicit _factory_name(std::shared_ptr<const ::gko::Executor> exec,   \
                               const _parameters_name##_type &parameters)     \
            : ::gko::multigrid::EnableDefaultRestrictProlongFactory<          \
                  _factory_name, _restrict_prolong, _parameters_name##_type>( \
                  std::move(exec), parameters)                                \
        {}                                                                    \
    };                                                                        \
    friend ::gko::multigrid::EnableDefaultRestrictProlongFactory<             \
        _factory_name, _restrict_prolong, _parameters_name##_type>;           \
                                                                              \
private:                                                                      \
    _parameters_name##_type _parameters_name##_;                              \
                                                                              \
public:                                                                       \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")


}  // namespace multigrid
}  // namespace gko


#endif  // GKO_CORE_MULTIGRID_RESTRICT_PROLONG_HPP_
