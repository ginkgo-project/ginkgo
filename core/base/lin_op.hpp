/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_CORE_BASE_LIN_OP_HPP_
#define GKO_CORE_BASE_LIN_OP_HPP_


#include "core/base/abstract_factory.hpp"
#include "core/base/exception_helpers.hpp"
#include "core/base/matrix_data.hpp"
#include "core/base/polymorphic_object.hpp"
#include "core/base/std_extensions.hpp"
#include "core/base/types.hpp"
#include "core/base/utils.hpp"


#include <memory>


namespace gko {


/**
 * The linear operator (LinOp) is a base class for all linear algebra objects
 * in Ginkgo. The main benefit of having a single base class for the
 * entire collection of linear algebra objects (as opposed to having separate
 * hierarchies for matrices, solvers and preconditioners) is the generality
 * it provides.
 *
 * First, since all subclasses provide a common interface, the library users are
 * exposed to a smaller set of routines. For example, a
 * matrix-vector product, a preconditioner application, or even a system solve
 * are just different terms given to the operation of applying a certain linear
 * operator to a vector. As such, Ginkgo uses the same routine name,
 * LinOp::apply() for each of these operations, where the actual
 * operation performed depends on the type of linear operator involved in
 * the operation.
 *
 * Second, a common interface often allows for writing more generic code. If a
 * user's routine requires only operations provided by the LinOp interface,
 * the same code can be used for any kind of linear operators, independent of
 * whether these are matrices, solvers or preconditioners. This feature is also
 * extensively used in Ginkgo itself. For example, a preconditioner used
 * inside a Krylov solver is a LinOp. This allows the user to supply a wide
 * variety of preconditioners: either the ones which were designed to be used
 * in this scenario (like ILU or block-Jacobi), a user-supplied matrix which is
 * known to be a good preconditioner for the specific problem,
 * or even another solver (e.g., if constructing a flexible GMRES solver).
 *
 * A key observation for providing a unified interface for matrices, solvers,
 * and preconditioners is that the most common operation performed on all of
 * them can be expressed as an application of a linear operator to a vector:
 *
 * +   the sparse matrix-vector product with a matrix \f$A\f$ is a linear
 *     operator application \f$y = Ax\f$;
 * +   the application of a preconditioner is a linear operator application
 *     \f$y = M^{-1}x\f$, where \f$M\f$ is an approximation of the original
 *     system matrix \f$A\f$ (thus a preconditioner represents an "approximate
 *     inverse" operator \f$M^{-1}\f$).
 * +   the system solve \f$Ax = b\f$ can be viewed as linear operator
 *     application
 *     \f$x = A^{-1}b\f$ (it goes without saying that the implementation of
 *     linear system solves does not follow this conceptual idea), so a linear
 *     system solver can be viewed as a representation of the operator
 *     \f$A^{-1}\f$.
 *
 * Finally, direct manipulation of LinOp objects is rarely required in
 * simple scenarios. As an illustrative example, one could construct a
 * fixed-point iteration routine \f$x_{k+1} = Lx_k + b\f$ as follows:
 *
 * ```cpp
 * std::unique_ptr<matrix::Dense<>> calculate_fixed_point(
 *         int iters, const LinOp *L, const matrix::Dense<> *x0
 *         const matrix::Dense<> *b)
 * {
 *     auto x = gko::clone(x0);
 *     auto tmp = gko::clone(x0);
 *     auto one = Dense<>::create(L->get_executor(), {1.0,});
 *     for (int i = 0; i < iters; ++i) {
 *         L->apply(gko::lend(tmp), gko::lend(x));
 *         x->add_scaled(gko::lend(one), gko::lend(b));
 *         tmp->copy_from(gko::lend(x));
 *     }
 *     return x;
 * }
 * ```
 *
 * Here, if \f$L\f$ is a matrix, LinOp::apply() refers to the matrix vector
 * product, and `L->apply(a, b)` computes \f$b = L \cdot a\f$.
 * `x->add_scaled(one.get(), b.get())` is the `axpy` vector update \f$x:=x+b\f$.
 *
 * The interesting part of this example is the apply() routine at line 4 of the
 * function body. Since this routine is part of the LinOp base class, the
 * fixed-point iteration routine can calculate a fixed point not only for
 * matrices, but for any type of linear operator.
 */
class LinOp : public EnableAbstractPolymorphicObject<LinOp> {
public:
    /**
     * Applies a linear operator to a vector (or a sequence of vectors).
     *
     * Performs the operation x = op(b), where op is this linear operator.
     *
     * @param b  the input vector(s) on which the operator is applied
     * @param x  the output vector(s) where the result is stored
     *
     * @return this
     */
    LinOp *apply(const LinOp *b, LinOp *x)
    {
        this->validate_application_parameters(b, x);
        auto exec = this->get_executor();
        this->apply_impl(make_temporary_clone(exec, b).get(),
                         make_temporary_clone(exec, x).get());
        return this;
    }

    /**
     * @copydoc apply(cost LinOp *, LinOp *)
     */
    const LinOp *apply(const LinOp *b, LinOp *x) const
    {
        this->validate_application_parameters(b, x);
        auto exec = this->get_executor();
        this->apply_impl(make_temporary_clone(exec, b).get(),
                         make_temporary_clone(exec, x).get());
        return this;
    }

    /**
     * Performs the operation x = alpha * op(b) + beta * x.
     *
     * @param alpha  scaling of the result of op(b)
     * @param b  vector(s) on which the operator is applied
     * @param beta  scaling of the input x
     * @param x  output vector(s)
     *
     * @return this
     */
    LinOp *apply(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                 LinOp *x)
    {
        this->validate_application_parameters(alpha, b, beta, x);
        auto exec = this->get_executor();
        this->apply_impl(make_temporary_clone(exec, alpha).get(),
                         make_temporary_clone(exec, b).get(),
                         make_temporary_clone(exec, beta).get(),
                         make_temporary_clone(exec, x).get());
        return this;
    }

    /**
     * @copydoc apply(const LinOp *, cost LinOp *, const LinOp *, LinOp *)
     */
    const LinOp *apply(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                       LinOp *x) const
    {
        this->validate_application_parameters(alpha, b, beta, x);
        auto exec = this->get_executor();
        this->apply_impl(make_temporary_clone(exec, alpha).get(),
                         make_temporary_clone(exec, b).get(),
                         make_temporary_clone(exec, beta).get(),
                         make_temporary_clone(exec, x).get());
        return this;
    }

    /**
     * Returns the size of the operator.
     *
     * @return size of the operator
     */
    const dim &get_size() const noexcept { return size_; }

protected:
    /**
     * Creates a linear operator.
     *
     * @param exec  the executor where all the operations are performed
     * @param size  the size of the operator
     */
    explicit LinOp(std::shared_ptr<const Executor> exec,
                   const dim &size = dim{})
        : EnableAbstractPolymorphicObject<LinOp>(exec), size_{size}
    {}

    /**
     * Sets the size of the operator.
     *
     * @param value  the new size of the operator
     */
    void set_size(const dim &value) noexcept { size_ = value; }

    /**
     * Implementers of LinOp should override this function instead
     * of apply(const LinOp *, LinOp *).
     *
     * Performs the operation x = op(b), where op is this linear operator.
     *
     * @param b  the input vector(s) on which the operator is applied
     * @param x  the output vector(s) where the result is stored
     */
    virtual void apply_impl(const LinOp *b, LinOp *x) const = 0;

    /**
     * Implementers of LinOp should override this function instead
     * of apply(const LinOp *, const LinOp *, const LinOp *, LinOp *).
     *
     * @param alpha  scaling of the result of op(b)
     * @param b  vector(s) on which the operator is applied
     * @param beta  scaling of the input x
     * @param x  output vector(s)
     */
    virtual void apply_impl(const LinOp *alpha, const LinOp *b,
                            const LinOp *beta, LinOp *x) const = 0;

    /**
     * Throws a DimensionMismatch exception if the parameters to `apply` are of
     * the wrong size.
     *
     * @param b  vector(s) on which the operator is applied
     * @param x  output vector(s)
     */
    void validate_application_parameters(const LinOp *b, const LinOp *x) const
    {
        ASSERT_CONFORMANT(this, b);
        ASSERT_EQUAL_ROWS(this, x);
        ASSERT_EQUAL_COLS(b, x);
    }

    /**
     * Throws a DimensionMismatch exception if the parameters to `apply` are of
     * the wrong size.
     *
     * @param alpha  scaling of the result of op(b)
     * @param b  vector(s) on which the operator is applied
     * @param beta  scaling of the input x
     * @param x  output vector(s)
     */
    void validate_application_parameters(const LinOp *alpha, const LinOp *b,
                                         const LinOp *beta,
                                         const LinOp *x) const
    {
        this->validate_application_parameters(b, x);
        ASSERT_EQUAL_DIMENSIONS(alpha, dim(1, 1));
        ASSERT_EQUAL_DIMENSIONS(beta, dim(1, 1));
    }

private:
    dim size_{};
};


/**
 * A LinOpFactory represents a higher order mapping which transforms one
 * linear operator into another.
 *
 * In Ginkgo, every linear solver is viewed as a mapping. For example,
 * given an s.p.d linear system \f$Ax = b\f$, the solution \f$x = A^{-1}b\f$
 * can be computed using the CG method. This algorithm can be represented in
 * terms of linear operators and mappings between them as follows:
 *
 * -   A Cg::Factory is a higher order mapping which, given an input operator
 *     \f$A\f$, returns a new linear operator \f$A^{-1}\f$ stored in "CG
 *     format"
 * -   Storing the operator \f$A^{-1}\f$ in "CG format" means that the data
 *     structure used to store the operator is just a simple pointer to the
 *     original matrix \f$A\f$. The application \f$x = A^{-1}b\f$ of such an
 *     operator can then be implemented by solving the linear system
 *     \f$Ax = b\f$ using the CG method. This is achieved in code by having a
 *     special class for each of those "formats" (e.g. the "Cg" class defines
 *     such a format for the CG solver).
 *
 * Another example of a LinOpFactory is a preconditioner. A preconditioner for
 * a linear operator \f$A\f$ is a linear operator \f$M^{-1}\f$, which
 * approximates \f$A^{-1}\f$. In addition, it is stored in a way such that
 * both the data of \f$M^{-1}\f$ is cheap to compute from \f$A\f$, and the
 * operation \f$x = M^{-1}b\f$ can be computed quickly. These operators are
 * useful to accelerate the convergence of  Krylov solvers.
 * Thus, a preconditioner also fits into the LinOpFactory framework:
 *
 * -   The factory maps a linear operator \f$A\f$ into a preconditioner
 *     \f$M^{-1}\f$ which is stored in suitable format (e.g. as a product of
 *     two factors in case of ILU preconditioners).
 * -   The resulting linear operator implements the application operation
 *     \f$x = M^{-1}b\f$ depending on the format the preconditioner is stored
 *     in (e.g. as two triangular solves in case of ILU)
 *
 * Example: using CG in Ginkgo
 * ---------------------------
 *
 * ```c++
 * // Suppose A is a matrix, b a rhs vector, and x an initial guess
 * // Create a CG which runs for at most 1000 iterations, and stops after
 * // reducing the residual norm by 6 orders of magnitude
 * auto cg_factory = solver::Cg::Factory<>::create()
 *     .with_max_iters(1000)
 *     .with_rel_residual_goal(1e-6)
 *     .on_executor(gpu);
 * // create a linear operator which represents the solver
 * auto cg = cg_factory->generate(A);
 * // solve the system
 * cg->apply(gko::lend(b), gko::lend(x));
 * ```
 */
using LinOpFactory = AbstractFactory<LinOp, std::shared_ptr<const LinOp>>;


/**
 * Linear operators which support transposition should implement the
 * Transposable interface.
 *
 * It provides two functionalities, the normal transpose and the
 * conjugate transpose.
 *
 * The normal transpose returns the transpose of the linear operator without
 * changing any of its elements representing the operation, \f$B = A^{T}\f$.
 *
 * The conjugate transpose returns the conjugate of each of the elements and
 * additionally transposes the linear operator representing the operation, \f$B
 * = A^{H}\f$.
 *
 * Example: Transposing a Csr matrix:
 * ------------------------------------
 *
 * ```c++
 * //Transposing an object of LinOp type.
 * //The object you want to transpose.
 * auto op = matrix::Csr::create(exec);
 * //Transpose the object by first converting it to a transposable type.
 * auto trans = op->transpose();
 * ```
 */
class Transposable {
public:
    virtual ~Transposable() = default;

    /**
     * Returns a LinOp representing the transpose of the Transposable object.
     *
     * @return a pointer to the new transposed object
     */
    virtual std::unique_ptr<LinOp> transpose() const = 0;

    /**
     * Returns a LinOp representing the conjugate transpose of the Transposable
     * object.
     *
     * @return a pointer to the new conjugate transposed object
     */
    virtual std::unique_ptr<LinOp> conj_transpose() const = 0;
};


/**
 * A LinOp implementing this interface can read its data from a matrix_data
 * structure.
 */
template <typename ValueType, typename IndexType>
class ReadableFromMatrixData {
public:
    virtual ~ReadableFromMatrixData() = default;

    /**
     * Reads a matrix from a matrix_data structure.
     *
     * @param data  the matrix_data structure
     */
    virtual void read(const matrix_data<ValueType, IndexType> &data) = 0;
};


/**
 * A LinOp implementing this interface can write its data to a matrix_data
 * structure.
 */
template <typename ValueType, typename IndexType>
class WritableToMatrixData {
public:
    virtual ~WritableToMatrixData() = default;

    /**
     * Writes a matrix to a matrix_data structure.
     *
     * @param data  the matrix_data structure
     */
    virtual void write(matrix_data<ValueType, IndexType> &data) const = 0;
};


/**
 * The EnableLinOp mixin can be used to provide sensible default implementations
 * of the majority of the LinOp and PolymorphicObject interface.
 *
 * The goal of the mixin is to facilitate the development of new LinOp, by
 * enabling the implementers to focus on the importan parts of their operator,
 * while the library takes care of generating the trivial utility functions.
 * The mixin will provide default implementations for the entire
 * PolymorphicObject interface, including a default implementation of
 * `copy_from` between objects of the new LinOp type. It will also hide the
 * default LinOp::apply() methods with versions that preserve the static type of
 * the object.
 *
 * Implementers of new LinOps are required to specify only the following
 * aspects:
 *
 * 1.  Creation of the LinOp: This can be facilitated via either
 *     EnableCreateMethod mixin (used mostly for matrix formats),
 *     or GKO_ENABLE_LIN_OP_FACTORY macro (used for operators created from other
 *     operators, like preconditioners and solvers).
 * 2.  Appllication of the LinOp: Implementers have to override the two
 *     overloads of the LinOp::apply_impl() virtual methods.
 *
 * @tparam ConcreteLinOp  the concrete LinOp which is being implemented
 *                        [CRTP parameter]
 * @tparam PolymorphicBase  parent of ConcreteLinOp in the polymorphic
 *                          hierarchy, has to be a subclass of LinOp
 */
template <typename ConcreteLinOp, typename PolymorphicBase = LinOp>
class EnableLinOp
    : public EnablePolymorphicObject<ConcreteLinOp, PolymorphicBase>,
      public EnablePolymorphicAssignment<ConcreteLinOp> {
public:
    using EnablePolymorphicObject<ConcreteLinOp,
                                  PolymorphicBase>::EnablePolymorphicObject;

    const ConcreteLinOp *apply(const LinOp *b, LinOp *x) const
    {
        this->validate_application_parameters(b, x);
        auto exec = this->get_executor();
        this->apply_impl(make_temporary_clone(exec, b).get(),
                         make_temporary_clone(exec, x).get());
        return self();
    }

    ConcreteLinOp *apply(const LinOp *b, LinOp *x)
    {
        this->validate_application_parameters(b, x);
        auto exec = this->get_executor();
        this->apply_impl(make_temporary_clone(exec, b).get(),
                         make_temporary_clone(exec, x).get());
        return self();
    }

    const ConcreteLinOp *apply(const LinOp *alpha, const LinOp *b,
                               const LinOp *beta, LinOp *x) const
    {
        this->validate_application_parameters(alpha, b, beta, x);
        auto exec = this->get_executor();
        this->apply_impl(make_temporary_clone(exec, alpha).get(),
                         make_temporary_clone(exec, b).get(),
                         make_temporary_clone(exec, beta).get(),
                         make_temporary_clone(exec, x).get());
        return self();
    }

    ConcreteLinOp *apply(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                         LinOp *x)
    {
        this->validate_application_parameters(alpha, b, beta, x);
        auto exec = this->get_executor();
        this->apply_impl(make_temporary_clone(exec, alpha).get(),
                         make_temporary_clone(exec, b).get(),
                         make_temporary_clone(exec, beta).get(),
                         make_temporary_clone(exec, x).get());
        return self();
    }

protected:
    GKO_ENABLE_SELF(ConcreteLinOp);
};


/**
 * This is an alias for the EnableDefaultFactory mixin, which correctly sets the
 * template parameters to enable a subclass of LinOpFactory.
 *
 * @tparam ConcreteFactory  the concrete factory which is being implemented
 *                          [CRTP parmeter]
 * @tparam ConcreteLinOp  the concrete LinOp type which this factory produces,
 *                        needs to have a constructor which takes a
 *                        const ConcreteFactory *, and an
 *                        std::shared_ptr<const LiOp> as parameters.
 * @tparam ParametersType  a subclass of parameters_type_base template which
 *                         defines all of the parameters of the factory
 */
template <typename ConcreteFactory, typename ConcreteLinOp,
          typename ParametersType>
using EnableDefaultLinOpFactory =
    EnableDefaultFactory<ConcreteFactory, ConcreteLinOp, LinOp,
                         std::shared_ptr<const LinOp>, ParametersType>;


/**
 * This macro will generate a default implementation of a LinOpFactory for the
 * LinOp subclass it is defined in.
 *
 * The list of parameters for the factory should be defined in a code block
 * after the macro definition, and should contain a list of
 * GKO_FACTORY_PARMETER declarations. The class should provide a constructor
 * with signature
 * _lin_op(const _factory_name *, std::shared_ptr<const LinOp>)
 * which the factory will use a callback to construct the object.
 *
 * A minimal example of a linear operator is the following:
 *
 * ```c++
 * struct MyLinOp : public EnableLinOp<MyLinOp> {
 *     GKO_ENABLE_LIN_OP_FACTORY(MyLinOp, my_parameters, Factory) {
 *         // a factory parameter named "my_value", of type int and default
 *         // value of 5
 *         int GKO_FACTORY_PARAMETER(my_value, 5);
 *     };
 *     // constructor needed by EnableLinOp
 *     explicit MyLinOp(std::shared_ptr<const Executor> exec) {
 *         : EnableLinOp<MyLinOp>(exec) {}
 *     // constructor needed by the factory
 *     explicit MyLinOp(const Factory *factory,
 *                      std::shared_ptr<const LinOp> matrix)
 *         : EnableLinOp<MyLinOp>(factory->get_executor()), matrix->get_size()),
 *           // store factory's parameters locally
 *           my_parameters_{factory->get_parameters()},
 *     {
 *          int value = my_parameters_.my_value;
 *          // do something with value
 *     }
 * ```
 *
 * MyLinOp can then be created as follows:
 *
 * ```c++
 * auto exec = gko::ReferenceExecutor::create();
 * // create a factory with default `my_value` parameter
 * auto fact = MyLinOp::Factory::create().on_executor(exec);
 * // create a operator using the factory:
 * auto my_op = fact->generate(gko::matrix::Identity::create(exec, 2));
 * std::cout << my_op->get_my_parameters().my_value;  // prints 5
 *
 * // create a factory with custom `my_value` parameter
 * auto fact = MyLinOp::Factory::create().with_my_value(0).on_executor(exec);
 * // create a operator using the factory:
 * auto my_op = fact->generate(gko::matrix::Identity::create(exec, 2));
 * std::cout << my_op->get_my_parameters().my_value;  // prints 0
 * ```
 *
 * @param _lin_op  concrete operator for which the factory is to be created
 *                 [CRTP parameter]
 * @param _parameters_name  name of the parameters member in the class
 *                          (it's type is `<_parameters_name>_type`, the
 *                          protected member's name is `<_parameters_name>_`,
 *                          and the public getter's name is
 *                          `get_<_parameters_name>()`)
 * @param _facory_name  name of the generated factory type
 */
#define GKO_ENABLE_LIN_OP_FACTORY(_lin_op, _parameters_name, _factory_name) \
public:                                                                     \
    struct _parameters_name##_type;                                         \
                                                                            \
    const _parameters_name##_type &get_##_parameters_name() const           \
    {                                                                       \
        return _parameters_name##_;                                         \
    }                                                                       \
                                                                            \
    class _factory_name                                                     \
        : public gko::EnableDefaultLinOpFactory<_factory_name, _lin_op,     \
                                                _parameters_name##_type> {  \
        friend class gko::EnablePolymorphicObject<_factory_name,            \
                                                  LinOpFactory>;            \
        friend class gko::parameters_type_base<_parameters_name##_type,     \
                                               _factory_name>;              \
        using gko::EnableDefaultLinOpFactory<                               \
            _factory_name, _lin_op,                                         \
            _parameters_name##_type>::EnableDefaultLinOpFactory;            \
    };                                                                      \
    friend gko::EnableDefaultLinOpFactory<_factory_name, _lin_op,           \
                                          _parameters_name##_type>;         \
                                                                            \
private:                                                                    \
    _parameters_name##_type _parameters_name##_;                            \
                                                                            \
public:                                                                     \
    struct _parameters_name##_type                                          \
        : gko::parameters_type_base<_parameters_name##_type, _factory_name>


/**
 * Creates a factory parameter in the factory parameters structure.
 *
 * @param _name  name of the parameter
 * @param __VA_ARGS__  default value of the parameter
 *
 * @see GKO_ENABLE_LIN_OP_FACTORY for more details, and usage example
 */
#define GKO_FACTORY_PARAMETER(_name, ...)             \
    mutable _name{__VA_ARGS__};                       \
                                                      \
    auto with_##_name(const decltype(_name) &value)   \
        const->const xstd::decay_t<decltype(*this)> & \
    {                                                 \
        _name = value;                                \
        return *this;                                 \
    }


}  // namespace gko


#endif  // GKO_CORE_BASE_LIN_OP_HPP_
