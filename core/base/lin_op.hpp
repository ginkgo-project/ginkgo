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


#include "core/base/convertible.hpp"
#include "core/base/executor.hpp"
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
 *     auto x = x0->clone();
 *     auto tmp = x0->clone();
 *     auto one = Dense<>::create(L->get_executor(), {1.0,});
 *     for (int i = 0; i < iters; ++i) {
 *         L->apply(tmp.get(), x.get());
 *         x->add_scaled(one.get(), b.get());
 *         tmp->copy_from(x.get());
 *     }
 *     return std::move(x);
 * }
 * ```
 * Here, if \f$L\f$ is a matrix, LinOp::apply() refers to the matrix vector
 * product, and `L->apply(a, b)` computes \f$b = L \cdot a\f$.
 * `x->add_scaled(one.get(), b.get())` is the `axpy` vector update \f$x:=x+b\f$.
 *
 * The interesting part of this example is the apply() routine at line 4 of the
 * function body. Since this routine is part of the LinOp base class, the
 * fixed-point iteration routine can calculate a fixed point not only for
 * matrices, but for any type of linear operator.
 */
class LinOp {
public:
    LinOp &operator=(const LinOp &other)
    {
        // keep the executor as it is
        num_rows_ = other.num_rows_;
        num_cols_ = other.num_cols_;
        num_stored_elements_ = other.num_stored_elements_;
        return *this;
    }

    virtual ~LinOp() = default;

    /**
     * Creates a copy of another LinOp.
     *
     * @param other  the LinOp to copy
     */
    virtual void copy_from(const LinOp *other) = 0;

    /**
     * Moves the data from another LinOp.
     *
     * @param other  the LinOp from which the data will be moved
     */
    virtual void copy_from(std::unique_ptr<LinOp> other) = 0;

    /**
     * Applies a linear operator to a vector (or a sequence of vectors).
     *
     * Performs the operation x = op(b), where op is this linear operator.
     *
     * @param b  the input vector(s) on which the operator is applied
     * @param x  the output vector(s) where the result is stored
     */
    virtual void apply(const LinOp *b, LinOp *x) const = 0;

    /**
     * Performs the operation x = alpha * op(b) + beta * x.
     *
     * @param alpha  scaling of the result of op(b)
     * @param b  vector(s) on which the operator is applied
     * @param beta  scaling of the input x
     * @param x  output vector(s)
     */
    virtual void apply(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                       LinOp *x) const = 0;

    /**
     * Creates a clone of the LinOp.
     *
     * @return A clone of the LinOp.
     */
    std::unique_ptr<LinOp> clone() const
    {
        auto new_op = this->clone_type();
        new_op->copy_from(this);
        return new_op;
    }

    /**
     * Creates a new 0x0 LinOp of the same type.
     *
     * @return  a LinOp object of the same type as this
     */
    virtual std::unique_ptr<LinOp> clone_type() const = 0;

    /**
     * Transforms the object into an empty LinOp.
     */
    virtual void clear() = 0;

    /**
     * Gets the Executor of this object.
     */
    std::shared_ptr<const Executor> get_executor() const noexcept
    {
        return exec_;
    }

    /**
     * Gets the dimension of the codomain of this LinOp.
     *
     * In other words, the number of rows of the coefficient matrix.
     *
     * @return the dimension of the codomain
     */
    size_type get_num_rows() const noexcept { return num_rows_; }

    /**
     * Gets the dimension of the domain of this LinOp.
     *
     * In other words, the number of columns of the coefficient matrix.
     *
     * @return the dimension of the domain
     */
    size_type get_num_cols() const noexcept { return num_cols_; }

    /**
     * Returns the number of elements that are explicitly stored in memory for
     * this LinOp.
     *
     * For example, for a matrix::Dense `A` it will always hold
     * ```cpp
     * A->get_num_stored_elements() == A->get_num_rows() * A->get_padding()
     * ```
     *
     * @return the number of elements explicitly stored in memory
     */
    size_type get_num_stored_elements() const noexcept
    {
        return num_stored_elements_;
    }

protected:
    LinOp(std::shared_ptr<const Executor> exec, size_type num_rows,
          size_type num_cols, size_type num_stored_elements)
        : exec_(exec),
          num_rows_(num_rows),
          num_cols_(num_cols),
          num_stored_elements_(num_stored_elements)
    {}

    void set_dimensions(size_type num_rows, size_type num_cols,
                        size_type num_stored_elements) noexcept
    {
        num_rows_ = num_rows;
        num_cols_ = num_cols;
        num_stored_elements_ = num_stored_elements;
    }

    void set_dimensions(const LinOp *op) noexcept
    {
        num_rows_ = op->num_rows_;
        num_cols_ = op->num_cols_;
        num_stored_elements_ = op->num_stored_elements_;
    }

private:
    std::shared_ptr<const Executor> exec_;
    size_type num_rows_;
    size_type num_cols_;
    size_type num_stored_elements_;
};


template <typename ConcreteLinOp>
class BasicLinOp : public LinOp, public ConvertibleTo<ConcreteLinOp> {
public:
    using LinOp::LinOp;

    BasicLinOp &operator=(const BasicLinOp &) = default;

    BasicLinOp &operator=(BasicLinOp &&) = default;

    void copy_from(const LinOp *other) override
    {
        auto conv = as<ConvertibleTo<ConcreteLinOp>>(other);
        conv->convert_to(self());
    }

    void copy_from(std::unique_ptr<LinOp> other) override
    {
        auto conv = as<ConvertibleTo<ConcreteLinOp>>(other.get());
        conv->move_to(self());
    }

    void convert_to(ConcreteLinOp *other) const override { *other = *self(); }

    void move_to(ConcreteLinOp *other) override { *other = std::move(*self()); }

private:
    ConcreteLinOp *self() noexcept
    {
        return static_cast<ConcreteLinOp *>(this);
    }

    const ConcreteLinOp *self() const noexcept
    {
        return static_cast<const ConcreteLinOp *>(this);
    }
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
 * -   A CgFactory is a higher order mapping which, given an input operator
 *     \f$A\f$, returns a new linear operator \f$A^{-1}\f$ stored in "CG
 *     format"
 * -   Storing the operator \f$A^{-1}\f$ in "CG format" means that the data
 *     structure used to store the operator is just a simple pointer to the
 *     original matrix \f$A\f$. The application \f$x = A^{-1}b\f$ of such an
 *     operator can then be implemented by solving the linear system
 *     \f$Ax = b\f$ using the CG method.
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
 * auto cg_factory = solver::CgFactory<>::create(gpu, 1000, 1e-6);
 * // create a linear operator which represents the solver
 * auto cg = cg_factory->generate(A);
 * // solve the system
 * cg->apply(b.get(), x.get());
 * ```
 */
class LinOpFactory {
public:
    /**
     * Generates a Linear operator from the base linear operator.
     *
     * @param base  The base linear operator.
     */
    virtual std::unique_ptr<LinOp> generate(
        std::shared_ptr<const LinOp> base) const = 0;
    /**
     * Gets the exector on which the Linear operator exists.
     *
     * @return exec_ The executor.
     */
    std::shared_ptr<const Executor> get_executor() const noexcept
    {
        return exec_;
    }

protected:
    explicit LinOpFactory(std::shared_ptr<const Executor> exec)
        : exec_(std::move(exec))
    {}

private:
    std::shared_ptr<const Executor> exec_;
};


/**
 * Linear operators which support transposition implement the Transposable
 * interface.
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
 * std::unique_ptr<LinOp> op = matrix::Csr::create(exec);
 * //Transpose the object by first converting it to a transposable type.
 * auto trans = as<Transposable>(op.get())->transpose();
 * //This returns the object of type LinOp, and needs to be cast to the
 * appropriate type for usage.
 * ```
 */
class Transposable {
public:
    /**
     * Returns a LinOp representing the transpose of the Transposable object.
     *
     * @return A pointer to the new transposed object.
     */
    virtual std::unique_ptr<LinOp> transpose() const = 0;

    /**
     * Returns a LinOp representing the conjugate transpose of the Transposable
     * object.
     *
     * @return A pointer to the new conjugate transposed object.
     */
    virtual std::unique_ptr<LinOp> conj_transpose() const = 0;
};


}  // namespace gko


#endif  // GKO_CORE_BASE_LIN_OP_HPP_
