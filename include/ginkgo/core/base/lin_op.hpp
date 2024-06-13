// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_LIN_OP_HPP_
#define GKO_PUBLIC_CORE_BASE_LIN_OP_HPP_


#include <memory>
#include <type_traits>
#include <utility>


#include <ginkgo/core/base/abstract_factory.hpp>
#include <ginkgo/core/base/device_matrix_data.hpp>
#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/matrix_assembly_data.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/log/logger.hpp>


namespace gko {
namespace matrix {


template <typename ValueType>
class Diagonal;


}  // namespace matrix


/**
 * @addtogroup LinOp
 *
 * @section linop_concept Linear operator as a concept
 *
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
 * +   the sparse matrix-vector product with a matrix $A$ is a linear
 *     operator application $y = Ax$;
 * +   the application of a preconditioner is a linear operator application
 *     $y = M^{-1}x$, where $M$ is an approximation of the original
 *     system matrix $A$ (thus a preconditioner represents an "approximate
 *     inverse" operator $M^{-1}$).
 * +   the system solve $Ax = b$ can be viewed as linear operator
 *     application
 *     $x = A^{-1}b$ (it goes without saying that the implementation of
 *     linear system solves does not follow this conceptual idea), so a linear
 *     system solver can be viewed as a representation of the operator
 *     $A^{-1}$.
 *
 * Finally, direct manipulation of LinOp objects is rarely required in
 * simple scenarios. As an illustrative example, one could construct a
 * fixed-point iteration routine $x_{k+1} = Lx_k + b$ as follows:
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
 *         L->apply(tmp, x);
 *         x->add_scaled(one, b);
 *         tmp->copy_from(x);
 *     }
 *     return x;
 * }
 * ```
 *
 * Here, if $L$ is a matrix, LinOp::apply() refers to the matrix vector
 * product, and `L->apply(a, b)` computes $b = L \cdot a$.
 * `x->add_scaled(one, b)` is the `axpy` vector update $x:=x+b$.
 *
 * The interesting part of this example is the apply() routine at line 4 of the
 * function body. Since this routine is part of the LinOp base class, the
 * fixed-point iteration routine can calculate a fixed point not only for
 * matrices, but for any type of linear operator.
 *
 * @ref LinOp
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
    LinOp* apply(ptr_param<const LinOp> b, ptr_param<LinOp> x)
    {
        this->template log<log::Logger::linop_apply_started>(this, b.get(),
                                                             x.get());
        this->validate_application_parameters(b.get(), x.get());
        auto exec = this->get_executor();
        this->apply_impl(make_temporary_clone(exec, b).get(),
                         make_temporary_clone(exec, x).get());
        this->template log<log::Logger::linop_apply_completed>(this, b.get(),
                                                               x.get());
        return this;
    }

    /**
     * @copydoc apply(const LinOp *, LinOp *)
     */
    const LinOp* apply(ptr_param<const LinOp> b, ptr_param<LinOp> x) const
    {
        this->template log<log::Logger::linop_apply_started>(this, b.get(),
                                                             x.get());
        this->validate_application_parameters(b.get(), x.get());
        auto exec = this->get_executor();
        this->apply_impl(make_temporary_clone(exec, b).get(),
                         make_temporary_clone(exec, x).get());
        this->template log<log::Logger::linop_apply_completed>(this, b.get(),
                                                               x.get());
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
    LinOp* apply(ptr_param<const LinOp> alpha, ptr_param<const LinOp> b,
                 ptr_param<const LinOp> beta, ptr_param<LinOp> x)
    {
        this->template log<log::Logger::linop_advanced_apply_started>(
            this, alpha.get(), b.get(), beta.get(), x.get());
        this->validate_application_parameters(alpha.get(), b.get(), beta.get(),
                                              x.get());
        auto exec = this->get_executor();
        this->apply_impl(make_temporary_clone(exec, alpha).get(),
                         make_temporary_clone(exec, b).get(),
                         make_temporary_clone(exec, beta).get(),
                         make_temporary_clone(exec, x).get());
        this->template log<log::Logger::linop_advanced_apply_completed>(
            this, alpha.get(), b.get(), beta.get(), x.get());
        return this;
    }

    /**
     * @copydoc apply(const LinOp *, const LinOp *, const LinOp *, LinOp *)
     */
    const LinOp* apply(ptr_param<const LinOp> alpha, ptr_param<const LinOp> b,
                       ptr_param<const LinOp> beta, ptr_param<LinOp> x) const
    {
        this->template log<log::Logger::linop_advanced_apply_started>(
            this, alpha.get(), b.get(), beta.get(), x.get());
        this->validate_application_parameters(alpha.get(), b.get(), beta.get(),
                                              x.get());
        auto exec = this->get_executor();
        this->apply_impl(make_temporary_clone(exec, alpha).get(),
                         make_temporary_clone(exec, b).get(),
                         make_temporary_clone(exec, beta).get(),
                         make_temporary_clone(exec, x).get());
        this->template log<log::Logger::linop_advanced_apply_completed>(
            this, alpha.get(), b.get(), beta.get(), x.get());
        return this;
    }

    /**
     * Returns the size of the operator.
     *
     * @return size of the operator
     */
    const dim<2>& get_size() const noexcept { return size_; }

    /**
     * Returns true if the linear operator uses the data given in x as
     * an initial guess. Returns false otherwise.
     *
     * @return true if the linear operator uses the data given in x as
     *         an initial guess. Returns false otherwise.
     */
    virtual bool apply_uses_initial_guess() const { return false; }

    /** Copy-assigns a LinOp. Preserves the executor and copies the size. */
    LinOp& operator=(const LinOp&) = default;

    /**
     * Move-assigns a LinOp. Preserves the executor and moves the size.
     * The moved-from object has size 0x0 afterwards, but its executor is
     * unchanged.
     */
    LinOp& operator=(LinOp&& other)
    {
        if (this != &other) {
            EnableAbstractPolymorphicObject<LinOp>::operator=(std::move(other));
            this->set_size(other.get_size());
            other.set_size({});
        }
        return *this;
    }

    /** Copy-constructs a LinOp. Inherits executor and size from the input. */
    LinOp(const LinOp&) = default;

    /**
     * Move-constructs a LinOp. Inherits executor and size from the input,
     * which will have size 0x0 and unchanged executor afterwards.
     */
    LinOp(LinOp&& other)
        : EnableAbstractPolymorphicObject<LinOp>(std::move(other)),
          size_{std::exchange(other.size_, dim<2>{})}
    {}

protected:
    /**
     * Creates a linear operator.
     *
     * @param exec  the executor where all the operations are performed
     * @param size  the size of the operator
     */
    explicit LinOp(std::shared_ptr<const Executor> exec,
                   const dim<2>& size = dim<2>{})
        : EnableAbstractPolymorphicObject<LinOp>(exec), size_{size}
    {}

    /**
     * Sets the size of the operator.
     *
     * @param value  the new size of the operator
     */
    void set_size(const dim<2>& value) noexcept { size_ = value; }

    /**
     * Implementers of LinOp should override this function instead
     * of apply(const LinOp *, LinOp *).
     *
     * Performs the operation x = op(b), where op is this linear operator.
     *
     * @param b  the input vector(s) on which the operator is applied
     * @param x  the output vector(s) where the result is stored
     */
    virtual void apply_impl(const LinOp* b, LinOp* x) const = 0;

    /**
     * Implementers of LinOp should override this function instead
     * of apply(const LinOp *, const LinOp *, const LinOp *, LinOp *).
     *
     * @param alpha  scaling of the result of op(b)
     * @param b  vector(s) on which the operator is applied
     * @param beta  scaling of the input x
     * @param x  output vector(s)
     */
    virtual void apply_impl(const LinOp* alpha, const LinOp* b,
                            const LinOp* beta, LinOp* x) const = 0;

    /**
     * Throws a DimensionMismatch exception if the parameters to `apply` are of
     * the wrong size.
     *
     * @param b  vector(s) on which the operator is applied
     * @param x  output vector(s)
     */
    void validate_application_parameters(const LinOp* b, const LinOp* x) const
    {
        GKO_ASSERT_CONFORMANT(this, b);
        GKO_ASSERT_EQUAL_ROWS(this, x);
        GKO_ASSERT_EQUAL_COLS(b, x);
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
    void validate_application_parameters(const LinOp* alpha, const LinOp* b,
                                         const LinOp* beta,
                                         const LinOp* x) const
    {
        this->validate_application_parameters(b, x);
        GKO_ASSERT_EQUAL_DIMENSIONS(alpha, dim<2>(1, 1));
        GKO_ASSERT_EQUAL_DIMENSIONS(beta, dim<2>(1, 1));
    }

private:
    dim<2> size_{};
};


/**
 * A LinOpFactory represents a higher order mapping which transforms one
 * linear operator into another.
 *
 * In Ginkgo, every linear solver is viewed as a mapping. For example,
 * given an s.p.d linear system $Ax = b$, the solution $x = A^{-1}b$
 * can be computed using the CG method. This algorithm can be represented in
 * terms of linear operators and mappings between them as follows:
 *
 * -   A Cg::Factory is a higher order mapping which, given an input operator
 *     $A$, returns a new linear operator $A^{-1}$ stored in "CG
 *     format"
 * -   Storing the operator $A^{-1}$ in "CG format" means that the data
 *     structure used to store the operator is just a simple pointer to the
 *     original matrix $A$. The application $x = A^{-1}b$ of such an
 *     operator can then be implemented by solving the linear system
 *     $Ax = b$ using the CG method. This is achieved in code by having a
 *     special class for each of those "formats" (e.g. the "Cg" class defines
 *     such a format for the CG solver).
 *
 * Another example of a LinOpFactory is a preconditioner. A preconditioner for
 * a linear operator $A$ is a linear operator $M^{-1}$, which
 * approximates $A^{-1}$. In addition, it is stored in a way such that
 * both the data of $M^{-1}$ is cheap to compute from $A$, and the
 * operation $x = M^{-1}b$ can be computed quickly. These operators are
 * useful to accelerate the convergence of  Krylov solvers.
 * Thus, a preconditioner also fits into the LinOpFactory framework:
 *
 * -   The factory maps a linear operator $A$ into a preconditioner
 *     $M^{-1}$ which is stored in suitable format (e.g. as a product of
 *     two factors in case of ILU preconditioners).
 * -   The resulting linear operator implements the application operation
 *     $x = M^{-1}b$ depending on the format the preconditioner is stored
 *     in (e.g. as two triangular solves in case of ILU)
 *
 * Example: using CG in Ginkgo
 * ---------------------------
 *
 * ```c++
 * // Suppose A is a matrix, b a rhs vector, and x an initial guess
 * // Create a CG which runs for at most 1000 iterations, and stops after
 * // reducing the residual norm by 6 orders of magnitude
 * auto cg_factory = solver::Cg<>::build()
 *     .with_max_iters(1000)
 *     .with_rel_residual_goal(1e-6)
 *     .on(cuda);
 * // create a linear operator which represents the solver
 * auto cg = cg_factory->generate(A);
 * // solve the system
 * cg->apply(b, x);
 * ```
 *
 * @ingroup LinOp
 */
class LinOpFactory
    : public AbstractFactory<LinOp, std::shared_ptr<const LinOp>> {
public:
    using AbstractFactory<LinOp, std::shared_ptr<const LinOp>>::AbstractFactory;

    std::unique_ptr<LinOp> generate(std::shared_ptr<const LinOp> input) const
    {
        this->template log<log::Logger::linop_factory_generate_started>(
            this, input.get());
        const auto exec = this->get_executor();
        std::unique_ptr<LinOp> generated;
        if (input->get_executor() == exec) {
            generated = this->AbstractFactory::generate(input);
        } else {
            generated =
                this->AbstractFactory::generate(gko::clone(exec, input));
        }
        this->template log<log::Logger::linop_factory_generate_completed>(
            this, input.get(), generated.get());
        return generated;
    }
};


/**
 * Linear operators which support transposition should implement the
 * Transposable interface.
 *
 * It provides two functionalities, the normal transpose and the
 * conjugate transpose.
 *
 * The normal transpose returns the transpose of the linear operator without
 * changing any of its elements representing the operation, $B = A^{T}$.
 *
 * The conjugate transpose returns the conjugate of each of the elements and
 * additionally transposes the linear operator representing the operation, $B
 * = A^{H}$.
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
 * Linear operators which support permutation should implement the
 * Permutable interface.
 *
 * It provides functions to permute the rows and columns of a LinOp,
 * independently or symmetrically, and with a regular or inverted permutation.
 *
 * After a regular row permutation with permutation array `perm` the row `i` in
 * the output LinOp contains the row `perm[i]` from the input LinOp.
 * After an inverse row permutation, the row `perm[i]` in the output LinOp
 * contains the row `i` from the input LinOp.
 * Equivalently, after a column permutation, the output stores in column `i`
 * the column `perm[i]` from the input, and an inverse column permutation
 * stores in column `perm[i]` the column `i` from the input.
 * A symmetric permutation is functionally equivalent to calling
 * `as<Permutable>(A->row_permute(perm))->column_permute(perm)`, but the
 * implementation can provide better performance due to kernel fusion.
 *
 * Example: Permuting a Csr matrix:
 * ------------------------------------
 *
 * ```c++
 * //Permuting an object of LinOp type.
 * //The object you want to permute.
 * auto op = matrix::Csr::create(exec);
 * //Permute the object by first converting it to a Permutable type.
 * auto perm = op->row_permute(permutation_indices);
 * ```
 */
template <typename IndexType>
class Permutable {
public:
    virtual ~Permutable() = default;

    /**
     * Returns a LinOp representing the symmetric row and column permutation of
     * the Permutable object.
     * In the resulting LinOp, the entry at location `(i,j)` contains the input
     * value `(perm[i],perm[j])`.
     *
     * From the linear algebra perspective, with $P_{ij} = \delta_{i
     * \pi(i)}$, this represents the operation $P A P^T$.
     *
     * @param permutation_indices  the array of indices containing the
     *                             permutation order.
     *
     * @return a pointer to the new permuted object
     */
    virtual std::unique_ptr<LinOp> permute(
        const array<IndexType>* permutation_indices) const
    {
        return as<Permutable>(this->row_permute(permutation_indices))
            ->column_permute(permutation_indices);
    }

    /**
     * Returns a LinOp representing the symmetric inverse row and column
     * permutation of the Permutable object.
     * In the resulting LinOp, the entry at location `(perm[i],perm[j])`
     * contains the input value `(i,j)`.
     *
     * From the linear algebra perspective, with $P_{ij} = \delta_{i
     * \pi(i)}$, this represents the operation $P^{-1} A P^{-T}$.
     *
     * @param permutation_indices  the array of indices containing the
     *                             permutation order.
     *
     * @return a pointer to the new permuted object
     */
    virtual std::unique_ptr<LinOp> inverse_permute(
        const array<IndexType>* permutation_indices) const
    {
        return as<Permutable>(this->inverse_row_permute(permutation_indices))
            ->inverse_column_permute(permutation_indices);
    }

    /**
     * Returns a LinOp representing the row permutation of the Permutable
     * object.
     * In the resulting LinOp, the row `i` contains the input row `perm[i]`.
     *
     * From the linear algebra perspective, with $P_{ij} = \delta_{i
     * \pi(i)}$, this represents the operation $P A$.
     *
     * @param permutation_indices  the array of indices containing the
     *                             permutation order.
     *
     * @return a pointer to the new permuted object
     */
    virtual std::unique_ptr<LinOp> row_permute(
        const array<IndexType>* permutation_indices) const = 0;

    /**
     * Returns a LinOp representing the column permutation of the Permutable
     * object.
     * In the resulting LinOp, the column `i` contains the input column
     * `perm[i]`.
     *
     * From the linear algebra perspective, with $P_{ij} = \delta_{i
     * \pi(i)}$, this represents the operation $A P^T$.
     *
     * @param permutation_indices  the array of indices containing the
     *                             permutation order `perm`.
     *
     * @return a pointer to the new column permuted object
     */
    virtual std::unique_ptr<LinOp> column_permute(
        const array<IndexType>* permutation_indices) const = 0;

    /**
     * Returns a LinOp representing the row permutation of the inverse permuted
     * object.
     * In the resulting LinOp, the row `perm[i]` contains the input row `i`.
     *
     * From the linear algebra perspective, with $P_{ij} = \delta_{i
     * \pi(i)}$, this represents the operation $P^{-1} A$.
     *
     * @param permutation_indices  the array of indices containing the
     *                             permutation order `perm`.
     *
     * @return a pointer to the new inverse permuted object
     */
    virtual std::unique_ptr<LinOp> inverse_row_permute(
        const array<IndexType>* permutation_indices) const = 0;

    /**
     * Returns a LinOp representing the row permutation of the inverse permuted
     * object.
     * In the resulting LinOp, the column `perm[i]` contains the input column
     * `i`.
     *
     * From the linear algebra perspective, with $P_{ij} = \delta_{i
     * \pi(i)}$, this represents the operation $A P^{-T}$.
     *
     * @param permutation_indices  the array of indices containing the
     *                             permutation order `perm`.
     *
     * @return a pointer to the new inverse permuted object
     */
    virtual std::unique_ptr<LinOp> inverse_column_permute(
        const array<IndexType>* permutation_indices) const = 0;
};


/**
 * A LinOp implementing this interface can read its data from a matrix_data
 * structure.
 *
 * @ingroup LinOp
 */
template <typename ValueType, typename IndexType>
class ReadableFromMatrixData {
public:
    using value_type = ValueType;
    using index_type = IndexType;

    virtual ~ReadableFromMatrixData() = default;

    /**
     * Reads a matrix from a matrix_data structure.
     *
     * @param data  the matrix_data structure
     */
    virtual void read(const matrix_data<ValueType, IndexType>& data) = 0;

    /**
     * Reads a matrix from a matrix_assembly_data structure.
     *
     * @param data  the matrix_assembly_data structure
     */
    void read(const matrix_assembly_data<ValueType, IndexType>& data)
    {
        this->read(data.get_ordered_data());
    }

    /**
     * Reads a matrix from a device_matrix_data structure.
     *
     * @param data  the device_matrix_data structure.
     */
    virtual void read(const device_matrix_data<ValueType, IndexType>& data)
    {
        this->read(data.copy_to_host());
    }

    /**
     * Reads a matrix from a device_matrix_data structure.
     * The structure may be emptied by this function.
     *
     * @param data  the device_matrix_data structure.
     */
    virtual void read(device_matrix_data<ValueType, IndexType>&& data)
    {
        this->read(data.copy_to_host());
        data.empty_out();
    }
};


/**
 * A LinOp implementing this interface can write its data to a matrix_data
 * structure.
 *
 * @ingroup LinOp
 */
template <typename ValueType, typename IndexType>
class WritableToMatrixData {
public:
    using value_type = ValueType;
    using index_type = IndexType;

    virtual ~WritableToMatrixData() = default;

    /**
     * Writes a matrix to a matrix_data structure.
     *
     * @param data  the matrix_data structure
     */
    virtual void write(matrix_data<ValueType, IndexType>& data) const = 0;
};


/**
 * A LinOp implementing this interface can be preconditioned.
 *
 * @ingroup precond
 * @ingroup LinOp
 */
class Preconditionable {
public:
    virtual ~Preconditionable() = default;

    /**
     * Returns the preconditioner operator used by the Preconditionable.
     *
     * @return the preconditioner operator used by the Preconditionable
     */
    virtual std::shared_ptr<const LinOp> get_preconditioner() const
    {
        return preconditioner_;
    }

    /**
     * Sets the preconditioner operator used by the Preconditionable.
     *
     * @param new_precond  the new preconditioner operator used by the
     *                     Preconditionable
     */
    virtual void set_preconditioner(std::shared_ptr<const LinOp> new_precond)
    {
        preconditioner_ = new_precond;
    }

private:
    std::shared_ptr<const LinOp> preconditioner_{};
};


/**
 * The diagonal of a LinOp can be extracted. It will be implemented by
 * DiagonalExtractable<ValueType>, so the class does not need to implement it.
 * extract_diagonal_linop returns a linop which extracts the elements whose col
 * and row index are the same and stores the result in a min(nrows, ncols) x 1
 * dense matrix.
 *
 * @ingroup diagonal
 * @ingroup LinOp
 */
class DiagonalLinOpExtractable {
public:
    virtual ~DiagonalLinOpExtractable() = default;

    /**
     * Extracts the diagonal entries of the matrix into a vector.
     *
     * @return linop  the linop of diagonal format
     */
    virtual std::unique_ptr<LinOp> extract_diagonal_linop() const = 0;
};


/**
 * The diagonal of a LinOp implementing this interface can be extracted.
 * extract_diagonal extracts the elements whose col and row index are the
 * same and stores the result in a min(nrows, ncols) x 1 dense matrix.
 *
 * @ingroup LinOp
 */
template <typename ValueType>
class DiagonalExtractable : public DiagonalLinOpExtractable {
public:
    using value_type = ValueType;

    virtual ~DiagonalExtractable() = default;

    std::unique_ptr<LinOp> extract_diagonal_linop() const override;

    /**
     * Extracts the diagonal entries of the matrix into a vector.
     *
     * @param diag  the vector into which the diagonal will be written
     */
    virtual std::unique_ptr<matrix::Diagonal<ValueType>> extract_diagonal()
        const = 0;
};


/**
 * The AbsoluteComputable is an interface that allows to get the component wise
 * absolute of a LinOp. Use EnableAbsoluteComputation<AbsoluteLinOp> to
 * implement this interface.
 */
class AbsoluteComputable {
public:
    /**
     * Gets the absolute LinOp
     *
     * @return a pointer to the new absolute LinOp
     */
    virtual std::unique_ptr<LinOp> compute_absolute_linop() const = 0;

    /**
     * Compute absolute inplace on each element.
     */
    virtual void compute_absolute_inplace() = 0;
};


/**
 * The EnableAbsoluteComputation mixin provides the default implementations of
 * `compute_absolute_linop` and the absolute interface. `compute_absolute` gets
 * a new AbsoluteLinOp. `compute_absolute_inplace` applies absolute
 * inplace, so it still keeps the value_type of the class.
 *
 * @tparam AbsoluteLinOp  the absolute LinOp which is being returned
 *                        [CRTP parameter]
 *
 * @ingroup LinOp
 */
template <typename AbsoluteLinOp>
class EnableAbsoluteComputation : public AbsoluteComputable {
public:
    using absolute_type = AbsoluteLinOp;

    virtual ~EnableAbsoluteComputation() = default;

    std::unique_ptr<LinOp> compute_absolute_linop() const override
    {
        return this->compute_absolute();
    }

    /**
     * Gets the AbsoluteLinOp
     *
     * @return a pointer to the new absolute object
     */
    virtual std::unique_ptr<absolute_type> compute_absolute() const = 0;
};


/**
 * Adds the operation M <- a I + b M for matrix M, identity
 * operator I and scalars a and b, where M is the calling object.
 */
class ScaledIdentityAddable {
public:
    /**
     * Scales this and adds another scalar times the identity to it.
     *
     * @param a  Scalar to multiply the identity operator before adding.
     * @param b  Scalar to multiply this before adding the scaled identity to
     *           it.
     */
    void add_scaled_identity(ptr_param<const LinOp> const a,
                             ptr_param<const LinOp> const b)
    {
        GKO_ASSERT_IS_SCALAR(a);
        GKO_ASSERT_IS_SCALAR(b);
        auto ae = make_temporary_clone(as<LinOp>(this)->get_executor(), a);
        auto be = make_temporary_clone(as<LinOp>(this)->get_executor(), b);
        add_scaled_identity_impl(ae.get(), be.get());
    }

private:
    virtual void add_scaled_identity_impl(const LinOp* a, const LinOp* b) = 0;
};


/**
 * The EnableLinOp mixin can be used to provide sensible default implementations
 * of the majority of the LinOp and PolymorphicObject interface.
 *
 * The goal of the mixin is to facilitate the development of new LinOp, by
 * enabling the implementers to focus on the important parts of their operator,
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
 * 2.  Application of the LinOp: Implementers have to override the two
 *     overloads of the LinOp::apply_impl() virtual methods.
 *
 * @note  This mixin can't be used with concrete types that derive from
 *        experimental::distributed::DistributedBase. In that case use
 *        experimental::EnableDistributedLinOp instead.
 *
 * @tparam ConcreteLinOp  the concrete LinOp which is being implemented
 *                        [CRTP parameter]
 * @tparam PolymorphicBase  parent of ConcreteLinOp in the polymorphic
 *                          hierarchy, has to be a subclass of LinOp
 *
 * @ingroup LinOp
 */
template <typename ConcreteLinOp, typename PolymorphicBase = LinOp>
class EnableLinOp
    : public EnablePolymorphicObject<ConcreteLinOp, PolymorphicBase>,
      public EnablePolymorphicAssignment<ConcreteLinOp> {
public:
    using EnablePolymorphicObject<ConcreteLinOp,
                                  PolymorphicBase>::EnablePolymorphicObject;

    const ConcreteLinOp* apply(ptr_param<const LinOp> b,
                               ptr_param<LinOp> x) const
    {
        PolymorphicBase::apply(b, x);
        return self();
    }

    ConcreteLinOp* apply(ptr_param<const LinOp> b, ptr_param<LinOp> x)
    {
        PolymorphicBase::apply(b, x);
        return self();
    }

    const ConcreteLinOp* apply(ptr_param<const LinOp> alpha,
                               ptr_param<const LinOp> b,
                               ptr_param<const LinOp> beta,
                               ptr_param<LinOp> x) const
    {
        PolymorphicBase::apply(alpha, b, beta, x);
        return self();
    }

    ConcreteLinOp* apply(ptr_param<const LinOp> alpha, ptr_param<const LinOp> b,
                         ptr_param<const LinOp> beta, ptr_param<LinOp> x)
    {
        PolymorphicBase::apply(alpha, b, beta, x);
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
 *                          [CRTP parameter]
 * @tparam ConcreteLinOp  the concrete LinOp type which this factory produces,
 *                        needs to have a constructor which takes a
 *                        const ConcreteFactory *, and an
 *                        std::shared_ptr<const LinOp> as parameters.
 * @tparam ParametersType  a subclass of enable_parameters_type template which
 *                         defines all of the parameters of the factory
 * @tparam PolymorphicBase  parent of ConcreteFactory in the polymorphic
 *                          hierarchy, has to be a subclass of LinOpFactory
 *
 * @ingroup LinOp
 */
template <typename ConcreteFactory, typename ConcreteLinOp,
          typename ParametersType, typename PolymorphicBase = LinOpFactory>
using EnableDefaultLinOpFactory =
    EnableDefaultFactory<ConcreteFactory, ConcreteLinOp, ParametersType,
                         PolymorphicBase>;


/**
 * This macro will generate a default implementation of a LinOpFactory for the
 * LinOp subclass it is defined in.
 *
 * It is required to first call the macro #GKO_CREATE_FACTORY_PARAMETERS()
 * before this one in order to instantiate the parameters type first.
 *
 * The list of parameters for the factory should be defined in a code block
 * after the macro definition, and should contain a list of
 * GKO_FACTORY_PARAMETER_* declarations. The class should provide a constructor
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
 *         int GKO_FACTORY_PARAMETER_SCALAR(my_value, 5);
 *         // a factory parameter named `my_pair` of type `std::pair<int,int>`
 *         // and default value {5, 5}
 *         std::pair<int, int> GKO_FACTORY_PARAMETER_VECTOR(my_pair, 5, 5);
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
 * auto fact = MyLinOp::build().on(exec);
 * // create a operator using the factory:
 * auto my_op = fact->generate(gko::matrix::Identity::create(exec, 2));
 * std::cout << my_op->get_my_parameters().my_value;  // prints 5
 *
 * // create a factory with custom `my_value` parameter
 * auto fact = MyLinOp::build().with_my_value(0).on(exec);
 * // create a operator using the factory:
 * auto my_op = fact->generate(gko::matrix::Identity::create(exec, 2));
 * std::cout << my_op->get_my_parameters().my_value;  // prints 0
 * ```
 *
 * @note It is possible to combine both the #GKO_CREATE_FACTORY_PARAMETER_*()
 * macros with this one in a unique macro for class __templates__ (not with
 * regular classes). Splitting this into two distinct macros allows to use them
 * in all contexts. See <https://stackoverflow.com/q/50202718/9385966> for more
 * details.
 *
 * @param _lin_op  concrete operator for which the factory is to be created
 *                 [CRTP parameter]
 * @param _parameters_name  name of the parameters member in the class
 *                          (its type is `<_parameters_name>_type`, the
 *                          protected member's name is `<_parameters_name>_`,
 *                          and the public getter's name is
 *                          `get_<_parameters_name>()`)
 * @param _factory_name  name of the generated factory type
 *
 * @ingroup LinOp
 */
#define GKO_ENABLE_LIN_OP_FACTORY(_lin_op, _parameters_name, _factory_name)  \
public:                                                                      \
    const _parameters_name##_type& get_##_parameters_name() const            \
    {                                                                        \
        return _parameters_name##_;                                          \
    }                                                                        \
                                                                             \
    class _factory_name                                                      \
        : public ::gko::EnableDefaultLinOpFactory<_factory_name, _lin_op,    \
                                                  _parameters_name##_type> { \
        friend class ::gko::EnablePolymorphicObject<_factory_name,           \
                                                    ::gko::LinOpFactory>;    \
        friend class ::gko::enable_parameters_type<_parameters_name##_type,  \
                                                   _factory_name>;           \
        explicit _factory_name(std::shared_ptr<const ::gko::Executor> exec)  \
            : ::gko::EnableDefaultLinOpFactory<_factory_name, _lin_op,       \
                                               _parameters_name##_type>(     \
                  std::move(exec))                                           \
        {}                                                                   \
        explicit _factory_name(std::shared_ptr<const ::gko::Executor> exec,  \
                               const _parameters_name##_type& parameters)    \
            : ::gko::EnableDefaultLinOpFactory<_factory_name, _lin_op,       \
                                               _parameters_name##_type>(     \
                  std::move(exec), parameters)                               \
        {}                                                                   \
    };                                                                       \
    friend ::gko::EnableDefaultLinOpFactory<_factory_name, _lin_op,          \
                                            _parameters_name##_type>;        \
                                                                             \
                                                                             \
private:                                                                     \
    _parameters_name##_type _parameters_name##_;                             \
                                                                             \
public:                                                                      \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_LIN_OP_HPP_
