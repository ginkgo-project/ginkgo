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

#ifndef GKO_PUBLIC_CORE_SOLVER_SOLVER_BASE_HPP_
#define GKO_PUBLIC_CORE_SOLVER_SOLVER_BASE_HPP_


#include <memory>
#include <utility>


#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/solver/workspace.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/criterion.hpp>


namespace gko {
namespace solver {


/**
 * Give a hint about the input of the apply method.
 */
enum class input_hint {
    /**
     * the input is zero
     */
    zero,
    /**
     * the input is right hand side
     */
    rhs,
    /**
     * the input is given
     */
    given
};


namespace {


inline void fill_zero(LinOp* input)
{
    if (auto dense = dynamic_cast<matrix::Dense<float>*>(input)) {
        dense->fill(gko::zero<float>());
    } else if (auto dense = dynamic_cast<matrix::Dense<double>*>(input)) {
        dense->fill(gko::zero<double>());
    } else if (auto dense =
                   dynamic_cast<matrix::Dense<std::complex<float>>*>(input)) {
        dense->fill(gko::zero<std::complex<float>>());
    } else if (auto dense =
                   dynamic_cast<matrix::Dense<std::complex<double>>*>(input)) {
        dense->fill(gko::zero<std::complex<double>>());
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}


}  // namespace


class ApplyHint {
public:
    ApplyHint* apply_hint(const LinOp* b, LinOp* x, input_hint hint)
    {
        this->apply_impl(b, x, hint);
        return this;
    }

    const ApplyHint* apply_hint(const LinOp* b, LinOp* x, input_hint hint) const
    {
        this->apply_impl(b, x, hint);
        return this;
    }

    ApplyHint* apply_hint(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                          LinOp* x, input_hint hint)
    {
        this->apply_impl(alpha, b, beta, x, hint);
        return this;
    }

    const ApplyHint* apply_hint(const LinOp* alpha, const LinOp* b,
                                const LinOp* beta, LinOp* x,
                                input_hint hint) const
    {
        this->apply_impl(alpha, b, beta, x, hint);
        return this;
    }

protected:
    virtual void apply_impl(const LinOp* b, LinOp* x, input_hint hint) const
    {
        if (hint == input_hint::zero) {
            fill_zero(x);
        } else if (hint == input_hint::rhs) {
            x->copy_from(b);
        }
        this->apply_impl(b, x);
    }

    virtual void apply_impl(const LinOp* alpha, const LinOp* b,
                            const LinOp* beta, LinOp* x, input_hint hint) const
    {
        if (hint == input_hint::zero) {
            fill_zero(x);
        } else if (hint == input_hint::rhs) {
            x->copy_from(b);
        }
        this->apply_impl(b, x);
    }

    // override at the same time when overriden
    virtual void apply_impl(const LinOp* b, LinOp* x) const = 0;
    virtual void apply_impl(const LinOp* alpha, const LinOp* b,
                            const LinOp* beta, LinOp* x) const = 0;
};

template <typename DerivedType>
class EnableApplyHint : public ApplyHint {
public:
    DerivedType* apply_hint(const LinOp* b, LinOp* x, input_hint hint)
    {
        ApplyHint::apply_hint(b, x, hint);
        return self();
    }

    const DerivedType* apply_hint(const LinOp* b, LinOp* x,
                                  input_hint hint) const
    {
        ApplyHint::apply_hint(b, x, hint);
        return self();
    }

    DerivedType* apply_hint(const LinOp* alpha, const LinOp* b,
                            const LinOp* beta, LinOp* x, input_hint hint)
    {
        ApplyHint::apply_hint(alpha, b, beta, x, hint);
        return self();
    }

    const DerivedType* apply_hint(const LinOp* alpha, const LinOp* b,
                                  const LinOp* beta, LinOp* x,
                                  input_hint hint) const
    {
        ApplyHint::apply_hint(alpha, b, beta, x, hint);
        return self();
    }

protected:
    GKO_ENABLE_SELF(DerivedType);
};

/**
 * Traits class providing information on the type and location of workspace
 * vectors inside a solver.
 */
template <typename Solver>
struct workspace_traits {
    // number of vectors used by this workspace
    static int num_vectors(const Solver&) { return 0; }
    // number of arrays used by this workspace
    static int num_arrays(const Solver&) { return 0; }
    // array containing the num_vectors names for the workspace vectors
    static std::vector<std::string> op_names(const Solver&) { return {}; }
    // array containing the num_arrays names for the workspace vectors
    static std::vector<std::string> array_names(const Solver&) { return {}; }
    // array containing all scalar vectors (independent of problem size)
    static std::vector<int> scalars(const Solver&) { return {}; }
    // array containing all vectors (dependent on problem size)
    static std::vector<int> vectors(const Solver&) { return {}; }
};


/**
 * Mixin providing default operation for Preconditionable with correct value
 * semantics. It ensures that the preconditioner stored in this class will
 * always have the same executor as the object this mixin is used in, creating a
 * clone on the correct executor if necessary.
 *
 * @tparam DerivedType  The type that this Mixin is used in. It must provide
 *                      get_size() and get_executor() functions that return
 *                      correctly initialized values when the
 *                      EnablePreconditionable constructor is called, i.e. the
 *                      constructor must be provided by a base class added
 *                      before EnablePreconditionable, since the member
 *                      initialization order also applying to multiple
 *                      inheritance.
 */
template <typename DerivedType>
class EnablePreconditionable : public Preconditionable {
public:
    /**
     * Sets the preconditioner operator used by the Preconditionable.
     *
     * @param new_precond  the new preconditioner operator used by the
     *                     Preconditionable
     */
    void set_preconditioner(std::shared_ptr<const LinOp> new_precond) override
    {
        auto exec = self()->get_executor();
        if (new_precond) {
            GKO_ASSERT_EQUAL_DIMENSIONS(self(), new_precond);
            GKO_ASSERT_IS_SQUARE_MATRIX(new_precond);
            if (new_precond->get_executor() != exec) {
                new_precond = gko::clone(exec, new_precond);
            }
        }
        Preconditionable::set_preconditioner(new_precond);
    }

    /**
     * Creates a shallow copy of the provided preconditioner, clones it onto
     * this executor if executors don't match.
     */
    EnablePreconditionable& operator=(const EnablePreconditionable& other)
    {
        if (&other != this) {
            set_preconditioner(other.get_preconditioner());
        }
        return *this;
    }

    /**
     * Moves the provided preconditioner, clones it onto this executor if
     * executors don't match. The moved-from object has a nullptr
     * preconditioner.
     */
    EnablePreconditionable& operator=(EnablePreconditionable&& other)
    {
        if (&other != this) {
            set_preconditioner(other.get_preconditioner());
            other.set_preconditioner(nullptr);
        }
        return *this;
    }

    EnablePreconditionable() = default;

    EnablePreconditionable(std::shared_ptr<const LinOp> preconditioner)
    {
        set_preconditioner(std::move(preconditioner));
    }

    /**
     * Creates a shallow copy of the provided preconditioner.
     */
    EnablePreconditionable(const EnablePreconditionable& other)
    {
        *this = other;
    }

    /**
     * Moves the provided preconditioner. The moved-from object has a nullptr
     * preconditioner.
     */
    EnablePreconditionable(EnablePreconditionable&& other)
    {
        *this = std::move(other);
    }

private:
    DerivedType* self() { return static_cast<DerivedType*>(this); }

    const DerivedType* self() const
    {
        return static_cast<const DerivedType*>(this);
    }
};


/**
 * A LinOp implementing this interface stores a system matrix.
 *
 * @ingroup solver
 * @ingroup LinOp
 */
template <typename MatrixType = LinOp>
class SolverBase {
public:
    SolverBase(std::shared_ptr<const Executor> exec)
        : workspace_{std::move(exec)}
    {}

    virtual ~SolverBase() = default;

    /**
     * Returns the system matrix used by the solver.
     *
     * @return the system matrix operator used by the solver
     */
    std::shared_ptr<const MatrixType> get_system_matrix() const
    {
        return system_matrix_;
    }

    const LinOp* get_workspace_op(int vector_id) const
    {
        return workspace_.get_op(vector_id);
    }

    virtual int get_num_workspace_ops() const { return 0; }

    virtual std::vector<std::string> get_workspace_op_names() const
    {
        return {};
    }

    /**
     * Returns the IDs of all scalars (workspace vectors with
     * system dimension-independent size, usually 1 x num_rhs).
     */
    virtual std::vector<int> get_workspace_scalars() const { return {}; }

    /**
     * Returns the IDs of all vectors (workspace vectors with system
     * dimension-dependent size, usually system_matrix_size x num_rhs).
     */
    virtual std::vector<int> get_workspace_vectors() const { return {}; }

protected:
    void set_system_matrix_base(std::shared_ptr<const MatrixType> system_matrix)
    {
        system_matrix_ = std::move(system_matrix);
    }

    void set_workspace_size(int num_operators, int num_arrays) const
    {
        workspace_.set_size(num_operators, num_arrays);
    }

    template <typename LinOpType>
    LinOpType* create_workspace_op(int vector_id, gko::dim<2> size) const
    {
        return workspace_.template create_or_get_op<LinOpType>(
            vector_id,
            [&] {
                return LinOpType::create(this->workspace_.get_executor(), size);
            },
            typeid(LinOpType), size, size[1]);
    }

    template <typename LinOpType>
    LinOpType* create_workspace_op_with_config_of(int vector_id,
                                                  const LinOpType* vec) const
    {
        return workspace_.template create_or_get_op<LinOpType>(
            vector_id, [&] { return LinOpType::create_with_config_of(vec); },
            typeid(*vec), vec->get_size(), vec->get_stride());
    }

    template <typename LinOpType>
    LinOpType* create_workspace_op_with_type_of(int vector_id,
                                                const LinOpType* vec,
                                                dim<2> size) const
    {
        return workspace_.template create_or_get_op<LinOpType>(
            vector_id,
            [&] {
                return LinOpType::create_with_type_of(
                    vec, workspace_.get_executor(), size, size[1]);
            },
            typeid(*vec), size, size[1]);
    }

    template <typename ValueType>
    matrix::Dense<ValueType>* create_workspace_scalar(int vector_id,
                                                      size_type size) const
    {
        return workspace_.template create_or_get_op<matrix::Dense<ValueType>>(
            vector_id,
            [&] {
                return matrix::Dense<ValueType>::create(
                    workspace_.get_executor(), dim<2>{1, size});
            },
            typeid(matrix::Dense<ValueType>), gko::dim<2>{1, size}, size);
    }

    template <typename ValueType>
    array<ValueType>& create_workspace_array(int array_id, size_type size) const
    {
        return workspace_.template create_or_get_array<ValueType>(array_id,
                                                                  size);
    }

    template <typename ValueType>
    array<ValueType>& create_workspace_array(int array_id) const
    {
        return workspace_.template init_or_get_array<ValueType>(array_id);
    }

private:
    mutable detail::workspace workspace_;

    std::shared_ptr<const MatrixType> system_matrix_;
};


/**
 * A LinOp deriving from this CRTP class stores a system matrix.
 *
 * @tparam DerivedType  the CRTP type that derives from this
 * @tparam MatrixType  the concrete matrix type to be stored as system_matrix
 *
 * @ingroup solver
 * @ingroup LinOp
 */
template <typename DerivedType, typename MatrixType = LinOp>
class EnableSolverBase : public SolverBase<MatrixType> {
public:
    /**
     * Creates a shallow copy of the provided system matrix, clones it onto
     * this executor if executors don't match.
     */
    EnableSolverBase& operator=(const EnableSolverBase& other)
    {
        if (&other != this) {
            set_system_matrix(other.get_system_matrix());
        }
        return *this;
    }

    /**
     * Moves the provided system matrix, clones it onto this executor if
     * executors don't match. The moved-from object has a nullptr system matrix.
     */
    EnableSolverBase& operator=(EnableSolverBase&& other)
    {
        if (&other != this) {
            set_system_matrix(other.get_system_matrix());
            other.set_system_matrix(nullptr);
        }
        return *this;
    }

    EnableSolverBase() : SolverBase<MatrixType>{self()->get_executor()} {}

    EnableSolverBase(std::shared_ptr<const MatrixType> system_matrix)
        : SolverBase<MatrixType>{self()->get_executor()}
    {
        set_system_matrix(std::move(system_matrix));
    }

    /**
     * Creates a shallow copy of the provided system matrix.
     */
    EnableSolverBase(const EnableSolverBase& other)
        : SolverBase<MatrixType>{other.self()->get_executor()}
    {
        *this = other;
    }

    /**
     * Moves the provided system matrix. The moved-from object has a nullptr
     * system matrix.
     */
    EnableSolverBase(EnableSolverBase&& other)
        : SolverBase<MatrixType>{other.self()->get_executor()}
    {
        *this = std::move(other);
    }

    int get_num_workspace_ops() const override
    {
        using traits = workspace_traits<DerivedType>;
        return traits::num_vectors(*self());
    }

    std::vector<std::string> get_workspace_op_names() const override
    {
        using traits = workspace_traits<DerivedType>;
        return traits::op_names(*self());
    }

    /**
     * Returns the IDs of all scalars (workspace vectors with
     * system dimension-independent size, usually 1 x num_rhs).
     */
    std::vector<int> get_workspace_scalars() const override
    {
        using traits = workspace_traits<DerivedType>;
        return traits::scalars(*self());
    }

    /**
     * Returns the IDs of all vectors (workspace vectors with system
     * dimension-dependent size, usually system_matrix_size x num_rhs).
     */
    std::vector<int> get_workspace_vectors() const override
    {
        using traits = workspace_traits<DerivedType>;
        return traits::vectors(*self());
    }

protected:
    void set_system_matrix(std::shared_ptr<const MatrixType> new_system_matrix)
    {
        auto exec = self()->get_executor();
        if (new_system_matrix) {
            GKO_ASSERT_EQUAL_DIMENSIONS(self(), new_system_matrix);
            GKO_ASSERT_IS_SQUARE_MATRIX(new_system_matrix);
            if (new_system_matrix->get_executor() != exec) {
                new_system_matrix = gko::clone(exec, new_system_matrix);
            }
        }
        this->set_system_matrix_base(new_system_matrix);
    }

    void setup_workspace() const
    {
        using traits = workspace_traits<DerivedType>;
        this->set_workspace_size(traits::num_vectors(*self()),
                                 traits::num_arrays(*self()));
    }

private:
    DerivedType* self() { return static_cast<DerivedType*>(this); }

    const DerivedType* self() const
    {
        return static_cast<const DerivedType*>(this);
    }
};


/**
 * A LinOp implementing this interface stores a stopping criterion factory.
 *
 * @ingroup solver
 * @ingroup LinOp
 */
class IterativeBase {
public:
    /**
     * Gets the stopping criterion factory of the solver.
     *
     * @return the stopping criterion factory
     */
    std::shared_ptr<const stop::CriterionFactory> get_stop_criterion_factory()
        const
    {
        return stop_factory_;
    }

    /**
     * Sets the stopping criterion of the solver.
     *
     * @param other  the new stopping criterion factory
     */
    virtual void set_stop_criterion_factory(
        std::shared_ptr<const stop::CriterionFactory> new_stop_factory)
    {
        stop_factory_ = new_stop_factory;
    }

private:
    std::shared_ptr<const stop::CriterionFactory> stop_factory_;
};


/**
 * A LinOp deriving from this CRTP class stores a stopping criterion factory and
 * allows applying with a hint.
 *
 * @tparam DerivedType  the CRTP type that derives from this
 *
 * @ingroup solver
 * @ingroup LinOp
 */
template <typename DerivedType>
class EnableIterativeBase : public IterativeBase,
                            public EnableApplyHint<DerivedType> {
public:
    /**
     * Creates a shallow copy of the provided stopping criterion, clones it onto
     * this executor if executors don't match.
     */
    EnableIterativeBase& operator=(const EnableIterativeBase& other)
    {
        if (&other != this) {
            set_stop_criterion_factory(other.get_stop_criterion_factory());
        }
        return *this;
    }

    /**
     * Moves the provided stopping criterion, clones it onto this executor if
     * executors don't match. The moved-from object has a nullptr
     * stopping criterion.
     */
    EnableIterativeBase& operator=(EnableIterativeBase&& other)
    {
        if (&other != this) {
            set_stop_criterion_factory(other.get_stop_criterion_factory());
            other.set_stop_criterion_factory(nullptr);
        }
        return *this;
    }

    EnableIterativeBase() = default;

    EnableIterativeBase(
        std::shared_ptr<const stop::CriterionFactory> stop_factory)
    {
        set_stop_criterion_factory(std::move(stop_factory));
    }

    /**
     * Creates a shallow copy of the provided stopping criterion.
     */
    EnableIterativeBase(const EnableIterativeBase& other) { *this = other; }

    /**
     * Moves the provided stopping criterion. The moved-from object has a
     * nullptr stopping criterion.
     */
    EnableIterativeBase(EnableIterativeBase&& other)
    {
        *this = std::move(other);
    }

    void set_stop_criterion_factory(
        std::shared_ptr<const stop::CriterionFactory> new_stop_factory) override
    {
        auto exec = self()->get_executor();
        if (new_stop_factory && new_stop_factory->get_executor() != exec) {
            new_stop_factory = gko::clone(exec, new_stop_factory);
        }
        IterativeBase::set_stop_criterion_factory(new_stop_factory);
    }

private:
    DerivedType* self() { return static_cast<DerivedType*>(this); }

    const DerivedType* self() const
    {
        return static_cast<const DerivedType*>(this);
    }
};


/**
 * A LinOp implementing this interface stores a system matrix and stopping
 * criterion factory.
 *
 * @tparam ValueType  the value type that iterative solver uses for its vectors
 * @tparam DerivedType  the CRTP type that derives from this
 *
 * @ingroup solver
 * @ingroup LinOp
 */
template <typename ValueType, typename DerivedType>
class EnablePreconditionedIterativeSolver
    : public EnableSolverBase<DerivedType>,
      public EnableIterativeBase<DerivedType>,
      public EnablePreconditionable<DerivedType> {
public:
    EnablePreconditionedIterativeSolver() = default;

    EnablePreconditionedIterativeSolver(
        std::shared_ptr<const LinOp> system_matrix,
        std::shared_ptr<const stop::CriterionFactory> stop_factory,
        std::shared_ptr<const LinOp> preconditioner)
        : EnableSolverBase<DerivedType>(std::move(system_matrix)),
          EnableIterativeBase<DerivedType>{std::move(stop_factory)},
          EnablePreconditionable<DerivedType>{std::move(preconditioner)}
    {}

    template <typename FactoryParameters>
    EnablePreconditionedIterativeSolver(
        std::shared_ptr<const LinOp> system_matrix,
        const FactoryParameters& params)
        : EnablePreconditionedIterativeSolver{
              system_matrix, stop::combine(params.criteria),
              generate_preconditioner(system_matrix, params)}
    {}

private:
    template <typename FactoryParameters>
    static std::shared_ptr<const LinOp> generate_preconditioner(
        std::shared_ptr<const LinOp> system_matrix,
        const FactoryParameters& params)
    {
        if (params.generated_preconditioner) {
            return params.generated_preconditioner;
        } else if (params.preconditioner) {
            return params.preconditioner->generate(system_matrix);
        } else {
            return matrix::Identity<ValueType>::create(
                system_matrix->get_executor(), system_matrix->get_size());
        }
    }
};


}  // namespace solver
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_SOLVER_SOLVER_BASE_HPP_
