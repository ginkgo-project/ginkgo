// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_SOLVER_SOLVER_BASE_HPP_
#define GKO_PUBLIC_CORE_SOLVER_SOLVER_BASE_HPP_


#include <memory>
#include <type_traits>
#include <utility>


#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/solver/workspace.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/criterion.hpp>


GKO_BEGIN_DISABLE_DEPRECATION_WARNINGS


namespace gko {
namespace solver {


/**
 * Give a initial guess mode about the input of the apply method.
 */
enum class initial_guess_mode {
    /**
     * the input is zero
     */
    zero,
    /**
     * the input is right hand side
     */
    rhs,
    /**
     * the input is provided
     */
    provided
};


namespace multigrid {
namespace detail {


class MultigridState;


}  // namespace detail
}  // namespace multigrid


/**
 * ApplyWithInitialGuess provides a way to give the input guess for apply
 * function. All functionalities have the protected access specifier. It should
 * only be used internally.
 */
class ApplyWithInitialGuess {
protected:
    friend class multigrid::detail::MultigridState;

    /**
     * Applies a linear operator to a vector (or a sequence of vectors) with
     * initial guess statement.
     *
     * Performs the operation x = op(b) with a initial guess statement, where op
     * is this linear operator and the initial guess parameter will modify the
     * input vector to the requested the initial guess mode (See
     * initial_guess_mode).
     *
     * @param b  the input vector(s) on which the operator is applied
     * @param x  the output vector(s) where the result is stored
     * @param guess  the input guess to handle the input vector(s)
     */
    virtual void apply_with_initial_guess(const LinOp* b, LinOp* x,
                                          initial_guess_mode guess) const = 0;

    void apply_with_initial_guess(ptr_param<const LinOp> b, ptr_param<LinOp> x,
                                  initial_guess_mode guess) const
    {
        apply_with_initial_guess(b.get(), x.get(), guess);
    }

    /**
     * Performs the operation x = alpha * op(b) + beta * x with a initial guess
     * statement, where op is this linear operator and the initial guess
     * parameter will modify the input vector to the requested the initial guess
     * mode (See initial_guess_mode) .
     *
     * @param alpha  scaling of the result of op(b)
     * @param b  vector(s) on which the operator is applied
     * @param beta  scaling of the input x
     * @param x  output vector(s)
     * @param guess  the input guess to handle the input vector(s)
     */
    virtual void apply_with_initial_guess(const LinOp* alpha, const LinOp* b,
                                          const LinOp* beta, LinOp* x,
                                          initial_guess_mode guess) const = 0;


    void apply_with_initial_guess(ptr_param<const LinOp> alpha,
                                  ptr_param<const LinOp> b,
                                  ptr_param<const LinOp> beta,
                                  ptr_param<LinOp> x,
                                  initial_guess_mode guess) const
    {
        apply_with_initial_guess(alpha.get(), b.get(), beta.get(), x.get(),
                                 guess);
    }

    /**
     * Get the default initial guess
     *
     * @return default initial guess
     */
    initial_guess_mode get_default_initial_guess() const { return guess_; }

    /**
     * ApplyWithInitialGuess constructor.
     *
     * @param guess  the input guess whose default is
     * initial_guess_mode::provided
     */
    explicit ApplyWithInitialGuess(
        initial_guess_mode guess = initial_guess_mode::provided)
        : guess_(guess)
    {}

    /**
     * set the default initial guess
     *
     * @param guess  the initial guess
     */
    void set_default_initial_guess(initial_guess_mode guess) { guess_ = guess; }

private:
    initial_guess_mode guess_;
};


/**
 * EnableApplyWithInitialGuess providing default operation for
 * ApplyWithInitialGuess with correct validation and log. It ensures that
 * vectors of apply_with_initial_guess will always have the same executor as the
 * object this mixin is used in, creating a clone on the correct executor if
 * necessary.
 *
 * @tparam DerivedType  The type that this Mixin is used in. It must provide
 *                      get_size() and get_executor() functions that return
 *                      correctly initialized values and the logger
 *                      functionality.
 */
template <typename DerivedType>
class EnableApplyWithInitialGuess : public ApplyWithInitialGuess {
protected:
    friend class multigrid::detail::MultigridState;

    explicit EnableApplyWithInitialGuess(
        initial_guess_mode guess = initial_guess_mode::provided)
        : ApplyWithInitialGuess(guess)
    {}

    /**
     * @copydoc apply_with_initial_guess(const LinOp*, LinOp*,
     *          initial_guess_mode)
     */
    void apply_with_initial_guess(const LinOp* b, LinOp* x,
                                  initial_guess_mode guess) const override
    {
        self()->template log<log::Logger::linop_apply_started>(self(), b, x);
        auto exec = self()->get_executor();
        GKO_ASSERT_CONFORMANT(self(), b);
        GKO_ASSERT_EQUAL_ROWS(self(), x);
        GKO_ASSERT_EQUAL_COLS(b, x);
        this->apply_with_initial_guess_impl(make_temporary_clone(exec, b).get(),
                                            make_temporary_clone(exec, x).get(),
                                            guess);
        self()->template log<log::Logger::linop_apply_completed>(self(), b, x);
    }

    /**
     * @copydoc apply_with_initial_guess(const LinOp*,const LinOp*,const LinOp*,
     *          LinOp*, initial_guess_mode)
     */
    void apply_with_initial_guess(const LinOp* alpha, const LinOp* b,
                                  const LinOp* beta, LinOp* x,
                                  initial_guess_mode guess) const override
    {
        self()->template log<log::Logger::linop_advanced_apply_started>(
            self(), alpha, b, beta, x);
        auto exec = self()->get_executor();
        GKO_ASSERT_CONFORMANT(self(), b);
        GKO_ASSERT_EQUAL_ROWS(self(), x);
        GKO_ASSERT_EQUAL_COLS(b, x);
        GKO_ASSERT_EQUAL_DIMENSIONS(alpha, dim<2>(1, 1));
        GKO_ASSERT_EQUAL_DIMENSIONS(beta, dim<2>(1, 1));
        this->apply_with_initial_guess_impl(
            make_temporary_clone(exec, alpha).get(),
            make_temporary_clone(exec, b).get(),
            make_temporary_clone(exec, beta).get(),
            make_temporary_clone(exec, x).get(), guess);
        self()->template log<log::Logger::linop_advanced_apply_completed>(
            self(), alpha, b, beta, x);
    }

    // TODO: should we provide the default implementation?
    /**
     * The class should override this method and must modify the input vectors
     * according to the initial_guess_mode
     */
    virtual void apply_with_initial_guess_impl(
        const LinOp* b, LinOp* x, initial_guess_mode guess) const = 0;

    /**
     * The class should override this method and must modify the input vectors
     * according to the initial_guess_mode
     */
    virtual void apply_with_initial_guess_impl(
        const LinOp* alpha, const LinOp* b, const LinOp* beta, LinOp* x,
        initial_guess_mode guess) const = 0;

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


namespace detail {


/**
 * A LinOp implementing this interface stores a system matrix.
 *
 * @note This class will replace SolverBase in a future release
 *
 * @ingroup solver
 * @ingroup LinOp
 */
class SolverBaseLinOp {
public:
    SolverBaseLinOp(std::shared_ptr<const Executor> exec)
        : workspace_{std::move(exec)}
    {}

    virtual ~SolverBaseLinOp() = default;

    /**
     * Returns the system matrix used by the solver.
     *
     * @return the system matrix operator used by the solver
     */
    std::shared_ptr<const LinOp> get_system_matrix() const
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
    void set_system_matrix_base(std::shared_ptr<const LinOp> system_matrix)
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

    template <typename LinOpType>
    LinOpType* create_workspace_op_with_type_of(int vector_id,
                                                const LinOpType* vec,
                                                dim<2> global_size,
                                                dim<2> local_size) const
    {
        return workspace_.template create_or_get_op<LinOpType>(
            vector_id,
            [&] {
                return LinOpType::create_with_type_of(
                    vec, workspace_.get_executor(), global_size, local_size,
                    local_size[1]);
            },
            typeid(*vec), global_size, local_size[1]);
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

    std::shared_ptr<const LinOp> system_matrix_;
};


}  // namespace detail


template <typename MatrixType>
class
    // clang-format off
    GKO_DEPRECATED("This class will be replaced by the template-less detail::SolverBaseLinOp in a future release") SolverBase
    // clang-format on
    : public detail::SolverBaseLinOp {
public:
    using detail::SolverBaseLinOp::SolverBaseLinOp;

    /**
     * Returns the system matrix, with its concrete type, used by the
     * solver.
     *
     * @return the system matrix operator, with its concrete type, used by
     * the solver
     */
    std::shared_ptr<const MatrixType> get_system_matrix() const
    {
        return std::dynamic_pointer_cast<const MatrixType>(
            SolverBaseLinOp::get_system_matrix());
    }

protected:
    void set_system_matrix_base(std::shared_ptr<const MatrixType> system_matrix)
    {
        SolverBaseLinOp::set_system_matrix_base(std::move(system_matrix));
    }
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
 * allows applying with a guess.
 *
 * @tparam DerivedType  the CRTP type that derives from this
 *
 * @ingroup solver
 * @ingroup LinOp
 */
template <typename DerivedType>
class EnableIterativeBase : public IterativeBase {
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


template <typename Parameters, typename Factory>
struct enable_iterative_solver_factory_parameters
    : enable_parameters_type<Parameters, Factory> {
    /**
     * Stopping criteria to be used by the solver.
     */
    std::vector<std::shared_ptr<const stop::CriterionFactory>>
        GKO_DEFERRED_FACTORY_VECTOR_PARAMETER(criteria);
};


template <typename Parameters, typename Factory>
struct enable_preconditioned_iterative_solver_factory_parameters
    : enable_iterative_solver_factory_parameters<Parameters, Factory> {
    /**
     * The preconditioner to be used by the iterative solver. By default, no
     * preconditioner is used.
     */
    std::shared_ptr<const LinOpFactory> GKO_DEFERRED_FACTORY_PARAMETER(
        preconditioner);

    /**
     * Already generated preconditioner. If one is provided, the factory
     * `preconditioner` will be ignored.
     */
    std::shared_ptr<const LinOp> GKO_FACTORY_PARAMETER_SCALAR(
        generated_preconditioner, nullptr);
};


}  // namespace solver
}  // namespace gko


GKO_END_DISABLE_DEPRECATION_WARNINGS


#endif  // GKO_PUBLIC_CORE_SOLVER_SOLVER_BASE_HPP_
