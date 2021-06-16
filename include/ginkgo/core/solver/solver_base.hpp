/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/criterion.hpp>


namespace gko {


namespace solver {


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

    EnablePreconditionable &operator=(const EnablePreconditionable &other)
    {
        if (&other != this) {
            set_preconditioner(other.get_preconditioner());
        }
        return *this;
    }

    EnablePreconditionable &operator=(EnablePreconditionable &&other)
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

    EnablePreconditionable(const EnablePreconditionable &other)
    {
        *this = other;
    }

    EnablePreconditionable(EnablePreconditionable &&other)
    {
        *this = std::move(other);
    }

private:
    DerivedType *self() { return static_cast<DerivedType *>(this); }

    const DerivedType *self() const
    {
        return static_cast<const DerivedType *>(this);
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

protected:
    void set_system_matrix(std::shared_ptr<const MatrixType> system_matrix)
    {
        system_matrix_ = std::move(system_matrix);
    }

    std::shared_ptr<const MatrixType> system_matrix_;
};


/**
 * A LinOp implementing this interface stores a system matrix.
 *
 * @ingroup solver
 * @ingroup LinOp
 */
template <typename DerivedType, typename MatrixType = LinOp>
class EnableSolverBase : public SolverBase<MatrixType> {
public:
    EnableSolverBase &operator=(const EnableSolverBase &other)
    {
        if (&other != this) {
            set_system_matrix(other.get_system_matrix());
        }
        return *this;
    }

    EnableSolverBase &operator=(EnableSolverBase &&other)
    {
        if (&other != this) {
            set_system_matrix(other.get_system_matrix());
            other.set_system_matrix(nullptr);
        }
        return *this;
    }

    EnableSolverBase() = default;

    EnableSolverBase(std::shared_ptr<const MatrixType> system_matrix)
    {
        set_system_matrix(std::move(system_matrix));
    }

    EnableSolverBase(const EnableSolverBase &other) { *this = other; }

    EnableSolverBase(EnableSolverBase &&other) { *this = std::move(other); }

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
        SolverBase<MatrixType>::set_system_matrix(new_system_matrix);
    }

private:
    DerivedType *self() { return static_cast<DerivedType *>(this); }

    const DerivedType *self() const
    {
        return static_cast<const DerivedType *>(this);
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


template <typename DerivedType>
class EnableIterativeBase : public IterativeBase {
public:
    EnableIterativeBase &operator=(const EnableIterativeBase &other)
    {
        if (&other != this) {
            set_stop_criterion_factory(other.get_stop_criterion_factory());
        }
        return *this;
    }

    EnableIterativeBase &operator=(EnableIterativeBase &&other)
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

    EnableIterativeBase(const EnableIterativeBase &other) { *this = other; }

    EnableIterativeBase(EnableIterativeBase &&other)
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
    DerivedType *self() { return static_cast<DerivedType *>(this); }

    const DerivedType *self() const
    {
        return static_cast<const DerivedType *>(this);
    }
};


/**
 * A LinOp implementing this interface stores a system matrix and stopping
 * criterion factory.
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
        const FactoryParameters &params)
        : EnablePreconditionedIterativeSolver{
              system_matrix, stop::combine(params.criteria),
              generate_preconditioner(system_matrix, params)}
    {}

private:
    template <typename FactoryParameters>
    static std::shared_ptr<const LinOp> generate_preconditioner(
        std::shared_ptr<const LinOp> system_matrix,
        const FactoryParameters &params)
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
