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

#ifndef GKO_CORE_LOG_RECORD_HPP_
#define GKO_CORE_LOG_RECORD_HPP_


#include "core/log/logger.hpp"


#include <deque>
#include <memory>


#include "core/matrix/dense.hpp"
#include "core/stop/criterion.hpp"


namespace gko {
namespace log {


/**
 * Struct representing iteration complete related data
 */
struct iteration_complete_data {
    std::unique_ptr<const LinOp> solver;
    const size_type num_iterations;
    std::unique_ptr<const LinOp> residual;
    std::unique_ptr<const LinOp> solution;
    std::unique_ptr<const LinOp> residual_norm;

    iteration_complete_data(const LinOp *solver, const size_type num_iterations,
                            const LinOp *residual = nullptr,
                            const LinOp *solution = nullptr,
                            const LinOp *residual_norm = nullptr)
        : num_iterations{num_iterations}
    {
        /* TODO: remove this solver part */
        if (solver != nullptr) {
            this->solver = solver->clone();
        }
        if (residual != nullptr) {
            this->residual = residual->clone();
        }
        if (solution != nullptr) {
            this->solution = solution->clone();
        }
        if (residual_norm != nullptr) {
            this->residual_norm = residual_norm->clone();
        }
    }
};


/**
 * Struct representing Executor related data
 */
struct executor_data {
    const Executor *exec;
    const size_type num_bytes;
    const uintptr location;
};


/**
 * Struct representing Operator related data
 */
struct operation_data {
    const Executor *exec;
    const Operation *operation;
};


/**
 * Struct representing PolymorphicObject related data
 */
struct polymorphic_object_data {
    const Executor *exec;
    std::unique_ptr<const PolymorphicObject> input;
    std::unique_ptr<const PolymorphicObject> output;  // optional

    polymorphic_object_data(const Executor *exec,
                            const PolymorphicObject *input,
                            const PolymorphicObject *output = nullptr)
        : exec{exec}
    {
        this->input = input->clone();
        if (output != nullptr) {
            this->output = output->clone();
        }
    }
};


/**
 * Struct representing LinOp related data
 */
struct linop_data {
    std::unique_ptr<const LinOp> A;
    std::unique_ptr<const LinOp> alpha;
    std::unique_ptr<const LinOp> b;
    std::unique_ptr<const LinOp> beta;
    std::unique_ptr<const LinOp> x;

    linop_data(const LinOp *A, const LinOp *alpha, const LinOp *b,
               const LinOp *beta, const LinOp *x)
    {
        this->A = A->clone();
        if (alpha != nullptr) {
            this->alpha = alpha->clone();
        }
        this->b = b->clone();
        if (beta != nullptr) {
            this->beta = beta->clone();
        }
        this->x = x->clone();
    }
};


/**
 * Struct representing LinOp factory related data
 */
struct linop_factory_data {
    const LinOpFactory *factory;
    std::unique_ptr<const LinOp> input;
    std::unique_ptr<const LinOp> output;

    linop_factory_data(const LinOpFactory *factory, const LinOp *input,
                       const LinOp *output)
        : factory{factory}
    {
        this->input = input->clone();
        if (output != nullptr) {
            this->output = output->clone();
        }
    }
};


/**
 * Struct representing Criterion related data
 */
struct criterion_data {
    std::unique_ptr<const stop::Criterion> criterion;
    const uint8 stoppingId;
    const bool setFinalized;
    const Array<stopping_status> *status;
    const bool oneChanged;
    const bool converged;

    criterion_data(const stop::Criterion *criterion, const uint8 stoppingId,
                   const bool setFinalized,
                   const Array<stopping_status> *status, const bool oneChanged,
                   const bool converged)
        : stoppingId{stoppingId},
          setFinalized{setFinalized},
          oneChanged{oneChanged},
          converged{converged}
    {
        this->criterion = criterion->clone();
        this->status =
            new Array<stopping_status>(status->get_executor(), *status);
    }
};


/**
 * Record is a Logger which logs every event to an object. The object can
 * then be accessed at any time by asking the logger to return it.
 */
class Record : public Logger {
public:
    /**
     * Struct storing the actually logged data
     */
    struct logged_data {
        std::deque<std::string> applies{};
        size_type num_iterations{};
        size_type converged_at_iteration{};
        std::deque<std::unique_ptr<const LinOp>> residuals{};

        std::deque<iteration_complete_data> iteration_completed{};

        std::deque<executor_data> allocation_started{};
        std::deque<executor_data> allocation_completed{};
        std::deque<executor_data> free_started{};
        std::deque<executor_data> free_completed{};
        std::deque<std::tuple<executor_data, executor_data>> copy_started{};
        std::deque<std::tuple<executor_data, executor_data>> copy_completed{};

        std::deque<operation_data> operation_started{};
        std::deque<operation_data> operation_completed{};

        std::deque<polymorphic_object_data> polymorphic_object_create_started{};
        std::deque<polymorphic_object_data>
            polymorphic_object_create_completed{};
        std::deque<polymorphic_object_data> polymorphic_object_copy_started{};
        std::deque<polymorphic_object_data> polymorphic_object_copy_completed{};
        std::deque<polymorphic_object_data> polymorphic_object_deleted{};

        std::deque<linop_data> linop_apply_started{};
        std::deque<linop_data> linop_apply_completed{};
        std::deque<linop_data> linop_advanced_apply_started{};
        std::deque<linop_data> linop_advanced_apply_completed{};
        std::deque<linop_factory_data> linop_factory_generate_started{};
        std::deque<linop_factory_data> linop_factory_generate_completed{};


        std::deque<criterion_data> criterion_check_started{};
        std::deque<criterion_data> criterion_check_completed{};
    };

    void on_iteration_complete(
        const LinOp *solver, const size_type &num_iterations,
        const LinOp *residual, const LinOp *solution = nullptr,
        const LinOp *residual_norm = nullptr) const override;

    void on_apply(const std::string &name) const override;

    void on_converged(const size_type &at_iteration,
                      const LinOp *residual) const override;

    /* Executor events */
    void on_allocation_started(const Executor *exec,
                               const size_type &num_bytes) const override;

    void on_allocation_completed(const Executor *exec,
                                 const size_type &num_bytes,
                                 const uintptr &location) const override;

    void on_free_started(const Executor *exec,
                         const uintptr &location) const override;

    void on_free_completed(const Executor *exec,
                           const uintptr &location) const override;

    void on_copy_started(const Executor *from, const Executor *to,
                         const uintptr &location_from,
                         const uintptr &location_to,
                         const size_type &num_bytes) const override;

    void on_copy_completed(const Executor *from, const Executor *to,
                           const uintptr &location_from,
                           const uintptr &location_to,
                           const size_type &num_bytes) const override;

    /* Operation events */
    void on_operation_launched(const Executor *exec,
                               const Operation *operation) const override;

    void on_operation_completed(const Executor *exec,
                                const Operation *operation) const override;

    /* PolymorphicObject events */
    void on_polymorphic_object_create_started(
        const Executor *exec, const PolymorphicObject *po) const override;

    void on_polymorphic_object_create_completed(
        const Executor *exec, const PolymorphicObject *input,
        const PolymorphicObject *output) const override;

    void on_polymorphic_object_copy_started(
        const Executor *exec, const PolymorphicObject *from,
        const PolymorphicObject *to) const override;

    void on_polymorphic_object_copy_completed(
        const Executor *exec, const PolymorphicObject *from,
        const PolymorphicObject *to) const override;

    void on_polymorphic_object_deleted(
        const Executor *exec, const PolymorphicObject *po) const override;

    /* LinOp events */
    void on_linop_apply_started(const LinOp *A, const LinOp *b,
                                const LinOp *x) const override;

    void on_linop_apply_completed(const LinOp *A, const LinOp *b,
                                  const LinOp *x) const override;

    void on_linop_advanced_apply_started(const LinOp *A, const LinOp *alpha,
                                         const LinOp *b, const LinOp *beta,
                                         const LinOp *x) const override;

    void on_linop_advanced_apply_completed(const LinOp *A, const LinOp *alpha,
                                           const LinOp *b, const LinOp *beta,
                                           const LinOp *x) const override;

    void on_linop_factory_generate_started(const LinOpFactory *factory,
                                           const LinOp *input) const override;

    void on_linop_factory_generate_completed(
        const LinOpFactory *factory, const LinOp *input,
        const LinOp *output) const override;

    /* Criterion events */
    void on_criterion_check_started(const stop::Criterion *criterion,
                                    const uint8 &stoppingId,
                                    const bool &setFinalized) const override;

    void on_criterion_check_completed(const stop::Criterion *criterion,
                                      const uint8 &stoppingId,
                                      const bool &setFinalized,
                                      const Array<stopping_status> *status,
                                      const bool &oneChanged,
                                      const bool &converged) const override;

    static std::unique_ptr<Record> create(
        std::shared_ptr<const Executor> exec,
        const mask_type &enabled_events = Logger::all_events_mask,
        size_type max_storage = 0)
    {
        return std::unique_ptr<Record>(
            new Record(exec, enabled_events, max_storage));
    }

    /**
     * Returns the logged data
     *
     * @return the logged data
     */
    const logged_data &get() const noexcept { return data_; }

    /**
     * @copydoc ::get()
     */
    logged_data &get() noexcept { return data_; }

protected:
    explicit Record(std::shared_ptr<const gko::Executor> exec,
                    const mask_type &enabled_events = Logger::all_events_mask,
                    size_type max_storage = 0)
        : Logger(exec, enabled_events), max_storage_{max_storage}
    {}

    mutable logged_data data_{};
    size_type max_storage_{};
};


}  // namespace log
}  // namespace gko


#endif  // GKO_CORE_LOG_RECORD_HPP_
