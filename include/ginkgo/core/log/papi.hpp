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

#ifndef GKO_CORE_LOG_PAPI_HPP_
#define GKO_CORE_LOG_PAPI_HPP_


#include <cstddef>
#include <iostream>
#include <map>


#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/log/logger.hpp>


#include <papi_sde_interface.h>


namespace gko {
namespace log {


/**
 * Papi is a Logger which logs every event to the PAPI software. Thanks to this
 * logger, applications which interface with PAPI can access Ginkgo internal
 * data through PAPI.
 *
 * @tparam ValueType  the type of values stored in the class (e.g. residuals)
 */
template <typename ValueType = default_precision>
class Papi : public Logger {
public:
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
        const Executor *, const PolymorphicObject *po) const override;

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

    /* LinOpFactory events */
    void on_linop_factory_generate_started(const LinOpFactory *factory,
                                           const LinOp *input) const override;

    void on_linop_factory_generate_completed(
        const LinOpFactory *factory, const LinOp *input,
        const LinOp *output) const override;

    void on_criterion_check_completed(
        const stop::Criterion *criterion, const size_type &num_iterations,
        const LinOp *residual, const LinOp *residual_norm,
        const LinOp *solutino, const uint8 &stopping_id,
        const bool &set_finalized, const Array<stopping_status> *status,
        const bool &one_changed, const bool &all_converged) const override;

    /* Internal solver events */
    void on_iteration_complete(
        const LinOp *solver, const size_type &num_iterations,
        const LinOp *residual, const LinOp *solution = nullptr,
        const LinOp *residual_norm = nullptr) const override;

    /**
     * Creates a Papi Logger.
     *
     * @param enabled_events  the events enabled for this Logger
     * @param handle  the papi handle
     */
    static std::shared_ptr<Papi> create(
        std::shared_ptr<const gko::Executor> exec,
        const Logger::mask_type &enabled_events)
    {
        return std::shared_ptr<Papi>(new Papi(exec, enabled_events));
    }


    ~Papi()
    {
        unregister_queue(allocation_started, "allocation_started");
        unregister_queue(allocation_completed, "allocation_completed");
        unregister_queue(free_started, "free_started");
        unregister_queue(free_completed, "free_completed");
        unregister_queue(copy_started_from, "copy_started_from");
        unregister_queue(copy_started_to, "copy_started_to");
        unregister_queue(copy_completed_from, "copy_completed_from");
        unregister_queue(copy_completed_to, "copy_completed_to");
        unregister_queue(operation_launched, "operation_launched");
        unregister_queue(operation_completed, "operation_completed");
        unregister_queue(polymorphic_object_create_started,
                         "polymorphic_object_create_started");
        unregister_queue(polymorphic_object_create_completed,
                         "polymorphic_object_create_completed");
        unregister_queue(polymorphic_object_copy_started,
                         "polymorphic_object_copy_started");
        unregister_queue(polymorphic_object_copy_completed,
                         "polymorphic_object_copy_completed");
        unregister_queue(polymorphic_object_deleted,
                         "polymorphic_object_deleted");
        unregister_queue(linop_factory_generate_started,
                         "linop_factory_generate_started");
        unregister_queue(linop_factory_generate_completed,
                         "linop_factory_generate_completed");
        unregister_queue(linop_apply_started, "linop_apply_started");
        unregister_queue(linop_apply_completed, "linop_apply_completed");
        unregister_queue(linop_advanced_apply_started,
                         "linop_advanced_apply_started");
        unregister_queue(linop_advanced_apply_completed,
                         "linop_advanced_apply_completed");
        unregister_queue(criterion_check_completed,
                         "criterion_check_completed");
        unregister_queue(iteration_complete, "iteration_complete");
    }

    const std::string get_handle_name() { return name; }

protected:
    explicit Papi(
        std::shared_ptr<const gko::Executor> exec,
        const Logger::mask_type &enabled_events = Logger::all_events_mask)
        : Logger(exec, enabled_events)
    {
        std::ostringstream os;
        os << "ginkgo" << logger_count;
        name = os.str();
        papi_handle = papi_sde_init(name.c_str());
        logger_count++;
    }

private:
    template <typename T>
    void unregister_queue(std::map<std::uintptr_t, T> &queue, const char *name)
    {
        for (auto e : queue) {
            std::ostringstream oss;
            oss << name << "::" << e.first;
            papi_sde_unregister_counter(this->papi_handle, oss.str().c_str());
        }
        queue.clear();
    }

    template <typename T, typename U>
    U &add_to_map(const T *ptr, std::map<std::uintptr_t, U> &map,
                  const char *name) const
    {
        const auto tmp = reinterpret_cast<uintptr>(ptr);
        if (map.find(tmp) == map.end()) {
            map[tmp] = 0;
        }
        auto &value = map[tmp];
        if (!value) {
            std::ostringstream oss;
            oss << name << "::" << tmp;
            papi_sde_register_counter(this->papi_handle, oss.str().c_str(),
                                      PAPI_SDE_RO | PAPI_SDE_INSTANT,
                                      PAPI_SDE_long_long, &value);
        }
        return map[tmp];
    }

    mutable std::map<std::uintptr_t, size_type> allocation_started;
    mutable std::map<std::uintptr_t, size_type> allocation_completed;
    mutable std::map<std::uintptr_t, size_type> free_started;
    mutable std::map<std::uintptr_t, size_type> free_completed;
    mutable std::map<std::uintptr_t, size_type> copy_started_from;
    mutable std::map<std::uintptr_t, size_type> copy_started_to;
    mutable std::map<std::uintptr_t, size_type> copy_completed_from;
    mutable std::map<std::uintptr_t, size_type> copy_completed_to;

    mutable std::map<std::uintptr_t, size_type> operation_launched;
    mutable std::map<std::uintptr_t, size_type> operation_completed;

    mutable std::map<std::uintptr_t, size_type>
        polymorphic_object_create_started;
    mutable std::map<std::uintptr_t, size_type>
        polymorphic_object_create_completed;
    mutable std::map<std::uintptr_t, size_type> polymorphic_object_copy_started;
    mutable std::map<std::uintptr_t, size_type>
        polymorphic_object_copy_completed;
    mutable std::map<std::uintptr_t, size_type> polymorphic_object_deleted;

    mutable std::map<std::uintptr_t, size_type> linop_factory_generate_started;
    mutable std::map<std::uintptr_t, size_type>
        linop_factory_generate_completed;

    mutable std::map<std::uintptr_t, size_type> linop_apply_started;
    mutable std::map<std::uintptr_t, size_type> linop_apply_completed;
    mutable std::map<std::uintptr_t, size_type> linop_advanced_apply_started;
    mutable std::map<std::uintptr_t, size_type> linop_advanced_apply_completed;

    mutable std::map<std::uintptr_t, void *> criterion_check_completed;

    mutable std::map<std::uintptr_t, size_type> iteration_complete;

    static size_type logger_count;

    std::string name = std::string("ginkgo");
    papi_handle_t papi_handle;
    // const papi_handle_t papi_handle = papi_sde_init("gko");
};


}  // namespace log
}  // namespace gko


#endif  // GKO_CORE_LOG_OSTREAM_HPP_
