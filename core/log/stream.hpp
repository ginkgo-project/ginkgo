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

#ifndef GKO_CORE_LOG_STREAM_HPP_
#define GKO_CORE_LOG_STREAM_HPP_


#include "core/log/logger.hpp"


#include <fstream>
#include <iostream>


#include "core/base/name_demangling.hpp"


namespace gko {
namespace log {


/**
 * Stream is a Logger which logs every event to a stream. This can typically be
 * used to log to a file or to the console.
 *
 * @tparam ValueType  the type of values stored in the class (i.e. ValueType
 *                    template parameter of the concrete Loggable this class
 *                    will log)
 */
template <typename ValueType = default_precision>
class Stream : public Logger {
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

    /* Internal solver events */
    void on_iteration_complete(
        const LinOp *solver, const size_type &num_iterations,
        const LinOp *residual, const LinOp *solution = nullptr,
        const LinOp *residual_norm = nullptr) const override;

    /**
     * Creates a Stream logger. This dynamically allocates the memory,
     * constructs the object and return an std::unique_ptr to this object.
     *
     * @param exec  the executor
     * @param enabled_events  the events enabled for this logger. By default all
     *                        events.
     * @param os  the stream used for this logger
     * @param verbose  whether we want detailed information or not. This
     *                 includes always printing residuals and other information
     *                 which can give a large output.
     *
     * @return an std::unique_ptr to the the constructed object
     *
     * @internal here I cannot use EnableCreateMethod due to complex circular
     * dependencies. At the same time, this method is short enough that it
     * shouldn't be a problem.
     */
    static std::unique_ptr<Stream> create(
        std::shared_ptr<const Executor> exec,
        const Logger::mask_type &enabled_events = Logger::all_events_mask,
        std::ostream &os = std::cout, bool verbose = false)
    {
        return std::unique_ptr<Stream>(
            new Stream(exec, enabled_events, os, verbose));
    }

protected:
    /**
     * Creates a Stream logger.
     *
     * @param exec  the executor
     * @param enabled_events  the events enabled for this logger. By default all
     *                        events.
     * @param os  the stream used for this logger
     * @param verbose  whether we want detailed information or not. This
     *                 includes always printing residuals and other information
     *                 which can give a large output.
     */
    explicit Stream(
        std::shared_ptr<const gko::Executor> exec,
        const Logger::mask_type &enabled_events = Logger::all_events_mask,
        std::ostream &os = std::cout, bool verbose = false)
        : Logger(exec, enabled_events), os_(os), verbose_(verbose)
    {}

    /**
     * Helper function which formats a number of bytes information.
     *
     * @param num_bytes  the number of bytes to print
     *
     * @return a properly formatted string containing the information
     */
    std::string bytes_name(const size_type &num_bytes) const
    {
        std::ostringstream oss;
        oss << "Bytes[" << num_bytes << "]";
        return oss.str();
    }

    /**
     * Helper function which formats a gko::Operation information.
     *
     * @param op  the operation to print
     *
     * @return a properly formatted string containing the information
     */
    std::string operation_name(const Operation *op) const
    {
        std::ostringstream oss;
        oss << "Operation[" << name_demangling::get_name(*op) << ";" << op
            << "]";
        return oss.str();
    }

    /**
     * Helper function which formats a gko::Executor information.
     *
     * @param exec  the executor to print
     *
     * @return a properly formatted string containing the information
     */
    std::string executor_name(const Executor *exec) const
    {
        std::ostringstream oss;
        oss << "Executor[" << name_demangling::get_name(*exec) << ";" << exec
            << "]";
        return oss.str();
    }

    /**
     * Helper function which formats a memory location information.
     *
     * @param location  the memory location to print
     *
     * @return a properly formatted string containing the information
     */
    std::string location_name(const uintptr &location) const
    {
        std::ostringstream oss;
        oss << "Location[" << std::hex << "0x" << location << "]" << std::dec;
        return oss.str();
    }

    /**
     * Helper function which formats a gko::PolymorphicObject information.
     *
     * @param po  the PolymorphicObject to print
     *
     * @return a properly formatted string containing the information
     */
    std::string po_name(const PolymorphicObject *po) const
    {
        std::ostringstream oss;
        oss << "PolymorphicObject[" << name_demangling::get_name(*po) << ","
            << po << "]";
        return oss.str();
    }

    /**
     * Helper function which formats a gko::LinOp information.
     *
     * @param linop  the LinOp to print
     *
     * @return a properly formatted string containing the information
     */
    std::string linop_name(const LinOp *linop) const
    {
        std::ostringstream oss;
        oss << "LinOp[";
        // Only these can eventually be nullptr
        // due to residual_norm, etc
        if (linop == nullptr) {
            oss << name_demangling::get_name(linop);
        } else {
            oss << name_demangling::get_name(*linop);
        }
        oss << "," << linop << "]";
        return oss.str();
    }

    /**
     * Helper function which formats a gko::LinOpFactory information.
     *
     * @param op  the LinOpFactory to print
     *
     * @return a properly formatted string containing the information
     */
    std::string linop_factory_name(const LinOpFactory *factory) const
    {
        std::ostringstream oss;
        oss << "LinOpFactory[" << name_demangling::get_name(*factory) << ","
            << factory << "]";
        return oss.str();
    }

    /**
     * Helper function which formats a gko::stop::Criterion information.
     *
     * @param op  the Criterion to print
     *
     * @return a properly formatted string containing the information
     */
    std::string criterion_name(const stop::Criterion *criterion) const
    {
        std::ostringstream oss;
        oss << "Criterion[" << name_demangling::get_name(*criterion) << ","
            << criterion << "]";
        return oss.str();
    }

private:
    std::ostream &os_;
    static constexpr const char *prefix_ = "[LOG] >>> ";
    bool verbose_;
};


}  // namespace log
}  // namespace gko


#endif  // GKO_CORE_LOG_STREAM_HPP_
