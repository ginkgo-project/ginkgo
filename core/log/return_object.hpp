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

#ifndef GKO_CORE_LOG_RETURN_OBJECT_HPP_
#define GKO_CORE_LOG_RETURN_OBJECT_HPP_


#include "core/log/logger.hpp"
#include "core/matrix/dense.hpp"


#include <memory>


namespace gko {
namespace log {


struct LoggedData {
    size_type num_iterations;
    size_type converged_at_iteration;
    std::unique_ptr<const gko::matrix::Dense<>> residual;
};


/**
 * ReturnObject is a Logger which logs every event to an object. The object can
 * then be accessed at any time by asking the logger to return it.
 */
class ReturnObject : public Logger {
public:
    /**
     * creates a ReturnObject Logger used to directly access logged data
     * @param enabled_events the events enabled for this Logger
     */
    static std::shared_ptr<ReturnObject> create(const mask_type &enabled_events)
    {
        return std::shared_ptr<ReturnObject>(new ReturnObject(enabled_events));
    }

    void on_iteration_complete(const size_type num_iterations) const override;
    void on_apply() const override;

    void on_converged(const size_type at_iteration,
                      const LinOp *residual) const override;

    /**
     * Returns a shared pointer to the logged data
     * @returns a shared pointer to the logged data
     */
    std::shared_ptr<const LoggedData> get_return_object() { return rod_; }

protected:
    explicit ReturnObject(const mask_type &enabled_events)
        : Logger(enabled_events)
    {
        rod_ = std::shared_ptr<LoggedData>(new LoggedData());
    }

    std::shared_ptr<LoggedData> rod_;
};


}  // namespace log
}  // namespace gko


#endif  // GKO_CORE_LOG_RETURN_OBJECT_HPP_
