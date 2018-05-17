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


#include <deque>
#include <memory>


namespace gko {
namespace log {


/**
 * ReturnObject is a Logger which logs every event to an object. The object can
 * then be accessed at any time by asking the logger to return it.
 */
class ReturnObject : public Logger {
public:
    /**
     * Struct storing the actually logged data
     */
    struct LoggedData {
        std::deque<std::string> applies;
        size_type num_iterations;
        size_type converged_at_iteration;
        std::deque<std::unique_ptr<const LinOp>> residuals;

        LoggedData() = default;
    };

    /**
     * Creates a ReturnObject Logger used to directly access logged data
     * @param enabled_events  the events enabled for this Logger
     * @param max_storage  the maximum storage allowed in `std::deque`
     */
    static std::shared_ptr<ReturnObject> create(const mask_type &enabled_events,
                                                size_type max_storage = 0)
    {
        return std::shared_ptr<ReturnObject>(
            new ReturnObject(enabled_events, max_storage));
    }

    void on_iteration_complete(const size_type num_iterations) const override;

    void on_apply(const std::string name) const override;

    void on_converged(const size_type at_iteration,
                      const LinOp *residual) const override;

    /**
     * Returns a shared pointer to the logged data
     * @return  a shared pointer to the logged data
     */
    std::shared_ptr<const LoggedData> get_return_object()
    {
        return logged_data_;
    }

protected:
    explicit ReturnObject(const mask_type &enabled_events,
                          size_type max_storage)
        : Logger(enabled_events), max_storage_{max_storage}
    {
        /* NOTE: there is a bug with some MacOS compiler not initializing
         * variables to `0` therefore `make_shared` should not be used here. */
        logged_data_ = std::shared_ptr<LoggedData>(new LoggedData());
    }

    std::shared_ptr<LoggedData> logged_data_;
    size_type max_storage_;
};


}  // namespace log
}  // namespace gko


#endif  // GKO_CORE_LOG_RETURN_OBJECT_HPP_
