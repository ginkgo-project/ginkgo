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


#include "core/log/logger.hpp"


#include <papi_sde_interface.h>


namespace gko {
namespace log {


extern const papi_handle_t papi_handle;


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
    /**
     * creates a Papi Logger
     * @param enabled_events  the events enabled for this Logger
     * @param handle  the papi handle
     */
    static std::shared_ptr<Papi> create(const mask_type &enabled_events)
    {
        return std::shared_ptr<Papi>(new Papi(enabled_events));
    }

    void on_iteration_complete(const size_type num_iterations) const override;
    void on_apply(const std::string name) const override;

    void on_converged(const size_type at_iteration,
                      const LinOp *residual) const override;

protected:
    explicit Papi(const mask_type &enabled_events) : Logger(enabled_events) {}
};


}  // namespace log
}  // namespace gko


#endif  // GKO_CORE_LOG_OSTREAM_HPP_
