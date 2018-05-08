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

#ifndef GKO_CORE_LOG_OSTREAM_HPP_
#define GKO_CORE_LOG_OSTREAM_HPP_


#include "core/log/logger.hpp"


#include <fstream>
#include <iostream>


namespace gko {
namespace log {


class Ostream : public Logger {
public:
    static std::shared_ptr<Ostream> create(const mask_type &enabled_events,
                                           std::ostream &os)
    {
        return std::shared_ptr<Ostream>(new Ostream(enabled_events, os));
    }

    void on_iteration_complete(size_type num_iterations) const override;
    void on_apply() const override;

    void on_converged(size_type at_iteration, LinOp *residual) const override;

protected:
    explicit Ostream(const mask_type &enabled_events, std::ostream &os)
        : Logger(enabled_events), os_(os)
    {}


    std::ostream &os_;

    const std::string prefix = "[LOG] >>> ";
};


}  // namespace log
}  // namespace gko


#endif  // GKO_CORE_LOG_OSTREAM_HPP_
