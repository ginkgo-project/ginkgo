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
class Stream : public EnablePolymorphicObject<Stream<ValueType>, Logger>,
               public EnableCreateMethod<Stream<ValueType>> {
    friend class EnablePolymorphicObject<Stream<ValueType>, Logger>;
    friend class EnableCreateMethod<Stream<ValueType>>;

public:
    using EnablePolymorphicObject<Stream<ValueType>,
                                  Logger>::EnablePolymorphicObject;

    void on_iteration_complete(const size_type &num_iterations) const override;

    void on_apply(const std::string &name) const override;

    void on_converged(const size_type &at_iteration,
                      const LinOp *residual) const override;

protected:
    explicit Stream(
        std::shared_ptr<const gko::Executor> exec,
        const Logger::mask_type &enabled_events = Logger::all_events_mask,
        std::ostream &os = std::cout)
        : EnablePolymorphicObject<Stream<ValueType>, Logger>(exec,
                                                             enabled_events),
          os_(os)
    {}

    Stream<ValueType> &operator=(const Stream<ValueType> &other)
    {
        return *this;
    }

    Stream<ValueType> &operator=(Stream<ValueType> &other) { return *this; }

    std::ostream &os_;
    const std::string prefix_ = "[LOG] >>> ";
};


}  // namespace log
}  // namespace gko


#endif  // GKO_CORE_LOG_STREAM_HPP_
