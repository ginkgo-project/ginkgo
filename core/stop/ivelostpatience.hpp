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

#ifndef GKO_CORE_STOP_IVELOSTPATIENCE_HPP_
#define GKO_CORE_STOP_IVELOSTPATIENCE_HPP_


#include "core/stop/criterion.hpp"


namespace gko {
namespace stop {


struct IveLostPatience : gko::stop::Criterion {
    struct Factory : gko::stop::Criterion::Factory {
        using t = volatile bool &;
        Factory(t v) : v_{v} {}

        static std::unique_ptr<Factory> create(t v)
        {
            return std::make_unique<Factory>(v);
        }
        std::unique_ptr<Criterion> create_criterion(
            std::shared_ptr<const LinOp> system_matrix,
            std::shared_ptr<const LinOp> b, const LinOp *x) const override
        {
            return std::make_unique<IveLostPatience>(v_);
        }
        t v_;
    };

    IveLostPatience(volatile bool &is_user_bored)
        : is_user_bored_{is_user_bored}
    {
        // assume user is not bored before even starting the solver
        is_user_bored_ = false;
    }

protected:
    bool check(Array<bool> &, const Updater &) override
    {
        return is_user_bored_;
    }
    volatile bool &is_user_bored_;
};


}  // namespace stop
}  // namespace gko


#endif  // GKO_CORE_STOP_IVELOSTPATIENCE_HPP_
