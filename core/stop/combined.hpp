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

#ifndef GKO_CORE_STOP_COMBINED_HPP_
#define GKO_CORE_STOP_COMBINED_HPP_


#include "core/stop/criterion.hpp"


#include <vector>


namespace gko {
namespace stop {


class Combined : public Criterion {
public:
    struct Factory : public Criterion::Factory {
        using t = std::vector<std::unique_ptr<Criterion::Factory>>;

        template <class... V>
        Factory(V... v)
        {
            emplace(std::move(v)...);
        }

        template <class V, class... R>
        void emplace(V v, R... rest)
        {
            v_.emplace_back(std::move(v));
            emplace(std::move(rest)...);
        }

        void emplace() {}

        template <class... V>
        static std::unique_ptr<Factory> create(V... v)
        {
            return std::unique_ptr<Factory>(new Factory(std::move(v)...));
        }

        std::unique_ptr<Criterion> create_criterion(
            std::shared_ptr<const LinOp> system_matrix,
            std::shared_ptr<const LinOp> b, const LinOp *x) const override;
        t v_{};
    };


    Combined() {}

    void add_subcriterion(std::unique_ptr<Criterion> c);

    bool check(Array<bool> &, const Updater &) override;

private:
    std::vector<std::unique_ptr<Criterion>> criterions_{};
};


}  // namespace stop
}  // namespace gko


#endif  // GKO_CORE_STOP_COMBINED_HPP_
