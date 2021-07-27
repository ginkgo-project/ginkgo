/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/overlap.hpp>
#include <ginkgo/core/base/range.hpp>
#include <ginkgo/core/base/types.hpp>


#include <tuple>
#include <vector>

namespace gko {


std::tuple<std::vector<span>, std::vector<span>>
calculate_overlap_row_and_col_spans(const dim<2> &size, span &rspan,
                                    span &cspan, bool unidir, size_type overlap,
                                    bool st_overlap)
{
    if (!unidir) {
        auto or_span = std::vector<span>{rspan, rspan};
        auto oc_span =
            std::vector<span>{span{cspan.begin - overlap, cspan.begin},
                              span{cspan.end, cspan.end + overlap}};
        return std::make_tuple<std::vector<span>, std::vector<span>>(
            std::move(or_span), std::move(oc_span));
    } else {
        auto or_span = std::vector<span>{rspan};
        std::vector<span> oc_span;
        if (st_overlap) {
            oc_span.push_back(span{cspan.begin - overlap, cspan.begin});
        } else {
            oc_span.push_back(span{cspan.end, cspan.end + overlap});
        }
        return std::make_tuple<std::vector<span>, std::vector<span>>(
            std::move(or_span), std::move(oc_span));
    }
}


}  // namespace gko
