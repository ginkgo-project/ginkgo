/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#ifndef GKO_CORE_BASE_SUBSET_HPP_
#define GKO_CORE_BASE_SUBSET_HPP_


#include <algorithm>
#include <mutex>
#include <vector>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>


namespace gko {


template <typename IndexType = int32>
struct subset {
    using index_type = IndexType;

    subset() = delete;

    subset(const index_type begin, const index_type end);

    static bool compare_end(const subset &x, const subset &y)
    {
        return x.end_ < y.end_;
    }

    static bool superset_index_compare(const subset &x, const subset &y)
    {
        return (x.superset_index_ + (x.end_ - x.begin_) <
                y.superset_index_ + (y.end_ - y.begin_));
    }

    inline bool operator<(const subset &subset2) const
    {
        return ((begin_ < subset2.begin_) ||
                ((begin_ == subset2.begin_) && (end_ < subset2.end_)));
    }

    inline bool operator==(const subset &subset2) const
    {
        return ((begin_ == subset2.begin_) && (end_ == subset2.end_));
    }

    index_type begin_;
    index_type end_;
    index_type superset_index_;
    // std::shared_ptr<const Executor> exec_;
};


}  // namespace gko


#endif  // GKO_CORE_BASE_SUBSET_HPP_
