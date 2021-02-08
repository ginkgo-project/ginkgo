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

#include "core/components/prefix_sum.hpp"


#include <omp.h>
#include <algorithm>


namespace gko {
namespace kernels {
namespace omp {
namespace components {


template <typename T>
static inline T power(const T x, const int exponent)
{
    T ans = static_cast<T>(1);
    for (int i = 0; i < exponent; i++) {
        ans *= x;
    }
    return ans;
}


/*
 * The last entry of the input array is never used, but is replaced.
 */
template <typename IndexType>
void prefix_sum(std::shared_ptr<const OmpExecutor> exec,
                IndexType *const counts, const size_type num_entries)
{
    const auto nentries = static_cast<IndexType>(num_entries);
    if (num_entries <= 1) {
        return;
    }

    const int nthreads = omp_get_max_threads();
    const IndexType def_num_witems = (num_entries - 1) / nthreads + 1;

#pragma omp parallel
    {
        const int thread_id = omp_get_thread_num();
        const IndexType startidx = thread_id * def_num_witems;
        const IndexType endidx =
            std::min(nentries, (thread_id + 1) * def_num_witems);
        const IndexType startval = counts[startidx];

#pragma omp barrier

        IndexType partial_sum = startval;
        for (IndexType i = startidx + 1; i < endidx; ++i) {
            auto nnz = counts[i];
            counts[i] = partial_sum;
            partial_sum += nnz;
        }
        if (thread_id != nthreads - 1) {
            counts[endidx] = partial_sum;
        }
    }

    counts[0] = 0;

    const auto levels = static_cast<int>(std::ceil(std::log(nthreads)));
    for (int ilvl = 0; ilvl < levels; ilvl++) {
        const IndexType factor = power(2, (ilvl + 1));
        const IndexType lvl_num_witems = factor * def_num_witems;
        const int ntasks = (nthreads - 1) / factor + 1;

#pragma omp parallel for
        for (int itask = 0; itask < ntasks; itask++) {
            const IndexType startidx = std::min(
                nentries, lvl_num_witems / 2 + itask * lvl_num_witems + 1);
            const IndexType endidx =
                std::min(nentries, (itask + 1) * lvl_num_witems + 1);
            const IndexType baseval = counts[startidx - 1];
            for (int i = startidx; i < endidx; i++) {
                counts[i] += baseval;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_PREFIX_SUM_KERNEL);

// instantiate for size_type as well, as this is used in the Sellp format
template GKO_DECLARE_PREFIX_SUM_KERNEL(size_type);


}  // namespace components
}  // namespace omp
}  // namespace kernels
}  // namespace gko
