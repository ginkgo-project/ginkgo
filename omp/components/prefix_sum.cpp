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


#include <algorithm>


#include <omp.h>


#include "core/base/allocator.hpp"


namespace gko {
namespace kernels {
namespace omp {
namespace components {


/*
 * The last entry of the input array is never used, but is replaced.
 */
template <typename IndexType>
void prefix_sum(std::shared_ptr<const OmpExecutor> exec,
                IndexType *const counts, const size_type num_entries)
{
    // the operation only makes sense for arrays of size at least 2
    if (num_entries < 2) {
        if (num_entries == 0) {
            return;
        } else {
            counts[0] = 0;
            return;
        }
    }

    const int nthreads = omp_get_max_threads();
    vector<IndexType> proc_sums(nthreads, 0, {exec});
    const size_type def_num_witems = (num_entries - 1) / nthreads + 1;

#pragma omp parallel
    {
        const int thread_id = omp_get_thread_num();
        const size_type startidx = thread_id * def_num_witems;
        const size_type endidx =
            std::min(num_entries, (thread_id + 1) * def_num_witems);

        IndexType partial_sum{0};
        for (size_type i = startidx; i < endidx; ++i) {
            auto nnz = counts[i];
            counts[i] = partial_sum;
            partial_sum += nnz;
        }

        proc_sums[thread_id] = partial_sum;

#pragma omp barrier

#pragma omp single
        {
            for (int i = 0; i < nthreads - 1; i++) {
                proc_sums[i + 1] += proc_sums[i];
            }
        }

        if (thread_id > 0) {
            for (size_type i = startidx; i < endidx; i++) {
                counts[i] += proc_sums[thread_id - 1];
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
