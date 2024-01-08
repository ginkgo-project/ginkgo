// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/prefix_sum_kernels.hpp"


#include <algorithm>
#include <limits>


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
void prefix_sum_nonnegative(std::shared_ptr<const OmpExecutor> exec,
                            IndexType* const counts,
                            const size_type num_entries)
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
    bool overflow = false;
    constexpr auto max = std::numeric_limits<IndexType>::max();

#pragma omp parallel
    {
        const int thread_id = omp_get_thread_num();
        const size_type startidx = thread_id * def_num_witems;
        const size_type endidx =
            std::min(num_entries, (thread_id + 1) * def_num_witems);

        IndexType partial_sum{0};
        for (size_type i = startidx; i < endidx; ++i) {
            auto nnz = i < num_entries - 1 ? counts[i] : IndexType{};
            counts[i] = partial_sum;
            if (max - partial_sum < nnz) {
                overflow = true;
            }
            partial_sum = partial_sum + nnz;
        }

        proc_sums[thread_id] = partial_sum;

#pragma omp barrier

#pragma omp single
        {
            for (int i = 0; i < nthreads - 1; i++) {
                if (max - proc_sums[i + 1] < proc_sums[i]) {
                    overflow = true;
                }
                proc_sums[i + 1] = proc_sums[i + 1] + proc_sums[i];
            }
        }

        if (thread_id > 0) {
            for (size_type i = startidx; i < endidx; i++) {
                if (max - counts[i] < proc_sums[thread_id - 1]) {
                    overflow = true;
                }
                counts[i] += proc_sums[thread_id - 1];
            }
        }
    }
    if (overflow) {
        throw OverflowError(__FILE__, __LINE__,
                            name_demangling::get_type_name(typeid(IndexType)));
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_PREFIX_SUM_NONNEGATIVE_KERNEL);

// instantiate for size_type as well, as this is used in the Sellp format
template GKO_DECLARE_PREFIX_SUM_NONNEGATIVE_KERNEL(size_type);


}  // namespace components
}  // namespace omp
}  // namespace kernels
}  // namespace gko
