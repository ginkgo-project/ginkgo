// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_OMP_COMPONENTS_PREFIX_SUM_HPP_
#define GKO_OMP_COMPONENTS_PREFIX_SUM_HPP_

#include <algorithm>
#include <iterator>
#include <limits>
#include <string>

#include <omp.h>

#include "core/base/allocator.hpp"
#include "core/base/iterator_factory.hpp"


namespace gko {
namespace kernels {
namespace omp {
namespace components {


/*
 * Similar to prefix_sum, but with a special scan operator that only reduces
 * within runs of the same key value (each key run must only occur once,
 * otherwise the scan operation is not necessarily associaive).
 * It also doesn't ignore the last value!
 */
template <typename KeyIterator, typename Iterator,
          typename ScanOp =
              std::plus<typename std::iterator_traits<Iterator>::value_type>>
void segmented_prefix_sum(
    std::shared_ptr<const OmpExecutor> exec, KeyIterator key, Iterator it,
    const size_type num_entries,
    typename std::iterator_traits<KeyIterator>::value_type key_init = {},
    typename std::iterator_traits<Iterator>::value_type init = {},
    ScanOp op = {})
{
    using key_type = typename std::iterator_traits<KeyIterator>::value_type;
    using value_type = typename std::iterator_traits<Iterator>::value_type;
    // the operation only makes sense for arrays of size at least 2
    if (num_entries < 2) {
        if (num_entries == 0) {
            return;
        } else {
            *it = init;
            return;
        }
    }

    const int nthreads = omp_get_max_threads();
    vector<value_type> proc_sums(nthreads, init, {exec});
    vector<key_type> proc_first_key(nthreads, key_init, {exec});
    vector<key_type> proc_last_key(nthreads, key_init, {exec});
    const size_type def_num_witems = (num_entries - 1) / nthreads + 1;

#pragma omp parallel
    {
        const int thread_id = omp_get_thread_num();
        const size_type startidx = thread_id * def_num_witems;
        const size_type endidx =
            std::min(num_entries, (thread_id + 1) * def_num_witems);

        auto partial_sum = init;
        auto cur_key = startidx < num_entries ? key[startidx] : key_init;
        proc_first_key[thread_id] = cur_key;
        for (size_type i = startidx; i < endidx; ++i) {
            auto value = it[i];
            auto new_key = key[i];
            if (cur_key != new_key) {
                partial_sum = init;
                cur_key = new_key;
            }
            it[i] = partial_sum;
            partial_sum = op(partial_sum, value);
        }

        proc_sums[thread_id] = partial_sum;
        proc_last_key[thread_id] = cur_key;

#pragma omp barrier

#pragma omp single
        {
            for (int i = 0; i < nthreads - 1; i++) {
                // the next block carries over the previous partial sum
                // if it starts and ends with the same key as the next one
                if (proc_last_key[i] == proc_first_key[i + 1] &&
                    proc_first_key[i + 1] == proc_last_key[i + 1]) {
                    proc_sums[i + 1] = op(proc_sums[i], proc_sums[i + 1]);
                }
            }
        }

        if (thread_id > 0) {
            for (size_type i = startidx; i < endidx; i++) {
                if (key[i] == proc_last_key[thread_id - 1]) {
                    it[i] = op(it[i], proc_sums[thread_id - 1]);
                }
            }
        }
    }
}


}  // namespace components
}  // namespace omp
}  // namespace kernels
}  // namespace gko

#endif  // GKO_OMP_COMPONENTS_PREFIX_SUM_HPP_
