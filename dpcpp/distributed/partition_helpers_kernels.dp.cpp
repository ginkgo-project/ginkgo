/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

// force-top: on
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
// force-top: off


#include "core/distributed/partition_helpers_kernels.hpp"


namespace gko {
namespace kernels {
namespace sycl {
namespace partition_helpers {

struct stride {
#if ONEDPL_VERSION_MAJOR >= 2022 && ONEDPL_VERSION_MINOR >= 1
    template <typename Index>
    Index operator()(const Index& i) const
    {
        return i * 2;
    }
#else
    // Some older version require [] while some require (), so I added both
    template <typename Index>
    Index operator[](const Index& i) const
    {
        return i * 2;
    }

    template <typename Index>
    Index operator()(const Index& i) const
    {
        return i * 2;
    }
#endif
};

template <typename GlobalIndexType>
void sort_by_range_start(
    std::shared_ptr<const DefaultExecutor> exec,
    array<GlobalIndexType>& range_start_ends,
    array<experimental::distributed::comm_index_type>& part_ids)
{
    auto policy =
        oneapi::dpl::execution::make_device_policy(*exec->get_queue());
    auto num_ranges = range_start_ends.get_num_elems() / 2;

    auto start_it = oneapi::dpl::make_permutation_iterator(
        range_start_ends.get_data(), stride{});
    auto end_it = oneapi::dpl::make_permutation_iterator(
        range_start_ends.get_data() + 1, stride{});

    // older versions of oneDPL have a bug when sorting permutation iterators
#if ONEDPL_VERSION_MAJOR >= 2022 && ONEDPL_VERSION_MINOR >= 1
    auto zip_it =
        oneapi::dpl::make_zip_iterator(start_it, end_it, part_ids.get_data());
    std::stable_sort(policy, zip_it, zip_it + num_ranges, [](auto a, auto b) {
        return std::get<0>(a) < std::get<0>(b);
    });
#else
    array<GlobalIndexType> starts(exec, num_ranges);
    array<GlobalIndexType> ends(exec, num_ranges);

    std::copy(policy, start_it, start_it + num_ranges, starts.get_data());
    std::copy(policy, end_it, end_it + num_ranges, ends.get_data());

    auto zip_it = oneapi::dpl::make_zip_iterator(
        starts.get_data(), ends.get_data(), part_ids.get_data());
    std::stable_sort(policy, zip_it, zip_it + num_ranges, [](auto a, auto b) {
        return std::get<0>(a) < std::get<0>(b);
    });

    std::copy(policy, starts.get_data(), starts.get_data() + num_ranges,
              start_it);
    std::copy(policy, ends.get_data(), ends.get_data() + num_ranges, end_it);
#endif
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_PARTITION_HELPERS_SORT_BY_RANGE_START);


}  // namespace partition_helpers
}  // namespace sycl
}  // namespace kernels
}  // namespace gko
