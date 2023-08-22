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

#include "core/components/prefix_sum_kernels.hpp"


#include <CL/sycl.hpp>


#include <ginkgo/core/base/types.hpp>


#include "core/base/types.hpp"
#include "dpcpp/base/helper.hpp"
#include "dpcpp/components/prefix_sum.dp.hpp"
#include "dpcpp/synthesizer/implementation_selection.hpp"


namespace gko {
namespace kernels {
namespace sycl {
namespace components {


static constexpr auto block_cfg_list = dcfg_block_list_t();


GKO_ENABLE_IMPLEMENTATION_CONFIG_SELECTION_TOTYPE(start_prefix_sum,
                                                  start_prefix_sum, DCFG_1D);
GKO_ENABLE_DEFAULT_CONFIG_CALL(start_prefix_sum_call, start_prefix_sum,
                               block_cfg_list)

GKO_ENABLE_IMPLEMENTATION_CONFIG_SELECTION_TOTYPE(finalize_prefix_sum,
                                                  finalize_prefix_sum, DCFG_1D);
GKO_ENABLE_DEFAULT_CONFIG_CALL(finalize_prefix_sum_call, finalize_prefix_sum,
                               block_cfg_list)


template <typename IndexType>
void prefix_sum_nonnegative(std::shared_ptr<const DpcppExecutor> exec,
                            IndexType* counts, size_type num_entries)
{
    // prefix_sum should only be performed on a valid array
    if (num_entries > 0) {
        // TODO detect overflow
        auto queue = exec->get_queue();
        constexpr auto block_cfg_array = as_array(block_cfg_list);
        const std::uint32_t cfg =
            get_first_cfg(block_cfg_array, [&queue](std::uint32_t cfg) {
                return validate(queue, DCFG_1D::decode<0>(cfg),
                                DCFG_1D::decode<1>(cfg));
            });
        const auto wg_size = DCFG_1D::decode<0>(cfg);
        auto num_blocks = ceildiv(num_entries, wg_size);
        array<IndexType> block_sum_array(exec, num_blocks - 1);
        auto block_sums = block_sum_array.get_data();
        start_prefix_sum_call(cfg, num_blocks, wg_size, 0, exec->get_queue(),
                              num_entries, counts, block_sums);
        // add the total sum of the previous block only when the number of
        // blocks is larger than 1.
        if (num_blocks > 1) {
            finalize_prefix_sum_call(cfg, num_blocks, wg_size, 0,
                                     exec->get_queue(), num_entries, counts,
                                     block_sums);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_PREFIX_SUM_NONNEGATIVE_KERNEL);

// instantiate for size_type as well, as this is used in the Sellp format
template GKO_DECLARE_PREFIX_SUM_NONNEGATIVE_KERNEL(size_type);


}  // namespace components
}  // namespace sycl
}  // namespace kernels
}  // namespace gko
