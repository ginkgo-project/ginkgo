// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/compressed_coo_kernels.hpp"

#include <ginkgo/core/base/math.hpp>

#include "common/unified/base/kernel_launch.hpp"
#include "core/base/intrinsics.hpp"
#include "core/components/bitvector.hpp"
#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace compressed_coo {


template <typename IndexType>
void idxs_to_bits(std::shared_ptr<const DefaultExecutor> exec,
                  const IndexType* idxs, size_type nnz, uint32* bits,
                  IndexType* ranks)
{
    if (nnz == 0) {
        return;
    }
    const auto num_blocks = ceildiv(nnz, 32);
    run_kernel(
        exec,
        [] GKO_KERNEL(auto block, auto idxs, auto nnz, auto bits, auto ranks) {
            uint32 mask{};
            const auto begin = block * 32;
            const auto end = begin + 32 <= nnz ? begin + 32 : nnz;
            for (auto i = begin; i < end; i++) {
                if (i < nnz - 1 && idxs[i + 1] > idxs[i]) {
                    assert(idxs[i + 1] == idxs[i] + 1);
                    mask |= uint32{1} << (i % 32);
                }
            }
            bits[block] = mask;
            ranks[block] = gko::detail::popcount(mask);
        },
        num_blocks, idxs, static_cast<int64>(nnz), bits, ranks);
    components::prefix_sum_nonnegative(exec, ranks,
                                       static_cast<size_type>(num_blocks));
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_CRCOO_IDXS_TO_BITS_KERNEL);


template <typename IndexType>
void bits_to_idxs(std::shared_ptr<const DefaultExecutor> exec,
                  const uint32* bits, const IndexType* ranks, size_type nnz,
                  IndexType* idxs)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto bits, auto ranks, auto nnz, auto idxs) {
            device_bitvector<IndexType> bv{bits, ranks, nnz};
            idxs[i] = bv.rank(i);
        },
        static_cast<int64>(nnz), bits, ranks, static_cast<int64>(nnz), idxs);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_CRCOO_BITS_TO_IDXS_KERNEL);


}  // namespace compressed_coo
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
