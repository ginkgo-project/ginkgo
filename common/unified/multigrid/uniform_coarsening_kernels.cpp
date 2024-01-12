// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/multigrid/uniform_coarsening_kernels.hpp"


#include <ginkgo/core/base/math.hpp>


#include "common/unified/base/kernel_launch.hpp"
#include "common/unified/base/kernel_launch_reduction.hpp"
#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The UniformCoarsening namespace.
 *
 * @ingroup uniform_coarsening
 */
namespace uniform_coarsening {


template <typename ValueType, typename IndexType>
void fill_restrict_op(std::shared_ptr<const DefaultExecutor> exec,
                      const array<IndexType>* coarse_rows,
                      matrix::Csr<ValueType, IndexType>* restrict_op)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto tidx, const auto coarse_data,
                      auto restrict_col_idxs) {
            if (coarse_data[tidx] >= 0) {
                restrict_col_idxs[coarse_data[tidx]] = tidx;
            }
        },
        coarse_rows->get_num_elems(), coarse_rows->get_const_data(),
        restrict_op->get_col_idxs());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_UNIFORM_COARSENING_FILL_RESTRICT_OP);


template <typename IndexType>
void fill_incremental_indices(std::shared_ptr<const DefaultExecutor> exec,
                              size_type num_jumps,
                              array<IndexType>* coarse_rows)
{
    IndexType num_elems = (coarse_rows->get_num_elems());
    run_kernel(
        exec,
        [] GKO_KERNEL(auto tidx, auto num_jumps, auto coarse_data, auto size) {
            if (tidx % num_jumps == 0 && tidx < size) {
                coarse_data[tidx] = tidx / num_jumps;
            }
        },
        num_elems, num_jumps, coarse_rows->get_data(), num_elems);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_UNIFORM_COARSENING_FILL_INCREMENTAL_INDICES);


}  // namespace uniform_coarsening
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
