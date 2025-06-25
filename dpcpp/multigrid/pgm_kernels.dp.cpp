// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <oneapi/dpl/algorithm>

#include "core/multigrid/pgm_kernels.hpp"

#include <memory>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>

#include "dpcpp/base/onedpl.hpp"
#include "dpcpp/components/atomic.dp.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The PGM solver namespace.
 *
 * @ingroup pgm
 */
namespace pgm {


template <typename IndexType>
void match_edge(std::shared_ptr<const DefaultExecutor> exec,
                const array<IndexType>& strongest_neighbor,
                array<IndexType>& agg)
{
    exec->get_queue()->submit([size = agg.get_size(), agg = agg.get_data(),
                               strongest_neighbor =
                                   strongest_neighbor.get_const_data()](
                                  sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::range<1>{static_cast<std::size_t>(size)},
            [=](sycl::id<1> idx_id) {
                auto tidx = static_cast<IndexType>(idx_id[0]);
                if (load(agg + tidx, sycl::memory_order_relaxed) != -1) {
                    return;
                }
                auto neighbor = strongest_neighbor[tidx];
                if (neighbor != -1 && strongest_neighbor[neighbor] == tidx &&
                    tidx <= neighbor) {
                    store(agg + tidx, tidx, sycl::memory_order_relaxed);
                    store(agg + neighbor, tidx, sycl::memory_order_relaxed);
                }
            });
    });
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_PGM_MATCH_EDGE_KERNEL);


template <typename IndexType>
void sort_agg(std::shared_ptr<const DefaultExecutor> exec, IndexType num,
              IndexType* row_idxs, IndexType* col_idxs)
{
    auto policy = onedpl_policy(exec);
    auto it = oneapi::dpl::make_zip_iterator(row_idxs, col_idxs);
    std::sort(policy, it, it + num, [](auto a, auto b) {
        return std::tie(std::get<0>(a), std::get<1>(a)) <
               std::tie(std::get<0>(b), std::get<1>(b));
    });
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_PGM_SORT_AGG_KERNEL);


template <typename ValueType, typename IndexType>
class coarse_coo_policy {};


template <typename ValueType, typename IndexType>
void compute_coarse_coo(std::shared_ptr<const DefaultExecutor> exec,
                        size_type fine_nnz, const IndexType* row_idxs,
                        const IndexType* col_idxs, const ValueType* vals,
                        matrix::Coo<ValueType, IndexType>* coarse_coo)
{
    // WORKAROUND: reduce_by_segment needs unique policy. Otherwise, dpcpp
    // throws same mangled name error. Related:
    // https://github.com/oneapi-src/oneDPL/issues/507
    auto policy = oneapi::dpl::execution::make_device_policy<
        coarse_coo_policy<ValueType, IndexType>>(*exec->get_queue());
    auto key_it = oneapi::dpl::make_zip_iterator(row_idxs, col_idxs);

    auto coarse_key_it = oneapi::dpl::make_zip_iterator(
        coarse_coo->get_row_idxs(), coarse_coo->get_col_idxs());

    oneapi::dpl::reduce_by_segment(
        policy, key_it, key_it + fine_nnz, vals, coarse_key_it,
        coarse_coo->get_values(),
        [](auto a, auto b) {
            return std::tie(std::get<0>(a), std::get<1>(a)) ==
                   std::tie(std::get<0>(b), std::get<1>(b));
        },
        [](auto a, auto b) { return a + b; });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PGM_COMPUTE_COARSE_COO);


}  // namespace pgm
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
