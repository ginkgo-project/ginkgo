// SPDX-FileCopyrightText: 2025 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/multigrid/pmis_kernels.hpp"

#include <random>

#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>

#include "common/unified/base/kernel_launch.hpp"
#include "common/unified/base/kernel_launch_reduction.hpp"
#include "core/base/array_access.hpp"
#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The Pmis namespace.
 *
 * @ingroup pmis
 */
namespace pmis {


// the number of threads working on the same row
constexpr int width = 32;


template <typename ValueType, typename IndexType>
void compute_row_maxabs(std::shared_ptr<const DefaultExecutor> exec,
                        const matrix::Csr<ValueType, IndexType>* csr,
                        remove_complex<ValueType>* row_maxabs)
{
    run_kernel_row_reduction(
        exec,
        [] GKO_KERNEL(auto row, auto tid, auto row_ptrs, auto col_idxs,
                      auto values) {
            auto maxabs = zero(abs(values[0]));
            for (auto idx = tid + row_ptrs[row]; idx < row_ptrs[row + 1];
                 idx += width) {
                if (row == col_idxs[idx]) {
                    continue;
                }
                maxabs = max(maxabs, abs(values[idx]));
            }
            return maxabs;
        },
        GKO_KERNEL_REDUCE_MAX(remove_complex<ValueType>), row_maxabs, 1,
        dim<2>{csr->get_size()[0], width}, csr->get_const_row_ptrs(),
        csr->get_const_col_idxs(), csr->get_const_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PMIS_COMPUTE_ROW_MAXABS_KERNEL);


template <typename ValueType, typename IndexType>
void compute_strong_dep_row(std::shared_ptr<const DefaultExecutor> exec,
                            const matrix::Csr<ValueType, IndexType>* csr,
                            const remove_complex<ValueType>* row_maxabs,
                            remove_complex<ValueType> strength_threshold,
                            IndexType* sparsity_rows)
{
    run_kernel_row_reduction(
        exec,
        [] GKO_KERNEL(auto row, auto tid, auto row_maxabs,
                      auto strength_threshold, auto row_ptrs, auto col_idxs,
                      auto values) {
            auto max_abs = row_maxabs[row];
            auto count = zero<IndexType>();
            if (max_abs == zero(max_abs)) {
                return count;
            }
            for (auto idx = tid + row_ptrs[row]; idx < row_ptrs[row + 1];
                 idx += width) {
                if (row == col_idxs[idx]) {
                    continue;
                }
                if (abs(values[idx]) >= strength_threshold * max_abs) {
                    count++;
                }
            }
            return count;
        },
        GKO_KERNEL_REDUCE_SUM(IndexType), sparsity_rows, 1,
        dim<2>{csr->get_size()[0], width}, row_maxabs, strength_threshold,
        csr->get_const_row_ptrs(), csr->get_const_col_idxs(),
        csr->get_const_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PMIS_COMPUTE_STRONG_DEP_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void compute_strong_dep(std::shared_ptr<const DefaultExecutor> exec,
                        const matrix::Csr<ValueType, IndexType>* csr,
                        const remove_complex<ValueType>* row_maxabs,
                        remove_complex<ValueType> strength_threshold,
                        matrix::SparsityCsr<ValueType, IndexType>* strong_dep)
{
    // we handle this by one thread per row. It might get improved if we use a
    // warp with popcount and prefix for a row.
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto row_maxabs, auto strength_threshold,
                      auto row_ptrs, auto col_idxs, auto values,
                      auto dep_row_ptrs, auto dep_col_idxs) {
            auto max_abs = row_maxabs[row];
            if (max_abs == zero(max_abs)) {
                return;
            }
            auto d_idx = dep_row_ptrs[row];
            for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
                const auto col = col_idxs[idx];
                if (row == col) {
                    continue;
                }
                if (abs(values[idx]) >= strength_threshold * max_abs) {
                    dep_col_idxs[d_idx] = col;
                    d_idx++;
                }
            }
        },
        csr->get_size()[0], row_maxabs, strength_threshold,
        csr->get_const_row_ptrs(), csr->get_const_col_idxs(),
        csr->get_const_values(), strong_dep->get_const_row_ptrs(),
        strong_dep->get_col_idxs());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PMIS_COMPUTE_STRONG_DEP_KERNEL);


template <typename ValueType, typename IndexType>
void initialize_weight_and_status(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::SparsityCsr<ValueType, IndexType>* trans_strong_dep,
    remove_complex<ValueType>* weight, int* status)
{
    auto num = trans_strong_dep->get_size()[0];
    array<float> random(exec, num);
    initialize_random_weight(exec, num, random.get_data());
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto row_ptrs, auto random, auto weight,
                      auto status) {
            using type = device_type<remove_complex<ValueType>>;
            auto w = static_cast<type>(row_ptrs[row + 1] - row_ptrs[row]);
            status[row] = (w == 0 ? 0 : -1);
            weight[row] = static_cast<type>(random[row]) + w;
        },
        num, trans_strong_dep->get_const_row_ptrs(), random.get_const_data(),
        weight, status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PMIS_INITIALIZE_WEIGHT_AND_STATUS_KERNEL);


template <typename ValueType, typename IndexType>
void classify(std::shared_ptr<const DefaultExecutor> exec,
              const remove_complex<ValueType>* weight,
              const matrix::SparsityCsr<ValueType, IndexType>* strong_dep,
              const matrix::SparsityCsr<ValueType, IndexType>* trans_strong_dep,
              const int* status, int* new_status)
{
    // mark coarse point
    run_kernel_row_reduction(
        exec,
        [] GKO_KERNEL(auto row, auto tid, auto status, auto weight,
                      auto row_ptrs, auto col_idxs) {
            auto ans = status[row];
            if (ans != -1) {
                return ans;
            }
            for (auto idx = tid + row_ptrs[row]; idx < row_ptrs[row + 1];
                 idx += width) {
                auto col = col_idxs[idx];
                if (row == col) {
                    continue;
                }
                if (status[col] == -1 && weight[col] >= weight[row]) {
                    return -1;
                }
            }
            return 1;
        },
        [] GKO_KERNEL(auto a, auto b) { return a < b ? a : b; } /* minimun */,
        [] GKO_KERNEL(auto a) { return a; }, int{1}, new_status, 1,
        dim<2>{csr->get_size()[0], width}, status, weight,
        csr->get_const_row_ptrs(), csr->get_const_col_idxs());
    // mark new fine point strongly influenced by the new coarse points
    // TODO: change to use strong_dep, which allows multiple read not multiple
    // write
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto tid, auto status, auto new_status,
                      auto trans_row_ptrs, auto trans_col_idxs) {
            if (new_status[row] != 1 || new_status[row] == status[row]) {
                return;
            }
            for (auto idx = tid + trans_row_ptrs[row];
                 idx < trans_row_ptrs[row + 1]; idx += width) {
                // It is correct even if more than one threads might
                // assign the
                // value
                auto col = trans_col_idxs[idx];
                if (new_status[col] == -1) {
                    new_status[col] = 0;
                }
            }
        },
        dim<2>{csr->get_size()[0], width}, status, new_status,
        trans_strong_dep->get_const_row_ptrs(),
        trans_strong_dep->get_const_col_idxs());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_PMIS_CLASSIFY_KERNEL);


void count(std::shared_ptr<const DefaultExecutor> exec, size_type num,
           const int* status, size_type* num_unassigned)
{
    array<size_type> d_result(exec, 1);
    run_kernel_reduction(
        exec,
        [] GKO_KERNEL(auto i, auto status) {
            return static_cast<size_type>(status[i] == -1);
        },
        GKO_KERNEL_REDUCE_SUM(size_type), d_result.get_data(), num, status);
    *num_unassigned = get_element(d_result, 0);
}


template <typename ValueType, typename IndexType>
void direct_interpolation_row_count(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::SparsityCsr<ValueType, IndexType>* strong_dep,
    const int* status, IndexType* prolong_row_ptr)
{
    run_kernel_row_reduction(
        exec,
        [] GKO_KERNEL(auto row, auto tid, auto status, auto row_ptrs,
                      auto col_idxs) {
            if (status[row] == 1) {
                return tid == 0 ? one<IndexType>() : zero<IndexType>();
            }
            auto count = zero<IndexType>();
            for (auto idx = tid + row_ptrs[row]; idx < row_ptrs[row + 1];
                 idx += width) {
                if (status[col_idxs[idx]] == 1) {
                    count++;
                }
            }
            return count;
        },
        GKO_KERNEL_REDUCE_SUM(IndexType), prolong_row_ptr, 1,
        dim<2>{strong_dep->get_size()[0], width}, status,
        strong_dep->get_const_row_ptrs(), strong_dep->get_const_col_idxs());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DIRECT_INTERPOLATION_ROW_COUNT);


template <typename ValueType, typename IndexType>
void direct_interpolation_fill(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* csr,
    const remove_complex<ValueType>* row_maxabs,
    const remove_complex<ValueType> strength_threshold,
    const IndexType* coarse_map, const IndexType* prolong_row_ptrs,
    IndexType* prolong_col_idxs, ValueType* prolong_values)
{
    // currently use one thread per row. It might get improved by using a warp
    // for row with prefix and popcount
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto row_maxabs, auto strength_threshold,
                      auto coarse_map, auto row_ptrs, auto col_idxs,
                      auto values, auto prolong_row_ptrs, auto prolong_col_idxs,
                      auto prolong_values) {
            if (coarse_map[row] != coarse_map[row + 1]) {
                auto idx = prolong_row_ptrs[row];
                prolong_col_idxs[idx] = coarse_map[row];
                prolong_values[idx] = one(prolong_values[idx]);
                return;
            }
            auto pos = zero(values[0]);
            auto pos_divisor = zero(values[0]);
            auto neg = zero(values[0]);
            auto neg_divisor = zero(values[0]);
            auto diag = zero(values[0]);
            bool enable_neg = false;
            bool enable_pos = false;
            // first compute alpha/beta
            auto max_abs = row_maxabs[row];
            for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
                auto val = values[idx];
                auto col = col_idxs[idx];
                if (col == row) {
                    diag = val;
                    continue;
                }
                if (real(val) >= 0) {
                    pos += val;
                    if (coarse_map[col] != coarse_map[col + 1] &&
                        abs(val) >= strength_threshold * max_abs) {
                        pos_divisor += val;
                        enable_pos = true;
                    }
                } else {
                    neg += val;
                    if (coarse_map[col] != coarse_map[col + 1] &&
                        abs(val) >= strength_threshold * max_abs) {
                        neg_divisor += val;
                        enable_neg = true;
                    }
                }
            }
            pos = safe_divide(pos, pos_divisor);
            neg = safe_divide(neg, neg_divisor);
            if (!enable_neg && !enable_pos) {
                return;
            }

            auto p_idx = prolong_row_ptrs[row];
            for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
                auto val = values[idx];
                auto col = col_idxs[idx];
                if (col == row || abs(val) < strength_threshold * max_abs) {
                    continue;
                }
                if (real(val) >= 0 && enable_pos &&
                    coarse_map[col] != coarse_map[col + 1]) {
                    prolong_col_idxs[p_idx] = coarse_map[col];
                    prolong_values[p_idx] = -pos * val / diag;
                    p_idx++;
                }
                if (real(val) < 0 && enable_neg &&
                    coarse_map[col] != coarse_map[col + 1]) {
                    prolong_col_idxs[p_idx] = coarse_map[col];
                    prolong_values[p_idx] = -neg * val / diag;
                    p_idx++;
                }
            }
        },
        csr->get_size()[0], row_maxabs, strength_threshold, coarse_map,
        csr->get_const_row_ptrs(), csr->get_const_col_idxs(),
        csr->get_const_values(), prolong_row_ptrs, prolong_col_idxs,
        prolong_values);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DIRECT_INTERPOLATION_FILL);


}  // namespace pmis
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
