// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/par_ilut_kernels.hpp"


#include <algorithm>
#include <tuple>
#include <unordered_map>
#include <unordered_set>


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/utils.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/coo_builder.hpp"
#include "core/matrix/csr_builder.hpp"
#include "reference/components/csr_spgeam.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The parallel ilut factorization namespace.
 *
 * @ingroup factor
 */
namespace par_ilut_factorization {


/**
 * @internal
 *
 * Selects the `rank`th smallest element (0-based, magnitude-wise)
 * from the values of `m`. It uses two temporary arrays.
 */
template <typename ValueType, typename IndexType>
void threshold_select(std::shared_ptr<const DefaultExecutor> exec,
                      const matrix::Csr<ValueType, IndexType>* m,
                      IndexType rank, array<ValueType>& tmp,
                      array<remove_complex<ValueType>>&,
                      remove_complex<ValueType>& threshold)
{
    auto values = m->get_const_values();
    IndexType size = m->get_num_stored_elements();
    tmp.resize_and_reset(size);
    std::copy_n(values, size, tmp.get_data());

    auto begin = tmp.get_data();
    auto target = begin + rank;
    auto end = begin + size;
    std::nth_element(begin, target, end,
                     [](ValueType a, ValueType b) { return abs(a) < abs(b); });
    threshold = abs(*target);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILUT_THRESHOLD_SELECT_KERNEL);


/**
 * Removes all the elements from the input matrix for which pred is false.
 * Stores the result in m_out and (if non-null) m_out_coo.
 * pred(row, nz) is called for each entry, where nz is the index in
 * values/col_idxs.
 */
template <typename Predicate, typename ValueType, typename IndexType>
void abstract_filter(std::shared_ptr<const DefaultExecutor> exec,
                     const matrix::Csr<ValueType, IndexType>* m,
                     matrix::Csr<ValueType, IndexType>* m_out,
                     matrix::Coo<ValueType, IndexType>* m_out_coo,
                     Predicate pred)
{
    auto num_rows = m->get_size()[0];
    auto row_ptrs = m->get_const_row_ptrs();
    auto col_idxs = m->get_const_col_idxs();
    auto vals = m->get_const_values();

    // first sweep: count nnz for each row
    auto new_row_ptrs = m_out->get_row_ptrs();
    for (size_type row = 0; row < num_rows; ++row) {
        IndexType count{};
        for (auto nz = row_ptrs[row]; nz < row_ptrs[row + 1]; ++nz) {
            count += pred(row, nz);
        }
        new_row_ptrs[row] = count;
    }

    // build row pointers
    components::prefix_sum_nonnegative(exec, new_row_ptrs, num_rows + 1);

    // second sweep: accumulate non-zeros
    auto new_nnz = new_row_ptrs[num_rows];
    // resize arrays and update aliases
    matrix::CsrBuilder<ValueType, IndexType> builder{m_out};
    builder.get_col_idx_array().resize_and_reset(new_nnz);
    builder.get_value_array().resize_and_reset(new_nnz);
    auto new_col_idxs = m_out->get_col_idxs();
    auto new_vals = m_out->get_values();
    IndexType* new_row_idxs{};
    if (m_out_coo) {
        matrix::CooBuilder<ValueType, IndexType> coo_builder{m_out_coo};
        coo_builder.get_row_idx_array().resize_and_reset(new_nnz);
        coo_builder.get_col_idx_array() =
            make_array_view(exec, new_nnz, new_col_idxs);
        coo_builder.get_value_array() =
            make_array_view(exec, new_nnz, new_vals);
        new_row_idxs = m_out_coo->get_row_idxs();
    }

    for (size_type row = 0; row < num_rows; ++row) {
        auto new_nz = new_row_ptrs[row];
        auto begin = row_ptrs[row];
        auto end = row_ptrs[row + 1];
        for (auto nz = begin; nz < end; ++nz) {
            if (pred(row, nz)) {
                if (new_row_idxs) {
                    new_row_idxs[new_nz] = row;
                }
                new_col_idxs[new_nz] = col_idxs[nz];
                new_vals[new_nz] = vals[nz];
                ++new_nz;
            }
        }
    }
}


/**
 * @internal
 *
 * Removes all elements below the given threshold from a matrix.
 */
template <typename ValueType, typename IndexType>
void threshold_filter(std::shared_ptr<const DefaultExecutor> exec,
                      const matrix::Csr<ValueType, IndexType>* m,
                      remove_complex<ValueType> threshold,
                      matrix::Csr<ValueType, IndexType>* m_out,
                      matrix::Coo<ValueType, IndexType>* m_out_coo, bool)
{
    auto col_idxs = m->get_const_col_idxs();
    auto vals = m->get_const_values();
    abstract_filter(
        exec, m, m_out, m_out_coo, [&](IndexType row, IndexType nz) {
            return abs(vals[nz]) >= threshold || col_idxs[nz] == row;
        });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILUT_THRESHOLD_FILTER_KERNEL);


constexpr auto bucket_count = 1 << sampleselect_searchtree_height;
constexpr auto sample_size = bucket_count * sampleselect_oversampling;


/**
 * @internal
 *
 * Approximately selects the `rank`th smallest element as a threshold
 * and removes all elements below this threshold from the input matrix.
 */
template <typename ValueType, typename IndexType>
void threshold_filter_approx(std::shared_ptr<const DefaultExecutor> exec,
                             const matrix::Csr<ValueType, IndexType>* m,
                             IndexType rank, array<ValueType>& tmp,
                             remove_complex<ValueType>& threshold,
                             matrix::Csr<ValueType, IndexType>* m_out,
                             matrix::Coo<ValueType, IndexType>* m_out_coo)
{
    auto vals = m->get_const_values();
    auto col_idxs = m->get_const_col_idxs();
    auto size = static_cast<IndexType>(m->get_num_stored_elements());
    using AbsType = remove_complex<ValueType>;
    constexpr auto storage_size = ceildiv(
        sample_size * sizeof(AbsType) + bucket_count * sizeof(IndexType),
        sizeof(ValueType));
    tmp.resize_and_reset(storage_size);
    // pick and sort sample
    auto sample = reinterpret_cast<AbsType*>(tmp.get_data());
    // assuming rounding towards zero
    auto stride = double(size) / sample_size;
    for (IndexType i = 0; i < sample_size; ++i) {
        sample[i] = abs(vals[static_cast<IndexType>(i * stride)]);
    }
    std::sort(sample, sample + sample_size);
    // pick splitters
    for (IndexType i = 0; i < bucket_count - 1; ++i) {
        // shift by one so we get upper bounds for the buckets
        sample[i] = sample[(i + 1) * sampleselect_oversampling];
    }
    // count elements per bucket
    auto histogram = reinterpret_cast<IndexType*>(sample + bucket_count);
    for (IndexType bucket = 0; bucket < bucket_count; ++bucket) {
        histogram[bucket] = 0;
    }
    for (IndexType nz = 0; nz < size; ++nz) {
        auto bucket_it =
            std::upper_bound(sample, sample + bucket_count - 1, abs(vals[nz]));
        auto bucket = std::distance(sample, bucket_it);
        // smallest bucket s.t. sample[bucket] >= abs(val[nz])
        histogram[bucket]++;
    }
    // determine splitter ranks: prefix sum over bucket counts
    components::prefix_sum_nonnegative(exec, histogram, bucket_count + 1);
    // determine the bucket containing the threshold rank:
    // prefix_sum[bucket] <= rank < prefix_sum[bucket + 1]
    auto it = std::upper_bound(histogram, histogram + bucket_count + 1, rank);
    auto threshold_bucket = std::distance(histogram + 1, it);
    // sample contains upper bounds for the buckets
    threshold = threshold_bucket > 0 ? sample[threshold_bucket - 1]
                                     : zero<remove_complex<ValueType>>();
    // filter elements
    abstract_filter(
        exec, m, m_out, m_out_coo, [&](IndexType row, IndexType nz) {
            return abs(vals[nz]) >= threshold || col_idxs[nz] == row;
        });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILUT_THRESHOLD_FILTER_APPROX_KERNEL);


/**
 * @internal
 *
 * Computes a ParILUT sweep on the input matrices.
 */
template <typename ValueType, typename IndexType>
void compute_l_u_factors(std::shared_ptr<const DefaultExecutor> exec,
                         const matrix::Csr<ValueType, IndexType>* a,
                         matrix::Csr<ValueType, IndexType>* l,
                         const matrix::Coo<ValueType, IndexType>*,
                         matrix::Csr<ValueType, IndexType>* u,
                         const matrix::Coo<ValueType, IndexType>*,
                         matrix::Csr<ValueType, IndexType>* u_csc)
{
    auto num_rows = a->get_size()[0];
    auto l_row_ptrs = l->get_const_row_ptrs();
    auto l_col_idxs = l->get_const_col_idxs();
    auto l_vals = l->get_values();
    auto u_row_ptrs = u->get_const_row_ptrs();
    auto u_col_idxs = u->get_const_col_idxs();
    auto u_vals = u->get_values();
    auto ut_col_ptrs = u_csc->get_const_row_ptrs();
    auto ut_row_idxs = u_csc->get_const_col_idxs();
    auto ut_vals = u_csc->get_values();
    auto a_row_ptrs = a->get_const_row_ptrs();
    auto a_col_idxs = a->get_const_col_idxs();
    auto a_vals = a->get_const_values();

    auto compute_sum = [&](IndexType row, IndexType col) {
        // find value from A
        auto a_begin = a_row_ptrs[row];
        auto a_end = a_row_ptrs[row + 1];
        auto a_nz_it =
            std::lower_bound(a_col_idxs + a_begin, a_col_idxs + a_end, col);
        auto a_nz = std::distance(a_col_idxs, a_nz_it);
        auto has_a = a_nz < a_end && a_col_idxs[a_nz] == col;
        auto a_val = has_a ? a_vals[a_nz] : zero<ValueType>();
        // accumulate l(row,:) * u(:,col) without the last entry (row, col)
        ValueType sum{};
        IndexType ut_nz{};
        auto l_begin = l_row_ptrs[row];
        auto l_end = l_row_ptrs[row + 1];
        auto u_begin = ut_col_ptrs[col];
        auto u_end = ut_col_ptrs[col + 1];
        auto last_entry = min(row, col);
        while (l_begin < l_end && u_begin < u_end) {
            auto l_col = l_col_idxs[l_begin];
            auto u_row = ut_row_idxs[u_begin];
            if (l_col == u_row && l_col < last_entry) {
                sum += l_vals[l_begin] * ut_vals[u_begin];
            }
            if (u_row == row) {
                ut_nz = u_begin;
            }
            l_begin += (l_col <= u_row);
            u_begin += (u_row <= l_col);
        }
        return std::make_pair(a_val - sum, ut_nz);
    };

    for (size_type row = 0; row < num_rows; ++row) {
        for (size_type l_nz = l_row_ptrs[row]; l_nz < l_row_ptrs[row + 1] - 1;
             ++l_nz) {
            auto col = l_col_idxs[l_nz];
            auto u_diag = ut_vals[ut_col_ptrs[col + 1] - 1];
            auto new_val = compute_sum(row, col).first / u_diag;
            if (is_finite(new_val)) {
                l_vals[l_nz] = new_val;
            }
        }
        for (size_type u_nz = u_row_ptrs[row]; u_nz < u_row_ptrs[row + 1];
             ++u_nz) {
            auto col = u_col_idxs[u_nz];
            auto result = compute_sum(row, col);
            auto new_val = result.first;
            auto ut_nz = result.second;
            if (is_finite(new_val)) {
                u_vals[u_nz] = new_val;
                ut_vals[ut_nz] = new_val;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILUT_COMPUTE_LU_FACTORS_KERNEL);


/**
 * @internal
 *
 * Adds new entries from the sparsity pattern of A - L * U
 * to L and U, where new values are chosen based on the residual
 * value divided by the corresponding diagonal entry.
 */
template <typename ValueType, typename IndexType>
void add_candidates(std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Csr<ValueType, IndexType>* lu,
                    const matrix::Csr<ValueType, IndexType>* a,
                    const matrix::Csr<ValueType, IndexType>* l,
                    const matrix::Csr<ValueType, IndexType>* u,
                    matrix::Csr<ValueType, IndexType>* l_new,
                    matrix::Csr<ValueType, IndexType>* u_new)
{
    auto num_rows = a->get_size()[0];
    auto l_row_ptrs = l->get_const_row_ptrs();
    auto l_col_idxs = l->get_const_col_idxs();
    auto l_vals = l->get_const_values();
    auto u_row_ptrs = u->get_const_row_ptrs();
    auto u_col_idxs = u->get_const_col_idxs();
    auto u_vals = u->get_const_values();
    auto l_new_row_ptrs = l_new->get_row_ptrs();
    auto u_new_row_ptrs = u_new->get_row_ptrs();
    constexpr auto sentinel = std::numeric_limits<IndexType>::max();
    // count nnz
    IndexType l_nnz{};
    IndexType u_nnz{};
    abstract_spgeam(
        a, lu,
        [&](IndexType row) {
            l_new_row_ptrs[row] = l_nnz;
            u_new_row_ptrs[row] = u_nnz;
            return 0;
        },
        [&](IndexType row, IndexType col, ValueType, ValueType, int) {
            l_nnz += col <= row;
            u_nnz += col >= row;
        },
        [](IndexType, int) {});
    l_new_row_ptrs[num_rows] = l_nnz;
    u_new_row_ptrs[num_rows] = u_nnz;

    // resize arrays
    matrix::CsrBuilder<ValueType, IndexType> l_builder{l_new};
    matrix::CsrBuilder<ValueType, IndexType> u_builder{u_new};
    l_builder.get_col_idx_array().resize_and_reset(l_nnz);
    l_builder.get_value_array().resize_and_reset(l_nnz);
    u_builder.get_col_idx_array().resize_and_reset(u_nnz);
    u_builder.get_value_array().resize_and_reset(u_nnz);
    auto l_new_col_idxs = l_new->get_col_idxs();
    auto l_new_vals = l_new->get_values();
    auto u_new_col_idxs = u_new->get_col_idxs();
    auto u_new_vals = u_new->get_values();

    // accumulate non-zeros
    struct row_state {
        IndexType l_new_nz;
        IndexType u_new_nz;
        IndexType l_old_begin;
        IndexType l_old_end;
        IndexType u_old_begin;
        IndexType u_old_end;
        bool finished_l;
    };
    abstract_spgeam(
        a, lu,
        [&](IndexType row) {
            row_state state{};
            state.l_new_nz = l_new_row_ptrs[row];
            state.u_new_nz = u_new_row_ptrs[row];
            state.l_old_begin = l_row_ptrs[row];
            state.l_old_end = l_row_ptrs[row + 1] - 1;  // skip diagonal
            state.u_old_begin = u_row_ptrs[row];
            state.u_old_end = u_row_ptrs[row + 1];
            state.finished_l = (state.l_old_begin == state.l_old_end);
            return state;
        },
        [&](IndexType row, IndexType col, ValueType a_val, ValueType lu_val,
            row_state& state) {
            auto r_val = a_val - lu_val;
            // load matching entry of L + U
            auto lpu_col = state.finished_l
                               ? checked_load(u_col_idxs, state.u_old_begin,
                                              state.u_old_end, sentinel)
                               : l_col_idxs[state.l_old_begin];
            auto lpu_val =
                state.finished_l
                    ? checked_load(u_vals, state.u_old_begin, state.u_old_end,
                                   zero<ValueType>())
                    : l_vals[state.l_old_begin];
            // load diagonal entry of U for lower diagonal entries
            auto diag = col < row ? u_vals[u_row_ptrs[col]] : one<ValueType>();
            // if there is already an entry present, use that instead.
            auto out_val = lpu_col == col ? lpu_val : r_val / diag;
            // store output entries
            if (row >= col) {
                l_new_col_idxs[state.l_new_nz] = col;
                l_new_vals[state.l_new_nz] =
                    row == col ? one<ValueType>() : out_val;
                state.l_new_nz++;
            }
            if (row <= col) {
                u_new_col_idxs[state.u_new_nz] = col;
                u_new_vals[state.u_new_nz] = out_val;
                state.u_new_nz++;
            }
            // advance entry of L + U if we used it
            if (state.finished_l) {
                state.u_old_begin += (lpu_col == col);
            } else {
                state.l_old_begin += (lpu_col == col);
                state.finished_l = (state.l_old_begin == state.l_old_end);
            }
        },
        [](IndexType, row_state) {});
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILUT_ADD_CANDIDATES_KERNEL);


}  // namespace par_ilut_factorization
}  // namespace reference
}  // namespace kernels
}  // namespace gko
