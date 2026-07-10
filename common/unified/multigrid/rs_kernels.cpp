// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/multigrid/rs_kernels.hpp"

#include <ginkgo/core/base/math.hpp>

#include "common/unified/base/kernel_launch.hpp"
#include "common/unified/base/kernel_launch_reduction.hpp"
#include "core/base/array_access.hpp"
#include "core/components/prefix_sum_kernels.hpp"

namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The Rs namespace.
 *
 * @ingroup rs
 */
namespace rs {


template <typename ValueType, typename IndexType>
void check_m_matrix(std::shared_ptr<const DefaultExecutor> exec,
                    matrix::view::csr<const ValueType, const IndexType> matrix,
                    array<bool>& is_m_matrix_array)
{
    const auto num_rows = matrix->get_size()[0];
    const auto row_ptrs = matrix->get_const_row_ptrs();
    const auto col_idxs = matrix->get_const_col_idxs();
    const auto values = matrix->get_const_values();

    array<IndexType> d_result(exec, 1);
    run_kernel_reduction(
        exec,
        [] GKO_KERNEL(auto row, auto row_ptrs, auto col_idxs, auto values) {
            bool has_diag = false;
            bool valid = true;
            for (auto nz = row_ptrs[row]; nz < row_ptrs[row + 1]; ++nz) {
                const auto col = col_idxs[nz];
                const auto val = real(values[nz]);
                using real_t = decltype(val);

                if (row == col) {
                    has_diag = true;
                    if (val <= zero<real_t>()) valid = false;
                } else {
                    if (val > zero<real_t>()) valid = false;
                }
            }
            return (valid && has_diag) ? IndexType{1} : IndexType{0};
        },
        GKO_KERNEL_REDUCE_SUM(IndexType), d_result.get_data(), num_rows,
        row_ptrs, col_idxs, values);

    const IndexType count = get_element(d_result, 0);

    run_kernel(
        exec,
        [count, num_rows] GKO_KERNEL(auto tidx, auto is_m_matrix) {
            is_m_matrix[0] = (count == static_cast<IndexType>(num_rows));
        },
        1, is_m_matrix_array.get_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_RS_CHECK_M_MATRIX_KERNEL);


template <typename ValueType, typename IndexType>
void compute_soc_and_run_rs(
    std::shared_ptr<const DefaultExecutor> exec,
    matrix::view::csr<const ValueType, const IndexType> A, double theta,
    array<bool>& is_strong, array<IndexType>& lambda,
    array<IndexType>& cf_marker, IndexType& coarse_size)
{
    const auto n = A->get_size()[0];
    const auto* a_row_ptrs = A->get_const_row_ptrs();
    const auto* a_col_idxs = A->get_const_col_idxs();
    const auto* a_vals = A->get_const_values();
    bool* is_strong_vals = is_strong.get_data();
    auto* lambda_vals = lambda.get_data();
    auto* cf = cf_marker.get_data();

    /// 1. COMPUTE SoC MASK
    run_kernel(
        exec,
        [theta] GKO_KERNEL(auto i, auto row_ptrs, auto col_idxs, auto vals,
                           auto is_strong_vals) {
            auto max_offdiag = zero<decltype(real(vals[0]))>();
            using real_t = decltype(max_offdiag);
            const auto theta_cast = static_cast<real_t>(theta);
            for (auto jj = row_ptrs[i]; jj < row_ptrs[i + 1]; ++jj) {
                if (col_idxs[jj] != i) {
                    max_offdiag = gko::max(max_offdiag, -real(vals[jj]));
                }
            }
            for (auto jj = row_ptrs[i]; jj < row_ptrs[i + 1]; ++jj) {
                const auto j = col_idxs[jj];
                is_strong_vals[jj] =
                    (j != i && -real(vals[jj]) >= theta_cast * max_offdiag);
            }
        },
        n, a_row_ptrs, a_col_idxs, a_vals, is_strong_vals);

    /// 2. COMPUTE lambda_i = number of strong nbrs
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto row_ptrs, auto is_strong_vals,
                      auto lambda_vals) {
            IndexType count = 0;
            for (auto jj = row_ptrs[i]; jj < row_ptrs[i + 1]; ++jj) {
                if (is_strong_vals[jj]) {
                    count++;
                }
            }
            lambda_vals[i] = count;
        },
        n, a_row_ptrs, is_strong_vals, lambda_vals);

    /// 3. INIT ALL NODES AS UNDECIDED (0)
    run_kernel(
        exec, [] GKO_KERNEL(auto i, auto cf) { cf[i] = 0; },
        cf_marker.get_size(), cf);

    /// 4. RS-COARSENING (on host. inherently sequential algorithm?)
    {
        auto host_exec = exec->get_master();
        const auto nnz = A->get_num_stored_elements();

        // alternatives on host, initialise with these pointers first
        const auto* hr_ptrs = a_row_ptrs;
        const auto* hc_idxs = a_col_idxs;
        const auto* h_is_str = is_strong.get_const_data();
        auto* h_lam = lambda.get_data();
        auto* h_cf_v = cf_marker.get_data();

        // Copy device arrays to host if needed
        if (exec != host_exec) {
            array<IndexType> h_row_ptrs(host_exec, a_row_ptrs,
                                        a_row_ptrs + n + 1);
            array<IndexType> h_col_idxs(host_exec, a_col_idxs,
                                        a_col_idxs + nnz);
            array<bool> h_is_strong(host_exec, is_strong);
            array<IndexType> h_lambda(host_exec, lambda);
            array<IndexType> h_cf(host_exec, cf_marker);

            hr_ptrs = h_row_ptrs.get_const_data();
            hc_idxs = h_col_idxs.get_const_data();
            h_is_str = h_is_strong.get_const_data();
            h_lam = h_lambda.get_data();
            h_cf_v = h_cf.get_data();
        }

        while (true) {
            IndexType max_idx = -1;
            IndexType max_val = -1;
            for (IndexType i = 0; i < static_cast<IndexType>(n); ++i) {
                if (h_cf_v[i] == 0 && h_lam[i] > max_val) {
                    max_val = h_lam[i];
                    max_idx = i;
                }
            }
            if (max_idx == -1) break;

            h_cf_v[max_idx] = 1;  // C-point
            for (auto jj = hr_ptrs[max_idx]; jj < hr_ptrs[max_idx + 1]; ++jj) {
                if (h_is_str[jj]) {
                    const auto j = hc_idxs[jj];
                    if (h_cf_v[j] == 0) {
                        h_cf_v[j] = -1;  // F-point
                        for (auto kk = hr_ptrs[j]; kk < hr_ptrs[j + 1]; ++kk) {
                            if (h_is_str[kk] && h_cf_v[hc_idxs[kk]] == 0) {
                                h_lam[hc_idxs[kk]]--;
                            }
                        }
                    }
                }
            }
        }

        // Copy results back to device
        cf_marker = array<IndexType>(exec, cf_marker.get_size(), h_cf_v);
        cf = cf_marker.get_data();
    }

    /// 5. CLEANUP, MAKE SURE NO UNDECIDED REMAIN
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto cf) {
            if (cf[i] == 0) {
                cf[i] = -1;
            }
        },
        cf_marker.get_size(), cf);

    /// 6. COUNT C-POINTS
    array<IndexType> d_coarse_size(exec, 1);
    run_kernel_reduction(
        exec,
        [] GKO_KERNEL(auto i, auto cf) {
            return cf[i] == 1 ? IndexType{1} : IndexType{0};
        },
        GKO_KERNEL_REDUCE_SUM(IndexType), d_coarse_size.get_data(),
        cf_marker.get_size(), cf);

    coarse_size = get_element(d_coarse_size, 0);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_RS_COMPUTE_SOC_AND_RUN_RS_KERNEL);


template <typename ValueType, typename IndexType>
void fill_coarse_and_compute_prolong_row_ptrs(
    std::shared_ptr<const DefaultExecutor> exec,
    const array<IndexType>& cf_marker, array<IndexType>& coarse_rows,
    array<IndexType>& fine_to_coarse,
    matrix::view::csr<const ValueType, const IndexType> A,
    const array<bool>& is_strong, array<IndexType>& row_ptrs)
{
    const auto* cf = cf_marker.get_const_data();
    auto* coarse_rows_vals = coarse_rows.get_data();
    auto* fine_to_coarse_vals = fine_to_coarse.get_data();
    auto* row_ptrs_vals = row_ptrs.get_data();
    const bool* is_strong_vals = is_strong.get_const_data();
    const auto n = A->get_size()[0];
    const auto* a_row_ptrs = A->get_const_row_ptrs();
    const auto* a_col_idxs = A->get_const_col_idxs();
    const auto num_elements = cf_marker.get_size();

    /// 1 & 2. PARALLELISED COARSE MAPPING VIA EXCLUSIVE PREFIX SUM
    array<IndexType> coarse_map(exec, num_elements + 1);
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto cf, auto coarse_map) {
            coarse_map[i] = (cf[i] == 1) ? 1 : 0;
        },
        num_elements, cf, coarse_map.get_data());

    run_kernel(
        exec, [] GKO_KERNEL(auto i, auto coarse_map) { coarse_map[i] = 0; }, 1,
        coarse_map.get_data() + num_elements);

    components::prefix_sum_nonnegative(exec, coarse_map.get_data(),
                                       coarse_map.get_size());

    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto cf, auto coarse_map, auto fine_to_coarse,
                      auto coarse_rows) {
            if (cf[i] == 1) {
                auto coarse_id = coarse_map[i];
                fine_to_coarse[i] = coarse_id;
                coarse_rows[coarse_id] = static_cast<IndexType>(i);
            } else {
                fine_to_coarse[i] = -1;
            }
        },
        num_elements, cf, coarse_map.get_const_data(), fine_to_coarse_vals,
        coarse_rows_vals);

    /// 3. COMPUTE INTERPOLATION ROW PTRS
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto cf, auto a_row_ptrs, auto a_col_idxs,
                      auto is_strong_vals, auto row_ptrs) {
            IndexType row_nnz = 0;
            if (cf[i] == 1) {
                row_nnz = 1;
            } else {
                for (auto jj = a_row_ptrs[i]; jj < a_row_ptrs[i + 1]; ++jj) {
                    if (is_strong_vals[jj] && cf[a_col_idxs[jj]] == 1) {
                        row_nnz++;
                    }
                }
            }
            row_ptrs[i] = row_nnz;
        },
        n, cf, a_row_ptrs, a_col_idxs, is_strong_vals, row_ptrs_vals);

    run_kernel(
        exec, [] GKO_KERNEL(auto i, auto row_ptrs) { row_ptrs[0] = 0; }, 1,
        row_ptrs_vals + n);

    components::prefix_sum_nonnegative(exec, row_ptrs_vals, n + 1);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_RS_FILL_COARSE_AND_COMPUTE_PROLONG_ROW_PTRS_KERNEL);


template <typename ValueType, typename IndexType>
void compute_interpolation(
    std::shared_ptr<const DefaultExecutor> exec,
    matrix::view::csr<const ValueType, const IndexType> A,
    const bool* is_strong, const array<IndexType>& cf_marker,
    const IndexType* fine_to_coarse, matrix::view::csr<ValueType, IndexType> P)
{
    const auto n = A->get_size()[0];
    const auto* a_row_ptrs = A->get_const_row_ptrs();
    const auto* a_col_idxs = A->get_const_col_idxs();
    const auto* a_vals = A->get_const_values();
    const auto* cf = cf_marker.get_const_data();
    auto* p_row_ptrs = P->get_const_row_ptrs();
    auto* p_col_idxs = P->get_col_idxs();
    auto* p_vals = P->get_values();

    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto a_row_ptrs, auto a_col_idxs, auto a_vals,
                      auto is_strong, auto cf, auto fine_to_coarse,
                      auto p_row_ptrs, auto p_col_idxs, auto p_vals) {
            auto p_idx = p_row_ptrs[i];
            using value_type = device_type<ValueType>;
            if (cf[i] == 1) {
                p_col_idxs[p_idx] = fine_to_coarse[i];
                p_vals[p_idx] = one<value_type>();
            } else {
                auto diag = zero<value_type>();
                auto sum_weak = zero<value_type>();
                auto sum_strong_c_val = zero<value_type>();

                for (auto jj = a_row_ptrs[i]; jj < a_row_ptrs[i + 1]; ++jj) {
                    auto j = a_col_idxs[jj];
                    if (i == j) {
                        diag = a_vals[jj];
                    } else if (!is_strong[jj]) {
                        sum_weak += a_vals[jj];
                    } else if (cf[j] == 1) {
                        sum_strong_c_val += a_vals[jj];
                    }
                }

                auto denominator = diag + sum_weak;

                for (auto jj = a_row_ptrs[i]; jj < a_row_ptrs[i + 1]; ++jj) {
                    if (is_strong[jj] && cf[a_col_idxs[jj]] == 1) {
                        auto j = a_col_idxs[jj];
                        auto numerator = a_vals[jj];

                        for (auto kk = a_row_ptrs[i]; kk < a_row_ptrs[i + 1];
                             ++kk) {
                            if (is_strong[kk] && cf[a_col_idxs[kk]] == -1) {
                                auto k = a_col_idxs[kk];
                                auto a_ik = a_vals[kk];
                                auto a_kj = zero<value_type>();
                                for (auto n_kj = a_row_ptrs[k];
                                     n_kj < a_row_ptrs[k + 1]; ++n_kj) {
                                    if (a_col_idxs[n_kj] == j) {
                                        a_kj = a_vals[n_kj];
                                        break;
                                    }
                                }
                                numerator += (a_ik * a_kj) / sum_strong_c_val;
                            }
                        }
                        p_col_idxs[p_idx] = fine_to_coarse[j];
                        p_vals[p_idx] = -numerator / denominator;
                        p_idx++;
                    }
                }
            }
        },
        n, a_row_ptrs, a_col_idxs, a_vals, is_strong, cf, fine_to_coarse,
        p_row_ptrs, p_col_idxs, p_vals);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_RS_COMPUTE_INTERPOLATION_KERNEL);


}  // namespace rs
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
