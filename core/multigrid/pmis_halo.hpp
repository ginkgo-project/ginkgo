// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MULTIGRID_PMIS_HALO_HPP_
#define GKO_CORE_MULTIGRID_PMIS_HALO_HPP_


#include <ginkgo/config.hpp>

#if GINKGO_BUILD_MPI


#include <algorithm>
#include <memory>
#include <vector>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/distributed/collective_communicator.hpp>
#include <ginkgo/core/distributed/index_map.hpp>
#include <ginkgo/core/matrix/csr.hpp>


namespace gko {
namespace multigrid {
namespace pmis {


/**
 * Fetches one row per halo node from its owner, in the extended local
 * numbering.
 *
 * Interpolation richer than direct needs the halo nodes' rows, not just their
 * values, so the rows named by the row gatherer's send indices are exchanged
 * over its communication pattern, resized to each row's length. Columns
 * outside [0, n_ext) are second-hop nodes this rank cannot address and are
 * dropped, which is the relaxation that defines this variant.
 *
 * @note Collective over `coll_comm`'s base communicator; every rank must call
 *       it, including one with an empty halo.
 *
 * @param exec  the executor the returned matrix is created on
 * @param coll_comm  the row gatherer's collective communicator
 * @param imap  the matrix's index map, which defines the extended numbering
 * @param diag  the diagonal block, n_loc x n_loc
 * @param off_diag  the off-diagonal block, n_loc x n_halo
 * @param send_idxs  the row gatherer's send indices, on `exec`
 * @param n_send  the number of send indices
 *
 * @return an (n_halo x n_ext) Csr on `exec` whose row (k - n_loc) holds halo
 *         node k's row, restricted to the addressable columns and sorted
 *         ascending by column
 */
template <typename ValueType, typename IndexType, typename GlobalIndexType>
std::unique_ptr<matrix::Csr<ValueType, IndexType>> fetch_halo_rows(
    std::shared_ptr<const Executor> exec,
    const experimental::mpi::CollectiveCommunicator* coll_comm,
    const experimental::distributed::index_map<IndexType, GlobalIndexType>&
        imap,
    const matrix::Csr<ValueType, IndexType>* diag,
    const matrix::Csr<ValueType, IndexType>* off_diag,
    const IndexType* send_idxs, size_type n_send)
{
    using csr_type = matrix::Csr<ValueType, IndexType>;
    using index_space = experimental::distributed::index_space;
    using comm_index_type = experimental::mpi::comm_index_type;
    constexpr auto invalid = invalid_index<IndexType>();

    auto host = exec->get_master();
    const auto n_loc = diag->get_size()[0];
    const auto n_halo = off_diag->get_size()[1];
    const auto n_ext = n_loc + n_halo;
    // checked before the first collective, so no rank is left blocked in the
    // next one
    GKO_ASSERT_EQ(static_cast<size_type>(coll_comm->get_recv_size()), n_halo);

    // the packing and the rebuild below both run on the host
    auto diag_host = make_temporary_clone(host, diag);
    auto off_diag_host = make_temporary_clone(host, off_diag);
    const auto diag_row_ptrs = diag_host->get_const_row_ptrs();
    const auto diag_col_idxs = diag_host->get_const_col_idxs();
    const auto diag_vals = diag_host->get_const_values();
    const auto offd_row_ptrs = off_diag_host->get_const_row_ptrs();
    const auto offd_col_idxs = off_diag_host->get_const_col_idxs();
    const auto offd_vals = off_diag_host->get_const_values();

    array<IndexType> host_send_idxs{host, n_send};
    if (n_send > 0) {
        host->copy_from(exec, n_send, send_idxs, host_send_idxs.get_data());
    }
    const auto send_rows = host_send_idxs.get_const_data();

    // pack in the extended numbering: an off-diagonal column c becomes n_loc +
    // c, so one map_to_global call covers both blocks
    std::vector<comm_index_type> send_nnz_counts(n_send);
    size_type total_send_nnz = 0;
    for (size_type i = 0; i < n_send; i++) {
        const auto row = static_cast<size_type>(send_rows[i]);
        const auto nnz = (diag_row_ptrs[row + 1] - diag_row_ptrs[row]) +
                         (offd_row_ptrs[row + 1] - offd_row_ptrs[row]);
        send_nnz_counts[i] = static_cast<comm_index_type>(nnz);
        total_send_nnz += static_cast<size_type>(nnz);
    }
    std::vector<IndexType> send_combined_cols;
    std::vector<ValueType> send_vals;
    // one reserved element keeps data() non-null on a rank with nothing to send
    send_combined_cols.reserve(std::max<size_type>(total_send_nnz, 1));
    send_vals.reserve(std::max<size_type>(total_send_nnz, 1));
    for (size_type i = 0; i < n_send; i++) {
        const auto row = static_cast<size_type>(send_rows[i]);
        for (auto k = diag_row_ptrs[row]; k < diag_row_ptrs[row + 1]; k++) {
            send_combined_cols.push_back(diag_col_idxs[k]);
            send_vals.push_back(diag_vals[k]);
        }
        for (auto k = offd_row_ptrs[row]; k < offd_row_ptrs[row + 1]; k++) {
            send_combined_cols.push_back(static_cast<IndexType>(n_loc) +
                                         offd_col_idxs[k]);
            send_vals.push_back(offd_vals[k]);
        }
    }

    // map_to_global runs on the index map's executor
    auto imap_exec = imap.get_executor();
    const array<IndexType> combined_cols{imap_exec, send_combined_cols.begin(),
                                         send_combined_cols.end()};
    const auto send_global_cols_dev =
        imap.map_to_global(combined_cols, index_space::combined);
    auto send_global_cols = make_temporary_clone(host, &send_global_cols_dev);

    // resizing keeps the ordering inside the row gatherer's communicator
    auto [resized_comm, recv_nnz_counts] =
        coll_comm->resize(exec, send_nnz_counts);
    const auto total_recv_nnz =
        static_cast<size_type>(resized_comm->get_recv_size());

    std::vector<GlobalIndexType> recv_global_cols(
        std::max<size_type>(total_recv_nnz, 1));
    resized_comm
        ->i_all_to_all_v(host, send_global_cols->get_const_data(),
                         recv_global_cols.data())
        .wait();
    std::vector<ValueType> recv_vals(std::max<size_type>(total_recv_nnz, 1));
    resized_comm->i_all_to_all_v(host, send_vals.data(), recv_vals.data())
        .wait();

    // a column neither owned nor in the halo comes back as invalid_index, which
    // is how second-hop nodes drop out
    const array<GlobalIndexType> recv_global_array{
        imap_exec, recv_global_cols.begin(),
        recv_global_cols.begin() + total_recv_nnz};
    const auto recv_local_dev =
        imap.map_to_local(recv_global_array, index_space::combined);
    auto recv_local = make_temporary_clone(host, &recv_local_dev);
    const auto mapped = recv_local->get_const_data();

    std::vector<IndexType> row_ptrs(n_halo + 1, IndexType{});
    size_type offset = 0;
    for (size_type r = 0; r < n_halo; r++) {
        IndexType kept = 0;
        for (comm_index_type k = 0; k < recv_nnz_counts[r]; k++) {
            if (mapped[offset + static_cast<size_type>(k)] != invalid) {
                kept++;
            }
        }
        row_ptrs[r + 1] = row_ptrs[r] + kept;
        offset += static_cast<size_type>(recv_nnz_counts[r]);
    }

    const auto nnz = static_cast<size_type>(row_ptrs[n_halo]);
    std::vector<IndexType> col_idxs(nnz);
    std::vector<ValueType> values(nnz);
    std::vector<size_type> order;
    offset = 0;
    for (size_type r = 0; r < n_halo; r++) {
        order.clear();
        for (comm_index_type k = 0; k < recv_nnz_counts[r]; k++) {
            const auto pos = offset + static_cast<size_type>(k);
            if (mapped[pos] != invalid) {
                order.push_back(pos);
            }
        }
        // the ext+i fill binary-searches each row, so sort after the mapping
        std::sort(order.begin(), order.end(), [&](size_type a, size_type b) {
            return mapped[a] < mapped[b];
        });
        auto out = static_cast<size_type>(row_ptrs[r]);
        for (const auto pos : order) {
            col_idxs[out] = mapped[pos];
            values[out] = recv_vals[pos];
            out++;
        }
        offset += static_cast<size_type>(recv_nnz_counts[r]);
    }

    return csr_type::create(
        exec, dim<2>{n_halo, n_ext},
        array<ValueType>{exec, values.begin(), values.end()},
        array<IndexType>{exec, col_idxs.begin(), col_idxs.end()},
        array<IndexType>{exec, row_ptrs.begin(), row_ptrs.end()});
}


/**
 * Splices the two local blocks and the fetched halo rows into the single
 * n_ext x n_ext matrix an interpolation kernel runs on unchanged; the caller
 * keeps the first n_loc output rows.
 *
 * Row i < n_loc is `diag`'s row i followed by `off_diag`'s row i shifted by
 * n_loc, and row n_loc + r is row r of `halo_rows`, already extended. Column
 * order is inherited from the inputs, so a kernel that binary-searches rows
 * needs the result sorted first.
 *
 * @param exec  the executor the returned matrix is created on
 * @param diag  the diagonal block, n_loc x n_loc
 * @param off_diag  the off-diagonal block, n_loc x n_halo
 * @param halo_rows  fetch_halo_rows' result, n_halo x n_ext
 *
 * @return an (n_ext x n_ext) Csr on `exec`, column order inherited from the
 *         inputs and not guaranteed sorted
 */
template <typename ValueType, typename IndexType>
std::unique_ptr<matrix::Csr<ValueType, IndexType>> splice_extended_matrix(
    std::shared_ptr<const Executor> exec,
    const matrix::Csr<ValueType, IndexType>* diag,
    const matrix::Csr<ValueType, IndexType>* off_diag,
    const matrix::Csr<ValueType, IndexType>* halo_rows)
{
    using csr_type = matrix::Csr<ValueType, IndexType>;
    auto host = exec->get_master();
    const auto n_loc = diag->get_size()[0];
    const auto n_halo = off_diag->get_size()[1];
    const auto n_ext = n_loc + n_halo;
    GKO_ASSERT_EQ(halo_rows->get_size()[0], n_halo);
    GKO_ASSERT_EQ(halo_rows->get_size()[1], n_ext);

    // the splice runs on the host
    auto diag_host = make_temporary_clone(host, diag);
    auto offd_host = make_temporary_clone(host, off_diag);
    auto halo_host = make_temporary_clone(host, halo_rows);
    const auto d_ptrs = diag_host->get_const_row_ptrs();
    const auto d_cols = diag_host->get_const_col_idxs();
    const auto d_vals = diag_host->get_const_values();
    const auto o_ptrs = offd_host->get_const_row_ptrs();
    const auto o_cols = offd_host->get_const_col_idxs();
    const auto o_vals = offd_host->get_const_values();
    const auto h_ptrs = halo_host->get_const_row_ptrs();
    const auto h_cols = halo_host->get_const_col_idxs();
    const auto h_vals = halo_host->get_const_values();

    std::vector<IndexType> row_ptrs(n_ext + 1, IndexType{});
    for (size_type i = 0; i < n_loc; i++) {
        row_ptrs[i + 1] = row_ptrs[i] + (d_ptrs[i + 1] - d_ptrs[i]) +
                          (o_ptrs[i + 1] - o_ptrs[i]);
    }
    for (size_type r = 0; r < n_halo; r++) {
        row_ptrs[n_loc + r + 1] =
            row_ptrs[n_loc + r] + (h_ptrs[r + 1] - h_ptrs[r]);
    }
    const auto local_nnz = static_cast<size_type>(row_ptrs[n_loc]);
    const auto nnz = static_cast<size_type>(row_ptrs[n_ext]);
    GKO_THROW_IF_INVALID(local_nnz == diag->get_num_stored_elements() +
                                          off_diag->get_num_stored_elements(),
                         "Invalid extended matrix local nnz");
    GKO_THROW_IF_INVALID(
        nnz == local_nnz + halo_rows->get_num_stored_elements(),
        "Invalid extended matrix nnz");

    std::vector<IndexType> col_idxs(nnz);
    std::vector<ValueType> values(nnz);
    for (size_type i = 0; i < n_loc; i++) {
        auto out = static_cast<size_type>(row_ptrs[i]);
        for (auto k = d_ptrs[i]; k < d_ptrs[i + 1]; k++) {
            col_idxs[out] = d_cols[k];
            values[out] = d_vals[k];
            out++;
        }
        for (auto k = o_ptrs[i]; k < o_ptrs[i + 1]; k++) {
            col_idxs[out] = static_cast<IndexType>(n_loc) + o_cols[k];
            values[out] = o_vals[k];
            out++;
        }
    }
    const auto halo_nnz = halo_rows->get_num_stored_elements();
    std::copy(h_cols, h_cols + halo_nnz,
              col_idxs.begin() + static_cast<std::ptrdiff_t>(local_nnz));
    std::copy(h_vals, h_vals + halo_nnz,
              values.begin() + static_cast<std::ptrdiff_t>(local_nnz));

    return csr_type::create(
        exec, dim<2>{n_ext, n_ext},
        array<ValueType>{exec, values.begin(), values.end()},
        array<IndexType>{exec, col_idxs.begin(), col_idxs.end()},
        array<IndexType>{exec, row_ptrs.begin(), row_ptrs.end()});
}


}  // namespace pmis
}  // namespace multigrid
}  // namespace gko


#endif  // GINKGO_BUILD_MPI
#endif  // GKO_CORE_MULTIGRID_PMIS_HALO_HPP_
