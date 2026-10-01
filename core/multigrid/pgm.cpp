// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/multigrid/pgm.hpp"

#include <limits>
#include <utility>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/distributed/base.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/distributed/partition_helpers.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/matrix/row_gatherer.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>

#include "core/base/device_matrix_data_kernels.hpp"
#include "core/base/dispatch_helper.hpp"
#include "core/base/iterator_factory.hpp"
#include "core/base/utils.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/config/config_helper.hpp"
#include "core/distributed/index_map_kernels.hpp"
#include "core/matrix/csr_builder.hpp"
#include "core/multigrid/pgm_kernels.hpp"


namespace gko {
namespace multigrid {
namespace pgm {
namespace {


GKO_REGISTER_OPERATION(match_edge, pgm::match_edge);
GKO_REGISTER_OPERATION(count_unagg, pgm::count_unagg);
GKO_REGISTER_OPERATION(renumber, pgm::renumber);
GKO_REGISTER_OPERATION(find_strongest_neighbor, pgm::find_strongest_neighbor);
GKO_REGISTER_OPERATION(assign_to_exist_agg, pgm::assign_to_exist_agg);
GKO_REGISTER_OPERATION(sort_agg, pgm::sort_agg);
GKO_REGISTER_OPERATION(map_row, pgm::map_row);
GKO_REGISTER_OPERATION(map_col, pgm::map_col);
GKO_REGISTER_OPERATION(sort_row_major, components::sort_row_major);
GKO_REGISTER_OPERATION(sort_row_major_with_mapping,
                       pgm::sort_row_major_with_mapping);
GKO_REGISTER_OPERATION(count_unrepeated_nnz, pgm::count_unrepeated_nnz);
GKO_REGISTER_OPERATION(compute_coarse_coo, pgm::compute_coarse_coo);
GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(fill_seq_array, components::fill_seq_array);
GKO_REGISTER_OPERATION(convert_idxs_to_ptrs, components::convert_idxs_to_ptrs);
GKO_REGISTER_OPERATION(gather_index, pgm::gather_index);
GKO_REGISTER_OPERATION(prefix_sum_nonnegative,
                       components::prefix_sum_nonnegative);


}  // anonymous namespace
}  // namespace pgm
namespace index_map {
namespace {


GKO_REGISTER_OPERATION(map_to_global, index_map::map_to_global);


}
}  // namespace index_map

namespace {


template <typename IndexType>
void agg_to_restrict(std::shared_ptr<const Executor> exec, IndexType num_agg,
                     const gko::array<IndexType>& agg, IndexType* row_ptrs,
                     IndexType* col_idxs)
{
    const IndexType num = agg.get_size();
    gko::array<IndexType> row_idxs(exec, agg);
    exec->run(pgm::make_fill_seq_array(col_idxs, num));
    // sort the pair (int, agg) to (row_idxs, col_idxs)
    exec->run(pgm::make_sort_agg(num, row_idxs.get_data(), col_idxs));
    // row_idxs->row_ptrs
    exec->run(pgm::make_convert_idxs_to_ptrs(row_idxs.get_data(), num, num_agg,
                                             row_ptrs));
}


/**
 * Builds the coarse matrix of an aggregation.
 *
 * If with_mapping is set, the coarse-to-fine mapping is built alongside it
 * and returned as the second element of the pair. The mapping is a
 * SparsityCsr of dimension (coarse nnz) x (fine nnz) whose row i holds the
 * fine nonzero indices summed into the coarse nonzero i, so that applying it
 * to the fine values recomputes the coarse values. Building it costs one
 * additional index array of the fine number of nonzeros, which is why it is
 * opt-in: without it the second element of the pair is a nullptr.
 */
template <typename ValueType, typename IndexType>
std::pair<std::shared_ptr<matrix::Csr<ValueType, IndexType>>,
          std::shared_ptr<matrix::SparsityCsr<ValueType, IndexType>>>
generate_coarse(std::shared_ptr<const Executor> exec,
                const matrix::Csr<ValueType, IndexType>* fine_csr,
                IndexType num_agg, const gko::array<IndexType>& agg,
                IndexType off_diag_num_agg,
                const gko::array<IndexType>& off_diag_agg, bool with_mapping)
{
    const auto num = fine_csr->get_size()[0];
    const auto nnz = fine_csr->get_num_stored_elements();
    // the mapping stores one fine index per nonzero, so IndexType has to be
    // able to address all of them
    if (with_mapping &&
        nnz > static_cast<size_type>(std::numeric_limits<IndexType>::max())) {
        throw OverflowError(__FILE__, __LINE__, "IndexType");
    }
    gko::array<IndexType> row_idxs(exec, nnz);
    gko::array<IndexType> col_idxs(exec, nnz);
    gko::array<ValueType> vals(exec, nnz);
    gko::array<IndexType> mapping_cols(exec, with_mapping ? nnz : 0);
    exec->copy_from(exec, nnz, fine_csr->get_const_values(), vals.get_data());

    if (nnz == 0) {
        // An empty fine block gives an empty coarse block. Still return a
        // valid (empty) mapping, such that value-only updates can apply it
        // unconditionally.
        auto empty_mapping =
            with_mapping ? matrix::SparsityCsr<ValueType, IndexType>::create(
                               exec, dim<2>{0, 0}, std::move(mapping_cols),
                               gko::array<IndexType>(exec, {zero<IndexType>()}))
                         : nullptr;
        return std::make_pair(matrix::Csr<ValueType, IndexType>::create(
                                  exec, dim<2>(num_agg, off_diag_num_agg)),
                              std::move(empty_mapping));
    }

    // map row_ptrs to coarse row index
    exec->run(pgm::make_map_row(num, fine_csr->get_const_row_ptrs(),
                                agg.get_const_data(), row_idxs.get_data()));
    // map col_idxs to coarse col index
    exec->run(pgm::make_map_col(nnz, fine_csr->get_const_col_idxs(),
                                off_diag_agg.get_const_data(),
                                col_idxs.get_data()));
    // sort by row, col
    // Because reduce_by_key is not deterministic, so we do not need
    // stable_sort_by_key
    // TODO: If we have deterministic reduce_by_key, we might consider
    // stable_sort_by_key
    if (with_mapping) {
        // carry the original position of every nonzero through the sort, it
        // becomes the column index of the mapping
        exec->run(pgm::make_fill_seq_array(mapping_cols.get_data(), nnz));
        exec->run(pgm::make_sort_row_major_with_mapping(
            nnz, row_idxs.get_data(), col_idxs.get_data(), vals.get_data(),
            mapping_cols.get_data()));
    } else {
        exec->run(pgm::make_sort_row_major(
            nnz, row_idxs.get_data(), col_idxs.get_data(), vals.get_data()));
    }
    // compute the total nnz and create the fine csr
    size_type coarse_nnz = 0;
    exec->run(pgm::make_count_unrepeated_nnz(nnz, row_idxs.get_const_data(),
                                             col_idxs.get_const_data(),
                                             &coarse_nnz));
    // the number of fine nonzeros reduced into each coarse nonzero, which
    // becomes the row pointers of the mapping after the prefix sum
    gko::array<IndexType> mapping_rows(exec, with_mapping ? coarse_nnz + 1 : 0);
    // reduce by key (row, col)
    auto coarse_coo = matrix::Coo<ValueType, IndexType>::create(
        exec,
        gko::dim<2>{static_cast<size_type>(num_agg),
                    static_cast<size_type>(off_diag_num_agg)},
        coarse_nnz);
    exec->run(pgm::make_compute_coarse_coo(
        nnz, row_idxs.get_const_data(), col_idxs.get_const_data(),
        vals.get_const_data(), coarse_coo->get_device_view(),
        with_mapping ? mapping_rows.get_data() : nullptr));
    // use move_to
    auto coarse_csr = matrix::Csr<ValueType, IndexType>::create(exec);
    coarse_csr->move_from(coarse_coo);
    if (!with_mapping) {
        return std::make_pair(std::move(coarse_csr), nullptr);
    }
    exec->run(pgm::make_prefix_sum_nonnegative(mapping_rows.get_data(),
                                               coarse_nnz + 1));
    auto mapping_csr = matrix::SparsityCsr<ValueType, IndexType>::create(
        exec, dim<2>{coarse_nnz, nnz}, std::move(mapping_cols),
        std::move(mapping_rows));
    return std::make_pair(std::move(coarse_csr), std::move(mapping_csr));
}


template <typename ValueType, typename IndexType>
std::pair<std::shared_ptr<matrix::Csr<ValueType, IndexType>>,
          std::shared_ptr<matrix::SparsityCsr<ValueType, IndexType>>>
generate_coarse(std::shared_ptr<const Executor> exec,
                const matrix::Csr<ValueType, IndexType>* fine_csr,
                IndexType num_agg, const gko::array<IndexType>& agg,
                bool with_mapping)
{
    return generate_coarse(exec, fine_csr, num_agg, agg, num_agg, agg,
                           with_mapping);
}


/**
 * Recomputes the values of a coarse matrix from the fine values, reusing the
 * coarse-to-fine mapping built during the generation.
 *
 * The sparsity of both matrices and the mapping between them are unchanged, so
 * only the coarse values are overwritten. The coarse matrix is only reachable
 * as const through the multigrid level, but updating its values in place is
 * exactly what this is for, hence the const_cast.
 */
template <typename ValueType, typename IndexType>
void update_coarse_values(
    std::shared_ptr<const Executor> exec,
    const matrix::SparsityCsr<ValueType, IndexType>* mapping,
    const matrix::Csr<ValueType, IndexType>* fine,
    const matrix::Csr<ValueType, IndexType>* coarse)
{
    auto fine_vals = matrix::Dense<ValueType>::create_const(
        exec, dim<2>{fine->get_num_stored_elements(), 1},
        make_const_array_view(exec, fine->get_num_stored_elements(),
                              fine->get_const_values()),
        1);
    auto mutable_coarse =
        const_cast<matrix::Csr<ValueType, IndexType>*>(coarse);
    auto coarse_vals = matrix::Dense<ValueType>::create(
        exec, dim<2>{mutable_coarse->get_num_stored_elements(), 1},
        make_array_view(exec, mutable_coarse->get_num_stored_elements(),
                        mutable_coarse->get_values()),
        1);
    mapping->apply(fine_vals, coarse_vals);
}


/**
 * Checks that a fine block still matches the mapping built for it.
 *
 * The mapping has one column per fine nonzero, so comparing the number of
 * columns against the number of nonzeros of the new block catches every
 * change in the number of nonzeros. A different pattern with the same number
 * of nonzeros is not detectable without comparing the patterns themselves,
 * which costs as much as the update, so it stays a precondition of
 * update_matrix_value().
 */
template <typename ValueType, typename IndexType>
size_type mapping_nnz(const matrix::SparsityCsr<ValueType, IndexType>* mapping)
{
    return mapping ? mapping->get_size()[1] : size_type{};
}


template <typename ValueType, typename IndexType>
bool mapping_matches(const matrix::SparsityCsr<ValueType, IndexType>* mapping,
                     const matrix::Csr<ValueType, IndexType>* fine)
{
    return mapping != nullptr &&
           mapping->get_size()[1] == fine->get_num_stored_elements();
}


#if GINKGO_BUILD_MPI


/** The distributed matrix types the fine operator can take. */
template <typename ValueType, typename IndexType>
using fst_mtx_type =
    experimental::distributed::Matrix<ValueType, IndexType, IndexType>;
template <typename ValueType, typename IndexType>
using snd_mtx_type =
    experimental::distributed::Matrix<ValueType, IndexType, int64>;


#endif  // GINKGO_BUILD_MPI


}  // namespace


template <typename ValueType, typename IndexType>
typename Pgm<ValueType, IndexType>::parameters_type
Pgm<ValueType, IndexType>::parse(const config::pnode& config,
                                 const config::registry& context,
                                 const config::type_descriptor& td_for_child)
{
    auto params = Pgm<ValueType, IndexType>::build();
    config::config_check_decorator config_check(config);
    if (auto& obj = config_check.get("max_iterations")) {
        params.with_max_iterations(config::get_value<unsigned>(obj));
    }
    if (auto& obj = config_check.get("max_unassigned_ratio")) {
        params.with_max_unassigned_ratio(config::get_value<double>(obj));
    }
    if (auto& obj = config_check.get("deterministic")) {
        params.with_deterministic(config::get_value<bool>(obj));
    }
    if (auto& obj = config_check.get("skip_sorting")) {
        params.with_skip_sorting(config::get_value<bool>(obj));
    }
    if (auto& obj = config_check.get("updatable_values")) {
        params.with_updatable_values(config::get_value<bool>(obj));
    }

    return params;
}


template <typename ValueType, typename IndexType>
std::tuple<std::shared_ptr<LinOp>, std::shared_ptr<LinOp>,
           std::shared_ptr<LinOp>>
Pgm<ValueType, IndexType>::generate_local(
    std::shared_ptr<const matrix::Csr<ValueType, IndexType>> local_matrix)
{
    using csr_type = matrix::Csr<ValueType, IndexType>;
    using real_type = remove_complex<ValueType>;
    using weight_csr_type = remove_complex<csr_type>;
    agg_.resize_and_reset(local_matrix->get_size()[0]);
    auto exec = this->get_executor();
    const auto num_rows = local_matrix->get_size()[0];
    array<IndexType> strongest_neighbor(this->get_executor(), num_rows);
    array<IndexType> intermediate_agg(this->get_executor(),
                                      parameters_.deterministic * num_rows);

    // Initial agg = -1
    exec->run(pgm::make_fill_array(agg_.get_data(), agg_.get_size(),
                                   -one<IndexType>()));
    IndexType num_unagg = num_rows;
    IndexType num_unagg_prev = num_rows;
    // TODO: if mtx is a hermitian matrix, weight_mtx = abs(mtx)
    // compute weight_mtx = (abs(mtx) + abs(mtx'))/2;
    auto abs_mtx = local_matrix->compute_absolute();
    // abs_mtx is already real valuetype, so transpose is enough
    auto weight_mtx = gko::as<weight_csr_type>(abs_mtx->transpose());
    auto half_scalar = initialize<matrix::Dense<real_type>>({0.5}, exec);
    auto identity = matrix::Identity<real_type>::create(exec, num_rows);
    // W = (abs_mtx + transpose(abs_mtx))/2
    abs_mtx->apply(half_scalar, identity, half_scalar, weight_mtx);
    // Extract the diagonal value of matrix
    auto diag = weight_mtx->extract_diagonal();
    for (int i = 0; i < parameters_.max_iterations; i++) {
        // Find the strongest neighbor of each row
        exec->run(pgm::make_find_strongest_neighbor(
            weight_mtx->get_const_device_view(), diag.get(), agg_,
            strongest_neighbor));
        // Match edges
        exec->run(pgm::make_match_edge(strongest_neighbor, agg_));
        // Get the num_unagg
        exec->run(pgm::make_count_unagg(agg_, &num_unagg));
        // no new match, all match, or the ratio of num_unagg/num is lower
        // than parameter.max_unassigned_ratio
        if (num_unagg == 0 || num_unagg == num_unagg_prev ||
            num_unagg < parameters_.max_unassigned_ratio * num_rows) {
            break;
        }
        num_unagg_prev = num_unagg;
    }
    // Handle the left unassign points
    if (num_unagg != 0 && parameters_.deterministic) {
        // copy the agg to intermediate_agg
        intermediate_agg = agg_;
    }
    if (num_unagg != 0) {
        // Assign all left points
        exec->run(
            pgm::make_assign_to_exist_agg(weight_mtx->get_const_device_view(),
                                          diag.get(), agg_, intermediate_agg));
    }
    IndexType num_agg = 0;
    // Renumber the index
    exec->run(pgm::make_renumber(agg_, &num_agg));
    gko::dim<2>::dimension_type coarse_dim = num_agg;
    auto fine_dim = local_matrix->get_size()[0];
    // prolong_row_gather is the lightway implementation for prolongation
    auto prolong_row_gather = share(matrix::RowGatherer<IndexType>::create(
        exec, gko::dim<2>{fine_dim, coarse_dim}));
    exec->copy_from(exec, agg_.get_size(), agg_.get_const_data(),
                    prolong_row_gather->get_row_idxs());
    auto restrict_sparsity =
        share(matrix::SparsityCsr<ValueType, IndexType>::create(
            exec, gko::dim<2>{coarse_dim, fine_dim}, fine_dim));
    agg_to_restrict(exec, num_agg, agg_, restrict_sparsity->get_row_ptrs(),
                    restrict_sparsity->get_col_idxs());

    // Construct the coarse matrix
    // TODO: improve it
    auto [coarse_matrix, mapping_matrix] = generate_coarse(
        exec, local_matrix.get(), num_agg, agg_, parameters_.updatable_values);
    mapping_local_ = mapping_matrix;
    return std::tie(prolong_row_gather, coarse_matrix, restrict_sparsity);
}


#if GINKGO_BUILD_MPI


template <typename ValueType, typename IndexType>
template <typename GlobalIndexType>
array<GlobalIndexType> Pgm<ValueType, IndexType>::communicate_off_diag_agg(
    std::shared_ptr<const experimental::distributed::Matrix<
        ValueType, IndexType, GlobalIndexType>>
        matrix,
    std::shared_ptr<
        experimental::distributed::Partition<IndexType, GlobalIndexType>>
        coarse_partition,
    const array<IndexType>& local_agg)
{
    auto exec = matrix->get_executor();
    const auto comm = matrix->get_communicator();
    auto coll_comm = matrix->row_gatherer_->get_collective_communicator();
    auto total_send_size = coll_comm->get_send_size();
    auto total_recv_size = coll_comm->get_recv_size();
    auto row_gatherer = matrix->row_gatherer_;

    array<IndexType> send_agg(exec, total_send_size);
    exec->run(pgm::make_gather_index(
        send_agg.get_size(), local_agg.get_const_data(),
        row_gatherer->get_const_send_idxs(), send_agg.get_data()));

    // There is no index map on the coarse level yet, so map the local indices
    // to global indices on the coarse level manually
    array<GlobalIndexType> send_global_agg(exec, send_agg.get_size());
    exec->run(index_map::make_map_to_global(
        to_device_const(coarse_partition.get()),
        device_segmented_array<const GlobalIndexType>{}, comm.rank(), send_agg,
        experimental::distributed::index_space::local, send_global_agg));

    array<GlobalIndexType> off_diag_agg(exec, total_recv_size);

    auto use_host_buffer = experimental::mpi::requires_host_buffer(exec, comm);
    array<GlobalIndexType> host_recv_buffer(exec->get_master());
    array<GlobalIndexType> host_send_buffer(exec->get_master());
    if (use_host_buffer) {
        host_recv_buffer.resize_and_reset(total_recv_size);
        host_send_buffer.resize_and_reset(total_send_size);
        exec->get_master()->copy_from(exec, total_send_size,
                                      send_global_agg.get_data(),
                                      host_send_buffer.get_data());
    }

    const auto send_ptr = use_host_buffer ? host_send_buffer.get_const_data()
                                          : send_global_agg.get_const_data();
    auto recv_ptr =
        use_host_buffer ? host_recv_buffer.get_data() : off_diag_agg.get_data();
    exec->synchronize();
    coll_comm
        ->i_all_to_all_v(use_host_buffer ? exec->get_master() : exec, send_ptr,
                         recv_ptr)
        .wait();
    if (use_host_buffer) {
        exec->copy_from(exec->get_master(), total_recv_size, recv_ptr,
                        off_diag_agg.get_data());
    }
    return off_diag_agg;
}


#endif


#if GINKGO_BUILD_MPI


template <typename ValueType, typename IndexType>
std::shared_ptr<const LinOp>
Pgm<ValueType, IndexType>::convert_distributed_fine_op(
    std::shared_ptr<const LinOp> system_matrix) const
{
    using csr_type = matrix::Csr<ValueType, IndexType>;
    using fst_mtx = fst_mtx_type<ValueType, IndexType>;
    using snd_mtx = snd_mtx_type<ValueType, IndexType>;
    std::shared_ptr<const LinOp> fine_op;
    auto convert_fine_op = [&](auto matrix) {
        using global_index_type = typename std::decay_t<
            decltype(*matrix)>::result_type::global_index_type;
        auto exec = this->get_executor();
        auto comm = as<experimental::distributed::DistributedBase>(matrix)
                        ->get_communicator();
        auto fine =
            share(experimental::distributed::
                      Matrix<ValueType, IndexType, global_index_type>::create(
                          exec, comm, csr_type::create(exec),
                          csr_type::create(exec)));
        matrix->convert_to(fine);
        fine_op = fine;
    };
    auto setup_fine_op = [&](auto matrix) {
        // Only support csr matrix currently.
        auto diag_csr = std::dynamic_pointer_cast<const csr_type>(
            matrix->get_diag_matrix());
        auto off_diag_csr = std::dynamic_pointer_cast<const csr_type>(
            matrix->get_off_diag_matrix());
        // If system matrix is not csr, lives on another executor or needs
        // sorting, generate the csr.
        if (!parameters_.skip_sorting || !diag_csr || !off_diag_csr ||
            as<LinOp>(matrix)->get_executor() != this->get_executor()) {
            using global_index_type =
                typename std::decay_t<decltype(*matrix)>::global_index_type;
            convert_fine_op(
                as<ConvertibleTo<experimental::distributed::Matrix<
                    ValueType, IndexType, global_index_type>>>(matrix));
        } else {
            // No conversion is required, so the system matrix is directly
            // usable as the fine op.
            fine_op = matrix;
        }
    };

    // setup the fine op using Csr with current ValueType
    // we do not use dispatcher run in the first place because we have the
    // fallback option for that.
    if (auto obj = std::dynamic_pointer_cast<const fst_mtx>(system_matrix)) {
        setup_fine_op(obj);
    } else if (auto obj =
                   std::dynamic_pointer_cast<const snd_mtx>(system_matrix)) {
        setup_fine_op(obj);
    } else {
        // handle other ValueTypes.
        run<ConvertibleTo, fst_mtx, snd_mtx>(system_matrix, convert_fine_op);
    }
    return fine_op;
}


#endif  // GINKGO_BUILD_MPI


template <typename ValueType, typename IndexType>
std::shared_ptr<const matrix::Csr<ValueType, IndexType>>
Pgm<ValueType, IndexType>::convert_local_fine_op(
    std::shared_ptr<const LinOp> system_matrix) const
{
    using csr_type = matrix::Csr<ValueType, IndexType>;
    // Only support csr matrix currently.
    auto pgm_op = std::dynamic_pointer_cast<const csr_type>(system_matrix);
    // If system matrix is not csr, lives on another executor or needs
    // sorting, generate the csr.
    // TODO: UniformCoarsening::generate does the same conversion, the two
    // should be deduplicated.
    if (!parameters_.skip_sorting || !pgm_op ||
        pgm_op->get_executor() != this->get_executor()) {
        pgm_op = convert_to_with_sorting<csr_type>(
            this->get_executor(), system_matrix, parameters_.skip_sorting);
    }
    return pgm_op;
}


template <typename ValueType, typename IndexType>
void Pgm<ValueType, IndexType>::generate()
{
    using csr_type = matrix::Csr<ValueType, IndexType>;
#if GINKGO_BUILD_MPI
    if (std::dynamic_pointer_cast<
            const experimental::distributed::DistributedBase>(system_matrix_)) {
        this->set_fine_op(this->convert_distributed_fine_op(system_matrix_));

        auto distributed_setup = [&](auto matrix) {
            using global_index_type =
                typename std::decay_t<decltype(*matrix)>::global_index_type;

            auto exec = gko::as<LinOp>(matrix)->get_executor();
            auto comm =
                gko::as<experimental::distributed::DistributedBase>(matrix)
                    ->get_communicator();
            auto pgm_local_op =
                gko::as<const csr_type>(matrix->get_diag_matrix());
            auto result = this->generate_local(pgm_local_op);

            // create the coarse partition
            // the coarse partition will have only one range per part
            // and only one part per rank.
            // The global indices are ordered block-wise by rank, i.e. rank
            // 0 owns [0, ..., N_1), rank 1 [N_1, ..., N_2), ...
            auto coarse_local_size =
                static_cast<int64>(std::get<1>(result)->get_size()[0]);
            auto coarse_partition = gko::share(
                experimental::distributed::build_partition_from_local_size<
                    IndexType, global_index_type>(exec, comm,
                                                  coarse_local_size));

            // get the off-diag aggregates as coarse global indices
            auto off_diag_agg =
                communicate_off_diag_agg(matrix, coarse_partition, agg_);

            // create a coarse index map based on the connection given by
            // the off-diag aggregates
            auto coarse_imap =
                experimental::distributed::index_map<IndexType,
                                                     global_index_type>(
                    exec, coarse_partition, comm.rank(), off_diag_agg);

            // a mapping from the fine off-diag indices to the coarse
            // off-diag indices.
            // off_diag_agg already maps the fine off-diag indices to
            // coarse global indices, so mapping it with the coarse index
            // map results in the coarse off-diag indices.
            auto off_diag_num_agg =
                static_cast<IndexType>(coarse_imap.get_non_local_size());
            auto off_diag_map = coarse_imap.map_to_local(
                off_diag_agg,
                experimental::distributed::index_space::non_local);

            // build csr from row and col map
            // unlike non-distributed version, generate_coarse uses
            // different row and col maps.
            auto off_diag_csr =
                as<const csr_type>(matrix->get_off_diag_matrix());
            auto [result_off_diag_csr, mapping_off_diag] = generate_coarse(
                exec, off_diag_csr.get(),
                static_cast<IndexType>(std::get<1>(result)->get_size()[0]),
                agg_, off_diag_num_agg, off_diag_map,
                parameters_.updatable_values);
            mapping_off_diag_ = mapping_off_diag;

            // setup the generated linop.
            auto coarse = share(
                experimental::distributed::
                    Matrix<ValueType, IndexType, global_index_type>::create(
                        exec, comm, std::move(coarse_imap), std::get<1>(result),
                        result_off_diag_csr));
            auto restrict_op = share(
                experimental::distributed::
                    Matrix<ValueType, IndexType, global_index_type>::create(
                        exec, comm,
                        dim<2>(coarse->get_size()[0],
                               gko::as<LinOp>(matrix)->get_size()[0]),
                        std::get<2>(result)));
            auto prolong_op = share(
                experimental::distributed::
                    Matrix<ValueType, IndexType, global_index_type>::create(
                        exec, comm,
                        dim<2>(gko::as<LinOp>(matrix)->get_size()[0],
                               coarse->get_size()[0]),
                        std::get<0>(result)));
            this->set_multigrid_level(prolong_op, coarse, restrict_op);
        };

        // the fine op is using csr with the current ValueType
        run<fst_mtx_type<ValueType, IndexType>,
            snd_mtx_type<ValueType, IndexType>>(this->get_fine_op(),
                                                distributed_setup);
    } else
#endif  // GINKGO_BUILD_MPI
    {
        auto pgm_op = this->convert_local_fine_op(system_matrix_);
        // keep the same precision data in fine_op
        this->set_fine_op(pgm_op);
        auto result = this->generate_local(pgm_op);
        this->set_multigrid_level(std::get<0>(result), std::get<1>(result),
                                  std::get<2>(result));
    }
}


template <typename ValueType, typename IndexType>
void Pgm<ValueType, IndexType>::update_matrix_value(
    std::shared_ptr<const LinOp> new_matrix)
{
    using csr_type = matrix::Csr<ValueType, IndexType>;
    auto exec = this->get_executor();
    // The aggregates and the coarse-to-fine mapping are reused as they are,
    // so a level generated without the mapping can not be updated. Reject it
    // before anything is touched.
    if (!parameters_.updatable_values) {
        throw NotSupported(__FILE__, __LINE__, __func__,
                           "update_matrix_value on a Pgm generated without "
                           "the updatable_values parameter");
    }
    GKO_ASSERT_EQUAL_DIMENSIONS(this, new_matrix);
    // generate() gets the system matrix through the factory, which puts it on
    // the factory executor, so do the same here.
    auto matrix = new_matrix->get_executor() == exec
                      ? new_matrix
                      : gko::clone(exec, new_matrix);
#if GINKGO_BUILD_MPI
    if (std::dynamic_pointer_cast<
            const experimental::distributed::DistributedBase>(matrix)) {
        // convert first, so that a rejected update leaves the level alone
        auto fine_op = this->convert_distributed_fine_op(matrix);

        auto distributed_setup = [&](auto fine) {
            using global_index_type =
                typename std::decay_t<decltype(*fine)>::global_index_type;

            auto diag_csr = gko::as<const csr_type>(fine->get_diag_matrix());
            auto off_diag_csr =
                gko::as<const csr_type>(fine->get_off_diag_matrix());
            // All ranks have to agree on whether the update can go ahead.
            // Throwing on some of them only would leave the others hanging in
            // the next collective call.
            int local_matches = static_cast<int>(
                mapping_matches(mapping_local_.get(), diag_csr.get()) &&
                mapping_matches(mapping_off_diag_.get(), off_diag_csr.get()));
            int matches = local_matches;
            fine->get_communicator().all_reduce(
                exec->get_master(), &local_matches, &matches, 1, MPI_MIN);
            if (!matches) {
                throw ValueMismatch(
                    __FILE__, __LINE__, __func__,
                    diag_csr->get_num_stored_elements() +
                        off_diag_csr->get_num_stored_elements(),
                    mapping_nnz(mapping_local_.get()) +
                        mapping_nnz(mapping_off_diag_.get()),
                    "at least one rank got a matrix with a different number "
                    "of nonzeros than the one this level was generated with");
            }

            system_matrix_ = matrix;
            this->set_fine_op(fine);
            auto coarse =
                as<experimental::distributed::Matrix<ValueType, IndexType,
                                                     global_index_type>>(
                    this->get_coarse_op());
            update_coarse_values(
                exec, mapping_local_.get(), diag_csr.get(),
                gko::as<csr_type>(coarse->get_diag_matrix()).get());
            update_coarse_values(
                exec, mapping_off_diag_.get(), off_diag_csr.get(),
                gko::as<csr_type>(coarse->get_off_diag_matrix()).get());
        };
        // the fine op is using csr with the current ValueType
        run<fst_mtx_type<ValueType, IndexType>,
            snd_mtx_type<ValueType, IndexType>>(fine_op, distributed_setup);
    } else
#endif
    {
        // convert first, so that a rejected update leaves the level alone
        auto pgm_op = this->convert_local_fine_op(matrix);
        if (!mapping_matches(mapping_local_.get(), pgm_op.get())) {
            throw ValueMismatch(__FILE__, __LINE__, __func__,
                                pgm_op->get_num_stored_elements(),
                                mapping_nnz(mapping_local_.get()),
                                "the new matrix does not have the same number "
                                "of nonzeros as the one this level was "
                                "generated with");
        }
        system_matrix_ = matrix;
        this->set_fine_op(pgm_op);
        update_coarse_values(exec, mapping_local_.get(), pgm_op.get(),
                             gko::as<csr_type>(this->get_coarse_op()).get());
    }
}


#define GKO_DECLARE_PGM(ValueType, IndexType) class Pgm<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_PGM);


}  // namespace multigrid
}  // namespace gko
