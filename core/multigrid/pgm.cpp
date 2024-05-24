// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/multigrid/pgm.hpp"

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

#include "core/base/dispatch_helper.hpp"
#include "core/base/iterator_factory.hpp"
#include "core/base/utils.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/config/config_helper.hpp"
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
GKO_REGISTER_OPERATION(sort_row_major, pgm::sort_row_major);
GKO_REGISTER_OPERATION(count_unrepeated_nnz, pgm::count_unrepeated_nnz);
GKO_REGISTER_OPERATION(compute_coarse_coo, pgm::compute_coarse_coo);
GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(fill_seq_array, components::fill_seq_array);
GKO_REGISTER_OPERATION(convert_idxs_to_ptrs, components::convert_idxs_to_ptrs);
GKO_REGISTER_OPERATION(gather_index, pgm::gather_index);


}  // anonymous namespace
}  // namespace pgm

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


template <typename ValueType, typename IndexType>
std::shared_ptr<matrix::Csr<ValueType, IndexType>> generate_coarse(
    std::shared_ptr<const Executor> exec,
    const matrix::Csr<ValueType, IndexType>* fine_csr, IndexType num_agg,
    const gko::array<IndexType>& agg, IndexType non_local_num_agg,
    const gko::array<IndexType>& non_local_agg)
{
    const auto num = fine_csr->get_size()[0];
    const auto nnz = fine_csr->get_num_stored_elements();
    gko::array<IndexType> row_idxs(exec, nnz);
    gko::array<IndexType> col_idxs(exec, nnz);
    gko::array<ValueType> vals(exec, nnz);
    exec->copy_from(exec, nnz, fine_csr->get_const_values(), vals.get_data());

    if (nnz == 0) {
        return matrix::Csr<ValueType, IndexType>::create(
            exec, dim<2>(num_agg, non_local_num_agg));
    }

    // map row_ptrs to coarse row index
    exec->run(pgm::make_map_row(num, fine_csr->get_const_row_ptrs(),
                                agg.get_const_data(), row_idxs.get_data()));
    // map col_idxs to coarse col index
    exec->run(pgm::make_map_col(nnz, fine_csr->get_const_col_idxs(),
                                non_local_agg.get_const_data(),
                                col_idxs.get_data()));
    // sort by row, col
    exec->run(pgm::make_sort_row_major(nnz, row_idxs.get_data(),
                                       col_idxs.get_data(), vals.get_data()));
    // compute the total nnz and create the fine csr
    size_type coarse_nnz = 0;
    exec->run(pgm::make_count_unrepeated_nnz(nnz, row_idxs.get_const_data(),
                                             col_idxs.get_const_data(),
                                             &coarse_nnz));
    // reduce by key (row, col)
    auto coarse_coo = matrix::Coo<ValueType, IndexType>::create(
        exec,
        gko::dim<2>{static_cast<size_type>(num_agg),
                    static_cast<size_type>(non_local_num_agg)},
        coarse_nnz);
    exec->run(pgm::make_compute_coarse_coo(
        nnz, row_idxs.get_const_data(), col_idxs.get_const_data(),
        vals.get_const_data(), coarse_coo.get()));
    // use move_to
    auto coarse_csr = matrix::Csr<ValueType, IndexType>::create(exec);
    coarse_csr->move_from(coarse_coo);
    return std::move(coarse_csr);
}


template <typename ValueType, typename IndexType>
std::shared_ptr<matrix::Csr<ValueType, IndexType>> generate_coarse(
    std::shared_ptr<const Executor> exec,
    const matrix::Csr<ValueType, IndexType>* fine_csr, IndexType num_agg,
    const gko::array<IndexType>& agg)
{
    return generate_coarse(exec, fine_csr, num_agg, agg, num_agg, agg);
}


}  // namespace


template <typename ValueType, typename IndexType>
typename Pgm<ValueType, IndexType>::parameters_type
Pgm<ValueType, IndexType>::parse(const config::pnode& config,
                                 const config::registry& context,
                                 const config::type_descriptor& td_for_child)
{
    auto params = Pgm<ValueType, IndexType>::build();
    if (auto& obj = config.get("max_iterations")) {
        params.with_max_iterations(gko::config::get_value<unsigned>(obj));
    }
    if (auto& obj = config.get("max_unassigned_ratio")) {
        params.with_max_unassigned_ratio(gko::config::get_value<double>(obj));
    }
    if (auto& obj = config.get("deterministic")) {
        params.with_deterministic(gko::config::get_value<bool>(obj));
    }
    if (auto& obj = config.get("skip_sorting")) {
        params.with_skip_sorting(gko::config::get_value<bool>(obj));
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
            weight_mtx.get(), diag.get(), agg_, strongest_neighbor));
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
        exec->run(pgm::make_assign_to_exist_agg(weight_mtx.get(), diag.get(),
                                                agg_, intermediate_agg));
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
    auto coarse_matrix =
        generate_coarse(exec, local_matrix.get(), num_agg, agg_);

    return std::tie(prolong_row_gather, coarse_matrix, restrict_sparsity);
}


#if GINKGO_BUILD_MPI


template <typename ValueType, typename IndexType>
template <typename GlobalIndexType>
array<GlobalIndexType> Pgm<ValueType, IndexType>::communicate_non_local_agg(
    std::shared_ptr<const experimental::distributed::Matrix<
        ValueType, IndexType, GlobalIndexType>>
        matrix,
    std::shared_ptr<
        experimental::distributed::Partition<IndexType, GlobalIndexType>>
        coarse_partition,
    const array<IndexType>& local_agg)
{
    auto exec = gko::as<LinOp>(matrix)->get_executor();
    const auto comm = matrix->get_communicator();
    auto send_sizes = matrix->send_sizes_;
    auto recv_sizes = matrix->recv_sizes_;
    auto send_offsets = matrix->send_offsets_;
    auto recv_offsets = matrix->recv_offsets_;
    auto gather_idxs = matrix->gather_idxs_;
    auto total_send_size = send_offsets.back();
    auto total_recv_size = recv_offsets.back();

    array<IndexType> send_agg(exec, total_send_size);
    exec->run(pgm::make_gather_index(
        send_agg.get_size(), local_agg.get_const_data(),
        gather_idxs.get_const_data(), send_agg.get_data()));

    // temporary index map that contains no remote connections to map
    // local indices to global
    experimental::distributed::index_map<IndexType, GlobalIndexType> imap(
        exec, coarse_partition, comm.rank(), array<GlobalIndexType>{exec});
    auto seng_global_agg = imap.map_to_global(
        send_agg, experimental::distributed::index_space::local);

    array<GlobalIndexType> non_local_agg(exec, total_recv_size);

    auto use_host_buffer = experimental::mpi::requires_host_buffer(exec, comm);
    array<GlobalIndexType> host_recv_buffer(exec->get_master());
    array<GlobalIndexType> host_send_buffer(exec->get_master());
    if (use_host_buffer) {
        host_recv_buffer.resize_and_reset(total_recv_size);
        host_send_buffer.resize_and_reset(total_send_size);
        exec->get_master()->copy_from(exec, total_send_size,
                                      seng_global_agg.get_data(),
                                      host_send_buffer.get_data());
    }
    auto type = experimental::mpi::type_impl<GlobalIndexType>::get_type();

    const auto send_ptr = use_host_buffer ? host_send_buffer.get_const_data()
                                          : seng_global_agg.get_const_data();
    auto recv_ptr = use_host_buffer ? host_recv_buffer.get_data()
                                    : non_local_agg.get_data();
    exec->synchronize();
    comm.all_to_all_v(use_host_buffer ? exec->get_master() : exec, send_ptr,
                      send_sizes.data(), send_offsets.data(), type, recv_ptr,
                      recv_sizes.data(), recv_offsets.data(), type);
    if (use_host_buffer) {
        exec->copy_from(exec->get_master(), total_recv_size, recv_ptr,
                        non_local_agg.get_data());
    }
    return non_local_agg;
}


#endif


template <typename ValueType, typename IndexType>
void Pgm<ValueType, IndexType>::generate()
{
    using csr_type = matrix::Csr<ValueType, IndexType>;
#if GINKGO_BUILD_MPI
    if (std::dynamic_pointer_cast<
            const experimental::distributed::DistributedBase>(system_matrix_)) {
        if constexpr (std::is_same_v<remove_complex<ValueType>, half>) {
            GKO_NOT_SUPPORTED(nullptr);
        } else {
            auto convert_fine_op = [&](auto matrix) {
                using global_index_type = typename std::decay_t<
                    decltype(*matrix)>::result_type::global_index_type;
                auto exec = as<LinOp>(matrix)->get_executor();
                auto comm =
                    as<experimental::distributed::DistributedBase>(matrix)
                        ->get_communicator();
                auto fine = share(
                    experimental::distributed::
                        Matrix<ValueType, IndexType, global_index_type>::create(
                            exec, comm,
                            matrix::Csr<ValueType, IndexType>::create(exec),
                            matrix::Csr<ValueType, IndexType>::create(exec)));
                matrix->convert_to(fine);
                this->set_fine_op(fine);
            };
            auto setup_fine_op = [&](auto matrix) {
                // Only support csr matrix currently.
                auto local_csr = std::dynamic_pointer_cast<const csr_type>(
                    matrix->get_local_matrix());
                auto non_local_csr = std::dynamic_pointer_cast<const csr_type>(
                    matrix->get_non_local_matrix());
                // If system matrix is not csr or need sorting, generate the
                // csr.
                if (!parameters_.skip_sorting || !local_csr || !non_local_csr) {
                    using global_index_type = typename std::decay_t<
                        decltype(*matrix)>::global_index_type;
                    convert_fine_op(
                        as<ConvertibleTo<experimental::distributed::Matrix<
                            ValueType, IndexType, global_index_type>>>(matrix));
                }
            };

            using fst_mtx_type =
                experimental::distributed::Matrix<ValueType, IndexType,
                                                  IndexType>;
            using snd_mtx_type =
                experimental::distributed::Matrix<ValueType, IndexType, int64>;
            // setup the fine op using Csr with current ValueType
            // we do not use dispatcher run in the first place because we have
            // the fallback option for that.
            if (auto obj = std::dynamic_pointer_cast<const fst_mtx_type>(
                    system_matrix_)) {
                setup_fine_op(obj);
            } else if (auto obj = std::dynamic_pointer_cast<const snd_mtx_type>(
                           system_matrix_)) {
                setup_fine_op(obj);
            } else {
                // handle other ValueTypes.
                run<ConvertibleTo, fst_mtx_type, snd_mtx_type>(system_matrix_,
                                                               convert_fine_op);
            }

            auto distributed_setup = [&](auto matrix) {
                using global_index_type =
                    typename std::decay_t<decltype(*matrix)>::global_index_type;

                auto exec = gko::as<LinOp>(matrix)->get_executor();
                auto comm =
                    gko::as<experimental::distributed::DistributedBase>(matrix)
                        ->get_communicator();
                auto pgm_local_op =
                    gko::as<const csr_type>(matrix->get_local_matrix());
                auto result = this->generate_local(pgm_local_op);

                // create the coarse partition
                // the coarse partition will have only one range per part
                // and only one part per rank.
                // The global indices are ordered block-wise by rank, i.e. rank
                // 1 owns [0, ..., N_1), rank 2 [N_1, ..., N_2), ...
                auto coarse_local_size =
                    static_cast<int64>(std::get<1>(result)->get_size()[0]);
                auto coarse_partition = gko::share(
                    experimental::distributed::build_partition_from_local_size<
                        IndexType, global_index_type>(exec, comm,
                                                      coarse_local_size));

                // get the non-local aggregates as coarse global indices
                auto non_local_agg =
                    communicate_non_local_agg(matrix, coarse_partition, agg_);

                // create a coarse index map based on the connection given by
                // the non-local aggregates
                auto coarse_imap =
                    experimental::distributed::index_map<IndexType,
                                                         global_index_type>(
                        exec, coarse_partition, comm.rank(), non_local_agg);

                // a mapping from the fine non-local indices to the coarse
                // non-local
                // indices.
                // non_local_agg already maps the fine non-local indices to
                // coarse global indices, so mapping it with the coarse index
                // map results in the coarse non-local indices.
                auto non_local_map = coarse_imap.map_to_local(
                    non_local_agg,
                    experimental::distributed::index_space::non_local);

                // build csr from row and col map
                // unlike non-distributed version, generate_coarse uses
                // differentrow and col maps.
                auto non_local_csr =
                    as<const csr_type>(matrix->get_non_local_matrix());
                auto result_non_local_csr = generate_coarse(
                    exec, non_local_csr.get(),
                    static_cast<IndexType>(std::get<1>(result)->get_size()[0]),
                    agg_,
                    static_cast<IndexType>(coarse_imap.get_non_local_size()),
                    non_local_map);

                // setup the generated linop.
                auto coarse = share(
                    experimental::distributed::
                        Matrix<ValueType, IndexType, global_index_type>::create(
                            exec, comm, std::move(coarse_imap),
                            std::get<1>(result), result_non_local_csr));
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
            run<fst_mtx_type, snd_mtx_type>(this->get_fine_op(),
                                            distributed_setup);
        }
    } else
#endif  // GINKGO_BUILD_MPI
    {
        auto exec = this->get_executor();
        // Only support csr matrix currently.
        auto pgm_op = std::dynamic_pointer_cast<const csr_type>(system_matrix_);
        // If system matrix is not csr or need sorting, generate the csr.
        if (!parameters_.skip_sorting || !pgm_op) {
            pgm_op = convert_to_with_sorting<csr_type>(
                exec, system_matrix_, parameters_.skip_sorting);
            // keep the same precision data in fine_op
            this->set_fine_op(pgm_op);
        }
        auto result = this->generate_local(pgm_op);
        this->set_multigrid_level(std::get<0>(result), std::get<1>(result),
                                  std::get<2>(result));
    }
}


#define GKO_DECLARE_PGM(_vtype, _itype) class Pgm<_vtype, _itype>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_PGM);


}  // namespace multigrid
}  // namespace gko
