// SPDX-FileCopyrightText: 2025 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/multigrid/pmis.hpp"

#include <type_traits>
#include <vector>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/device_matrix_data.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/distributed/base.hpp>
#include <ginkgo/core/distributed/collective_communicator.hpp>
#include <ginkgo/core/distributed/index_map.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/distributed/partition_helpers.hpp>
#include <ginkgo/core/distributed/row_gatherer.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>

#include "core/base/array_access.hpp"
#include "core/base/dispatch_helper.hpp"
#include "core/base/utils.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/components/gather_kernels.hpp"
#include "core/components/precision_conversion_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/config/config_helper.hpp"
#include "core/distributed/helpers.hpp"
#include "core/distributed/index_map_kernels.hpp"
#include "core/matrix/csr_builder.hpp"
#include "core/multigrid/pmis_helpers.hpp"
#include "core/multigrid/pmis_kernels.hpp"


namespace gko {
namespace multigrid {
namespace pmis {
namespace {


GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(compute_row_maxabs, pmis::compute_row_maxabs);
GKO_REGISTER_OPERATION(compute_strong_dep_row, pmis::compute_strong_dep_row);
GKO_REGISTER_OPERATION(compute_strong_dep, pmis::compute_strong_dep);
GKO_REGISTER_OPERATION(initialize_weight_and_status,
                       pmis::initialize_weight_and_status);
GKO_REGISTER_OPERATION(fill_seq_array, components::fill_seq_array);
GKO_REGISTER_OPERATION(add_at_indices, pmis::add_at_indices);
GKO_REGISTER_OPERATION(count, pmis::count);
GKO_REGISTER_OPERATION(direct_interpolation_row_count,
                       pmis::direct_interpolation_row_count);
GKO_REGISTER_OPERATION(coarse_global_index, pmis::coarse_global_index);
GKO_REGISTER_OPERATION(convert_ptrs_to_idxs, components::convert_ptrs_to_idxs);
GKO_REGISTER_OPERATION(prefix_sum_nonnegative,
                       components::prefix_sum_nonnegative);
GKO_REGISTER_OPERATION(convert_precision, components::convert_precision);


}  // anonymous namespace
}  // namespace pmis


template <typename ValueType, typename IndexType>
typename Pmis<ValueType, IndexType>::parameters_type
Pmis<ValueType, IndexType>::parse(const config::pnode& config,
                                  const config::registry& context,
                                  const config::type_descriptor& td_for_child)
{
    auto params = Pmis<ValueType, IndexType>::build();
    config::config_check_decorator config_check(config);
    if (auto& obj = config_check.get("strength_threshold")) {
        params.with_strength_threshold(
            config::get_value<remove_complex<ValueType>>(obj));
    }
    if (auto& obj = config_check.get("skip_sorting")) {
        params.with_skip_sorting(config::get_value<bool>(obj));
    }

    return params;
}


template <typename ValueType, typename IndexType>
void Pmis<ValueType, IndexType>::apply_impl(const LinOp* b, LinOp* x) const
{
    this->get_composition()->apply(b, x);
}


template <typename ValueType, typename IndexType>
void Pmis<ValueType, IndexType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                            const LinOp* beta, LinOp* x) const
{
    this->get_composition()->apply(alpha, b, beta, x);
}


template <typename ValueType, typename IndexType>
Pmis<ValueType, IndexType>::Pmis(std::shared_ptr<const Executor> exec)
    : LinOp(std::move(exec))
{}


template <typename ValueType, typename IndexType>
Pmis<ValueType, IndexType>::Pmis(const Factory* factory,
                                 std::shared_ptr<const LinOp> system_matrix)
    : LinOp(factory->get_executor(), system_matrix->get_size()),
      EnableMultigridLevel<ValueType>(system_matrix),
      parameters_{factory->get_parameters()},
      system_matrix_{system_matrix}
{
    GKO_ASSERT(parameters_.strength_threshold <= 1.0);
    GKO_ASSERT(parameters_.strength_threshold >= 0.0);
    if (system_matrix_->get_size()[0] != 0) {
        // generate on the existed matrix
        this->generate();
    }
}


template <typename ValueType, typename IndexType>
void Pmis<ValueType, IndexType>::generate()
{
    using csr_type = matrix::Csr<ValueType, IndexType>;
#if GINKGO_BUILD_MPI
    if (std::dynamic_pointer_cast<
            const experimental::distributed::DistributedBase>(system_matrix_)) {
        using fst_mtx_type =
            experimental::distributed::Matrix<ValueType, IndexType, IndexType>;
        using snd_mtx_type =
            experimental::distributed::Matrix<ValueType, IndexType, int64>;
        auto distributed_setup = [&](auto matrix) {
            using global_index_type =
                typename std::decay_t<decltype(*matrix)>::global_index_type;
            auto exec = gko::as<LinOp>(matrix)->get_executor();
            auto comm =
                gko::as<experimental::distributed::DistributedBase>(matrix)
                    ->get_communicator();
            // the transpose below requires both partitions, and only
            // read_distributed sets the row partition
            GKO_THROW_IF_INVALID(
                matrix->get_row_partition() != nullptr,
                "distributed Pmis requires a matrix filled by "
                "read_distributed, which is what sets the row partition");

            auto diag = gko::as<const csr_type>(matrix->get_diag_matrix());
            auto off_diag =
                gko::as<const csr_type>(matrix->get_off_diag_matrix());
            const auto n_loc = diag->get_size()[0];
            const auto n_halo = off_diag->get_size()[1];

            // the row maximum spans both blocks; restricting it to the
            // diagonal block would change S and coarsen block-locally
            array<remove_complex<ValueType>> row_maxabs(exec, n_loc);
            exec->run(pmis::make_fill_array(row_maxabs.get_data(), n_loc,
                                            zero<remove_complex<ValueType>>()));
            exec->run(pmis::make_compute_row_maxabs(diag.get(), true,
                                                    row_maxabs.get_data()));
            exec->run(pmis::make_compute_row_maxabs(off_diag.get(), false,
                                                    row_maxabs.get_data()));

            array<IndexType> s_diag_rows(exec, n_loc + 1);
            array<IndexType> s_offd_rows(exec, n_loc + 1);
            exec->run(pmis::make_compute_strong_dep_row(
                diag.get(), true, row_maxabs.get_const_data(),
                parameters_.strength_threshold, s_diag_rows.get_data()));
            exec->run(pmis::make_compute_strong_dep_row(
                off_diag.get(), false, row_maxabs.get_const_data(),
                parameters_.strength_threshold, s_offd_rows.get_data()));
            exec->run(pmis::make_prefix_sum_nonnegative(s_diag_rows.get_data(),
                                                        n_loc + 1));
            exec->run(pmis::make_prefix_sum_nonnegative(s_offd_rows.get_data(),
                                                        n_loc + 1));
            auto s_diag = matrix::SparsityCsr<ValueType, IndexType>::create(
                exec, dim<2>{n_loc, n_loc},
                array<IndexType>{exec, static_cast<size_type>(
                                           get_element(s_diag_rows, n_loc))},
                std::move(s_diag_rows));
            auto s_offd = matrix::SparsityCsr<ValueType, IndexType>::create(
                exec, dim<2>{n_loc, n_halo},
                array<IndexType>{exec, static_cast<size_type>(
                                           get_element(s_offd_rows, n_loc))},
                std::move(s_offd_rows));
            exec->run(pmis::make_compute_strong_dep(
                diag.get(), true, row_maxabs.get_const_data(),
                parameters_.strength_threshold, s_diag.get()));
            exec->run(pmis::make_compute_strong_dep(
                off_diag.get(), false, row_maxabs.get_const_data(),
                parameters_.strength_threshold, s_offd.get()));

            // the measure of node i is how often i occurs as a column of S,
            // counted by scatter-add because SparsityCsr::transpose has no
            // CUDA/HIP implementation
            const auto n_ext = n_loc + n_halo;
            // a null values argument means "add one"
            const IndexType* count_ones = nullptr;
            const auto count_columns = [&](const auto* graph, size_type num,
                                           IndexType* counts) {
                exec->run(
                    pmis::make_fill_array(counts, num, zero<IndexType>()));
                exec->run(pmis::make_add_at_indices(graph->get_num_nonzeros(),
                                                    graph->get_const_col_idxs(),
                                                    count_ones, counts));
            };
            array<IndexType> measure(exec, n_loc);
            count_columns(s_diag.get(), n_loc, measure.get_data());
            array<IndexType> halo_counts(exec, n_halo);
            count_columns(s_offd.get(), n_halo, halo_counts.get_data());

            auto row_gatherer = matrix->get_row_gatherer();
            auto coll_comm = row_gatherer->get_collective_communicator();
            const auto n_send = row_gatherer->get_num_send_idxs();
            const auto send_idxs = row_gatherer->get_const_send_idxs();
            GKO_ASSERT_EQ(static_cast<size_type>(coll_comm->get_recv_size()),
                          n_halo);
            // exchange_with_neighbors is forward-only; the measure travels
            // halo -> owner, hence the inverse communicator
            auto inverse_comm = coll_comm->create_inverse();
            auto owner_counts = gko::detail::exchange_with_neighbors(
                exec, comm, inverse_comm.get(), halo_counts);
            // a row sent to several neighbours appears several times in
            // send_idxs, so this scatter-add has to be atomic
            exec->run(pmis::make_add_at_indices(
                owner_counts.get_size(), send_idxs,
                owner_counts.get_const_data(), measure.get_data()));

            array<remove_complex<ValueType>> weight(exec, n_ext);
            array<int> status(exec, n_ext);
            array<global_index_type> global_idx(exec, n_ext);
            // a halo entry carries the global row index of its owner, and
            // the random draw is keyed on that index, so the index has to be
            // built before the draw
            const experimental::distributed::index_map<IndexType,
                                                       global_index_type>
                row_imap{exec, matrix->get_row_partition(), comm.rank(),
                         array<global_index_type>{exec}};
            array<IndexType> local_rows(exec, n_loc);
            exec->run(pmis::make_fill_seq_array(local_rows.get_data(), n_loc));
            const auto global_rows = row_imap.map_to_global(
                local_rows, experimental::distributed::index_space::local);
            exec->copy_from(exec, n_loc, global_rows.get_const_data(),
                            global_idx.get_data());
            exec->run(pmis::make_initialize_weight_and_status(
                n_loc, measure.get_const_data(), global_idx.get_const_data(),
                weight.get_data(), status.get_data()));

            // Broadcasts owner -> halo, in place on the extended array. It
            // may only be called once the local values are final, so that
            // every halo copy is identical to the value on its owner. The
            // buffers are owned by the caller, so that the loop below can
            // reuse them.
            auto broadcast_to_halo = [&](auto* ext, auto& send_buffer,
                                         auto& scratch) {
                using payload_type = std::decay_t<decltype(*ext)>;
                const payload_type* src = ext;
                send_buffer.resize_and_reset(n_send);
                exec->run(pmis::make_gather(n_send, src, send_idxs,
                                            send_buffer.get_data()));
                const auto& recv = gko::detail::exchange_with_neighbors(
                    exec, comm, coll_comm.get(), send_buffer, scratch);
                exec->copy_from(exec, n_halo, recv.get_const_data(),
                                ext + n_loc);
            };
            // for payloads that are broadcast only once
            auto broadcast_once = [&](auto* ext) {
                using payload_type = std::decay_t<decltype(*ext)>;
                array<payload_type> send_buffer{exec};
                gko::detail::neighbor_exchange_buffers<payload_type> scratch{
                    exec};
                broadcast_to_halo(ext, send_buffer, scratch);
            };
            broadcast_once(weight.get_data());
            broadcast_once(status.get_data());
            broadcast_once(global_idx.get_data());

            const std::vector<pmis::column_block<
                matrix::SparsityCsr<ValueType, IndexType>, IndexType>>
                strength_blocks{
                    {s_diag.get(), false, IndexType{0}},
                    {s_offd.get(), false, static_cast<IndexType>(n_loc)}};

            array<int> new_status(exec, n_ext);
            auto status_ptr = status.get_data();
            auto new_status_ptr = new_status.get_data();
            // the loop below refreshes the halo twice per round, so its
            // buffers are allocated outside of it
            array<int> status_send_buffer{exec};
            gko::detail::neighbor_exchange_buffers<int> status_scratch{exec};
            auto refresh_status_halo = [&](int* ext) {
                broadcast_to_halo(ext, status_send_buffer, status_scratch);
            };
            // termination is global: a rank can finish before its neighbours
            size_type local_unassigned = 0;
            exec->run(pmis::make_count(n_loc, status_ptr, &local_unassigned));
            size_type global_unassigned = 0;
            comm.all_reduce(exec->get_master(), &local_unassigned,
                            &global_unassigned, 1, MPI_SUM);
            while (global_unassigned != 0) {
                pmis::classify_round(
                    exec, n_loc, strength_blocks, weight.get_const_data(),
                    global_idx.get_const_data(), status_ptr, new_status_ptr,
                    // mark_fine has to see the C-points just selected on
                    // other ranks
                    [&] { refresh_status_halo(new_status_ptr); });
                // the select phase of the next round reads the halo
                // statuses again
                refresh_status_halo(new_status_ptr);
                size_type new_local = 0;
                exec->run(pmis::make_count(n_loc, new_status_ptr, &new_local));
                size_type new_global = 0;
                comm.all_reduce(exec->get_master(), &new_local, &new_global, 1,
                                MPI_SUM);
                GKO_THROW_IF_INVALID(new_global != global_unassigned,
                                     "no progress in Pmis");
                global_unassigned = new_global;
                std::swap(new_status_ptr, status_ptr);
            }
            // the swap leaves the final splitting in status_ptr, which is
            // not necessarily status.get_data()

            // coarse partition and coarse global indices
            static_assert(
                kernels::pmis::coarse == 1 && kernels::pmis::fine == 0,
                "the prefix sums below convert the status directly, "
                "which needs fine == 0 and coarse == 1");
            array<IndexType> coarse_map(exec, n_loc + 1);
            exec->run(pmis::make_convert_precision(n_loc, status_ptr,
                                                   coarse_map.get_data()));
            exec->run(pmis::make_prefix_sum_nonnegative(coarse_map.get_data(),
                                                        n_loc + 1));
            const auto local_coarse =
                static_cast<size_type>(get_element(coarse_map, n_loc));
            auto coarse_partition = share(
                experimental::distributed::build_partition_from_local_size<
                    IndexType, global_index_type>(exec, comm, local_coarse));
            GKO_THROW_IF_INVALID(coarse_partition->get_size() > 0,
                                 "Pmis produced an empty coarse level.");
            // this rank's first coarse global index: the exclusive prefix
            // sum of the local C-point counts
            size_type coarse_scan = 0;
            comm.scan(exec->get_master(), &local_coarse, &coarse_scan, 1,
                      MPI_SUM);
            const auto coarse_offset =
                static_cast<global_index_type>(coarse_scan - local_coarse);
            array<global_index_type> coarse_global(exec, n_ext);
            exec->run(pmis::make_coarse_global_index(
                n_loc, coarse_offset, status_ptr, coarse_map.get_const_data(),
                coarse_global.get_data()));
            // a fine row can interpolate from a C-point on another rank
            broadcast_once(coarse_global.get_data());

            array<IndexType> prolong_row_ptrs(exec, n_loc + 1);
            // seed each coarse row with its single identity entry, which is
            // exactly what converting the status yields
            exec->run(pmis::make_convert_precision(
                n_loc, status_ptr, prolong_row_ptrs.get_data()));
            for (const auto& block : strength_blocks) {
                exec->run(pmis::make_direct_interpolation_row_count(
                    block.col_offset, block.mtx, status_ptr,
                    prolong_row_ptrs.get_data()));
            }
            exec->run(pmis::make_prefix_sum_nonnegative(
                prolong_row_ptrs.get_data(), n_loc + 1));
            const auto prolong_nnz =
                static_cast<size_type>(get_element(prolong_row_ptrs, n_loc));

            array<IndexType> prolong_col_idxs(exec, prolong_nnz);
            array<ValueType> prolong_values(exec, prolong_nnz);
            pmis::fill_prolongation<ValueType, IndexType>(
                exec, n_loc,
                {{diag.get(), true, IndexType{0}},
                 {off_diag.get(), false, static_cast<IndexType>(n_loc)}},
                row_maxabs.get_const_data(), parameters_.strength_threshold,
                status_ptr, prolong_row_ptrs.get_const_data(),
                prolong_col_idxs.get_data(), prolong_values.get_data());

            // node indices cannot encode the coarse column of a halo
            // C-point, hence the global triplets
            array<global_index_type> p_cols(exec, prolong_nnz);
            exec->run(pmis::make_gather(
                prolong_nnz, coarse_global.get_const_data(),
                prolong_col_idxs.get_const_data(), p_cols.get_data()));
            array<IndexType> p_local_rows(exec, prolong_nnz);
            exec->run(pmis::make_convert_ptrs_to_idxs(
                prolong_row_ptrs.get_const_data(), n_loc,
                p_local_rows.get_data()));
            auto p_rows = row_imap.map_to_global(
                p_local_rows, experimental::distributed::index_space::local);

            using dist_mtx_type =
                experimental::distributed::Matrix<ValueType, IndexType,
                                                  global_index_type>;
            device_matrix_data<ValueType, global_index_type> p_data{
                exec,
                dim<2>{matrix->get_size()[0], coarse_partition->get_size()},
                std::move(p_rows), std::move(p_cols),
                std::move(prolong_values)};
            auto prolong_op = share(dist_mtx_type::create(exec, comm));
            // local_only: every rank emits only the rows it owns
            prolong_op->read_distributed(
                p_data, matrix->get_row_partition(), coarse_partition,
                experimental::distributed::assembly_mode::local_only);

            // restriction and Galerkin coarse operator
            auto restrict_op = share(dist_mtx_type::create(exec, comm));
            prolong_op->transpose(restrict_op);
            // input and output must not alias, hence the separate tmp
            auto tmp = dist_mtx_type::create(exec, comm);
            matrix->multiply(prolong_op, tmp);
            auto coarse_op = share(dist_mtx_type::create(exec, comm));
            restrict_op->multiply(tmp, coarse_op);
            this->set_multigrid_level(prolong_op, coarse_op, restrict_op);
        };
        run<fst_mtx_type, snd_mtx_type>(this->get_fine_op(), distributed_setup);
    } else
#endif  // GINKGO_BUILD_MPI
    {
        auto exec = this->get_executor();
        // the kernels require a sorted Csr matrix, so convert if necessary
        auto pmis_op =
            std::dynamic_pointer_cast<const csr_type>(system_matrix_);
        if (!parameters_.skip_sorting || !pmis_op) {
            pmis_op = convert_to_with_sorting<csr_type>(
                exec, system_matrix_, parameters_.skip_sorting);
            // keep the same precision data in fine_op
            this->set_fine_op(pmis_op);
        }

        array<IndexType> sparsity_rows(exec, pmis_op->get_size()[0] + 1);
        array<remove_complex<ValueType>> row_maxabs(exec,
                                                    pmis_op->get_size()[0]);
        gko::array<remove_complex<ValueType>> weight_(exec,
                                                      pmis_op->get_size()[0]);
        // compute_row_maxabs accumulates, so start from zero
        exec->run(pmis::make_fill_array(row_maxabs.get_data(),
                                        row_maxabs.get_size(),
                                        zero<remove_complex<ValueType>>()));
        exec->run(pmis::make_compute_row_maxabs(pmis_op.get(), true,
                                                row_maxabs.get_data()));
        // store |S_i| in sparsity_rows[i]
        exec->run(pmis::make_compute_strong_dep_row(
            pmis_op.get(), true, row_maxabs.get_const_data(),
            this->get_parameters().strength_threshold,
            sparsity_rows.get_data()));
        exec->run(pmis::make_prefix_sum_nonnegative(sparsity_rows.get_data(),
                                                    sparsity_rows.get_size()));
        auto nnz = get_element(sparsity_rows, pmis_op->get_size()[0]);
        array<IndexType> sparsity_cols(exec, nnz);
        auto strong_dep = matrix::SparsityCsr<ValueType, IndexType>::create(
            exec, pmis_op->get_size(), std::move(sparsity_cols),
            std::move(sparsity_rows));
        exec->run(pmis::make_compute_strong_dep(
            pmis_op.get(), true, row_maxabs.get_const_data(),
            this->get_parameters().strength_threshold, strong_dep.get()));
        // status: -1 unassigned, 0 fine, 1 coarse
        gko::array<int> status(exec, this->get_size()[0]);
        gko::array<int> new_status(exec, this->get_size()[0]);
        auto status_ptr = status.get_data();
        auto new_status_ptr = new_status.get_data();
        // how often i occurs as a column of S, counted by scatter-add
        array<IndexType> counts(exec, this->get_size()[0]);
        exec->run(pmis::make_fill_array(counts.get_data(), this->get_size()[0],
                                        zero<IndexType>()));
        // a null values argument means "add one"
        const IndexType* count_ones = nullptr;
        exec->run(pmis::make_add_at_indices(strong_dep->get_num_nonzeros(),
                                            strong_dep->get_const_col_idxs(),
                                            count_ones, counts.get_data()));
        // there is no halo in the local case, so the global index equals the
        // local one. The random draw is keyed on it, so it has to be built
        // first.
        array<IndexType> global_idx(exec, this->get_size()[0]);
        exec->run(pmis::make_fill_seq_array(global_idx.get_data(),
                                            this->get_size()[0]));
        exec->run(pmis::make_initialize_weight_and_status(
            this->get_size()[0], counts.get_const_data(),
            global_idx.get_const_data(), weight_.get_data(), status_ptr));
        size_type num_not_assigned = 0;

        // a single block, so no exchange between the classify phases
        const std::vector<pmis::column_block<
            matrix::SparsityCsr<ValueType, IndexType>, IndexType>>
            strength_blocks{{strong_dep.get(), false, IndexType{0}}};

        exec->run(pmis::make_count(this->get_size()[0], status_ptr,
                                   &num_not_assigned));
        while (num_not_assigned != 0) {
            pmis::classify_round(exec, this->get_size()[0], strength_blocks,
                                 weight_.get_const_data(),
                                 global_idx.get_const_data(), status_ptr,
                                 new_status_ptr, [] {});
            size_type new_num = 0;
            exec->run(pmis::make_count(this->get_size()[0], new_status_ptr,
                                       &new_num));
            GKO_THROW_IF_INVALID(new_num != num_not_assigned,
                                 "no progress in Pmis");
            num_not_assigned = new_num;
            std::swap(new_status_ptr, status_ptr);
        }
        array<IndexType> prolong_row_ptrs(exec, pmis_op->get_size()[0] + 1);
        // direct_interpolation_row_count skips coarse rows, so seed their
        // identity entry, which is exactly what converting the status yields
        exec->run(pmis::make_convert_precision(
            pmis_op->get_size()[0], status_ptr, prolong_row_ptrs.get_data()));
        exec->run(pmis::make_direct_interpolation_row_count(
            IndexType{0}, strong_dep.get(), status_ptr,
            prolong_row_ptrs.get_data()));

        // coarse_map[i] holds the coarse index of i if i is a C-point, and
        // is undefined otherwise. It is kept separate from status_ptr, which
        // the classify loop iterates on and which is cheaper to store as int.
        array<IndexType> coarse_map(exec, pmis_op->get_size()[0] + 1);
        static_assert(kernels::pmis::coarse == 1 && kernels::pmis::fine == 0,
                      "we perform prefix sum directly by having fine == 0 and "
                      "coarse == 1");
        exec->run(pmis::make_convert_precision(
            pmis_op->get_size()[0], status_ptr, coarse_map.get_data()));
        exec->run(pmis::make_prefix_sum_nonnegative(coarse_map.get_data(),
                                                    this->get_size()[0] + 1));
        auto num_coarse = static_cast<size_type>(
            get_element(coarse_map, this->get_size()[0]));
        GKO_THROW_IF_INVALID(num_coarse > 0,
                             "Pmis produced an empty coarse level.");

        exec->run(pmis::make_prefix_sum_nonnegative(prolong_row_ptrs.get_data(),
                                                    this->get_size()[0] + 1));
        IndexType prolong_nnz =
            get_element(prolong_row_ptrs, this->get_size()[0]);
        array<IndexType> prolong_col_idxs(exec, prolong_nnz);
        array<ValueType> prolong_values(exec, prolong_nnz);

        // Direct interpolation: a C-point i gets w_{i, c[i]} = 1, a fine
        // point gets w_{i, c[k]} = -{alpha_i or beta_i} * a_ik / a_ii for each
        // strong C-neighbour k, with
        //   alpha_i = sum_j(a_ij < 0) / sum_k(a_ik < 0, S_ik, k coarse)
        //   beta_i  = sum_j(a_ij > 0) / sum_k(a_ik > 0, S_ik, k coarse)
        // If the denominator of alpha_i or beta_i is empty, the entries that
        // would use it are dropped.
        pmis::fill_prolongation<ValueType, IndexType>(
            exec, pmis_op->get_size()[0], {{pmis_op.get(), true, IndexType{0}}},
            row_maxabs.get_const_data(),
            this->get_parameters().strength_threshold, status_ptr,
            prolong_row_ptrs.get_const_data(), prolong_col_idxs.get_data(),
            prolong_values.get_data());
        // without a halo, the prefix-summed coarse_map maps node indices to
        // coarse columns directly. Input and output alias, which is safe
        // because entry i only reads entry i.
        exec->run(pmis::make_gather(
            static_cast<size_type>(prolong_nnz), coarse_map.get_const_data(),
            prolong_col_idxs.get_const_data(), prolong_col_idxs.get_data()));
        auto prolongation = share(matrix::Csr<ValueType, IndexType>::create(
            exec, dim<2>{pmis_op->get_size()[0], num_coarse},
            std::move(prolong_values), std::move(prolong_col_idxs),
            std::move(prolong_row_ptrs)));
        auto restriction = share(prolongation->transpose());
        auto internal = matrix::Csr<ValueType, IndexType>::create(
            exec, prolongation->get_size());
        auto coarse = share(matrix::Csr<ValueType, IndexType>::create(
            exec, dim<2>{num_coarse, num_coarse}));
        pmis_op->apply(prolongation, internal);
        restriction->apply(internal, coarse);
        this->set_multigrid_level(prolongation, coarse, restriction);
    }
}


#define GKO_DECLARE_PMIS(_vtype, _itype) class Pmis<_vtype, _itype>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_PMIS);


}  // namespace multigrid
}  // namespace gko
