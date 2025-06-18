// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/multigrid/fixed_coarsening.hpp"

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/distributed/base.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/distributed/partition_helpers.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/matrix/row_gatherer.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>

#include "core/base/dispatch_helper.hpp"
#include "core/base/segmented_array.hpp"
#include "core/base/utils.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/distributed/index_map_kernels.hpp"
#include "core/matrix/csr_builder.hpp"
#include "core/multigrid/fixed_coarsening_kernels.hpp"
#include "core/multigrid/pgm_kernels.hpp"


namespace gko {
namespace multigrid {
namespace fixed_coarsening {
namespace {


GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(fill_seq_array, components::fill_seq_array);
GKO_REGISTER_OPERATION(build_row_ptrs, fixed_coarsening::build_row_ptrs);
GKO_REGISTER_OPERATION(renumber, fixed_coarsening::renumber);
GKO_REGISTER_OPERATION(map_to_coarse, fixed_coarsening::map_to_coarse);
GKO_REGISTER_OPERATION(gather_index, pgm::gather_index);

}  // anonymous namespace
}  // namespace fixed_coarsening
namespace index_map {
namespace {


GKO_REGISTER_OPERATION(map_to_global, index_map::map_to_global);


}
}  // namespace index_map


// selected_rows is the sorted index will be used
// selected_cols_map gives the new col index from the old col index (invalid)
template <typename ValueType, typename IndexType>
std::shared_ptr<matrix::Csr<ValueType, IndexType>> build_coarse_matrix(
    const matrix::Csr<ValueType, IndexType>* origin, const size_type num_cols,
    const array<IndexType>& selected_rows,
    const array<IndexType>& selected_cols_map)
{
    auto exec = origin->get_executor();
    auto coarse = matrix::Csr<ValueType, IndexType>::create(
        exec, dim<2>{selected_rows.get_size(), num_cols});
    exec->run(fixed_coarsening::make_build_row_ptrs(
        origin->get_size()[0], origin->get_const_row_ptrs(),
        origin->get_const_col_idxs(), selected_rows, selected_cols_map,
        coarse->get_size()[0], coarse->get_row_ptrs()));
    auto coarse_nnz = static_cast<size_type>(
        exec->copy_val_to_host(coarse->get_row_ptrs() + coarse->get_size()[0]));
    array<ValueType> new_value_array{exec, coarse_nnz};
    array<IndexType> new_col_idx_array{exec, coarse_nnz};
    exec->run(fixed_coarsening::make_map_to_coarse(
        origin->get_size()[0], origin->get_const_row_ptrs(),
        origin->get_const_col_idxs(), origin->get_const_values(), selected_rows,
        selected_cols_map, coarse->get_size()[0], coarse->get_const_row_ptrs(),
        new_col_idx_array.get_data(), new_value_array.get_data()));
    matrix::CsrBuilder<ValueType, IndexType> mtx_builder{coarse};
    mtx_builder.get_value_array() = std::move(new_value_array);
    mtx_builder.get_col_idx_array() = std::move(new_col_idx_array);
    return coarse;
}


#if GINKGO_BUILD_MPI


template <typename ValueType, typename IndexType>
template <typename GlobalIndexType>
array<GlobalIndexType>
FixedCoarsening<ValueType, IndexType>::communicate_non_local_map(
    std::shared_ptr<const experimental::distributed::Matrix<
        ValueType, IndexType, GlobalIndexType>>
        matrix,
    std::shared_ptr<
        experimental::distributed::Partition<IndexType, GlobalIndexType>>
        coarse_partition,
    const array<IndexType>& local_map)
{
    auto exec = matrix->get_executor();
    const auto comm = matrix->get_communicator();
    auto coll_comm = matrix->row_gatherer_->get_collective_communicator();
    auto total_send_size = coll_comm->get_send_size();
    auto total_recv_size = coll_comm->get_recv_size();
    auto row_gatherer = matrix->row_gatherer_;

    array<IndexType> send_agg(exec, total_send_size);
    exec->run(fixed_coarsening::make_gather_index(
        send_agg.get_size(), local_map.get_const_data(),
        row_gatherer->get_const_send_idxs(), send_agg.get_data()));

    // There is no index map on the coarse level yet, so map the local indices
    // to global indices on the coarse level manually
    array<GlobalIndexType> send_global_agg(exec, send_agg.get_size());
    exec->run(index_map::make_map_to_global(
        to_device_const(coarse_partition.get()),
        device_segmented_array<const GlobalIndexType>{}, comm.rank(), send_agg,
        experimental::distributed::index_space::local, send_global_agg));

    array<GlobalIndexType> non_local_map(exec, total_recv_size);

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
    auto recv_ptr = use_host_buffer ? host_recv_buffer.get_data()
                                    : non_local_map.get_data();
    exec->synchronize();
    coll_comm
        ->i_all_to_all_v(use_host_buffer ? exec->get_master() : exec, send_ptr,
                         recv_ptr)
        .wait();
    if (use_host_buffer) {
        exec->copy_from(exec->get_master(), total_recv_size, recv_ptr,
                        non_local_map.get_data());
    }
    return non_local_map;
}


#endif


template <typename ValueType, typename IndexType>
void FixedCoarsening<ValueType, IndexType>::generate()
{
    using csr_type = matrix::Csr<ValueType, IndexType>;
    using real_type = remove_complex<ValueType>;

#if GINKGO_BUILD_MPI
    if (std::dynamic_pointer_cast<
            const experimental::distributed::DistributedBase>(system_matrix_)) {
        auto convert_fine_op = [&](auto matrix) {
            using global_index_type = typename std::decay_t<
                decltype(*matrix)>::result_type::global_index_type;
            auto exec = as<LinOp>(matrix)->get_executor();
            auto comm = as<experimental::distributed::DistributedBase>(matrix)
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
                using global_index_type =
                    typename std::decay_t<decltype(*matrix)>::global_index_type;
                convert_fine_op(
                    as<ConvertibleTo<experimental::distributed::Matrix<
                        ValueType, IndexType, global_index_type>>>(matrix));
            }
        };

        using fst_mtx_type =
            experimental::distributed::Matrix<ValueType, IndexType, IndexType>;
        using snd_mtx_type =
            experimental::distributed::Matrix<ValueType, IndexType, int64>;
        // setup the fine op using Csr with current ValueType
        // we do not use dispatcher run in the first place because we have
        // the fallback option for that.
        if (auto obj =
                std::dynamic_pointer_cast<const fst_mtx_type>(system_matrix_)) {
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
            auto local_op = gko::as<const csr_type>(matrix->get_local_matrix());

            size_type local_coarse_dim = parameters_.coarse_rows.get_size();

            auto local_fine_dim = local_op->get_size()[0];
            auto local_restrict_op = share(csr_type::create(
                exec, gko::dim<2>{local_coarse_dim, local_fine_dim},
                local_coarse_dim));
            exec->copy_from(parameters_.coarse_rows.get_executor(),
                            local_coarse_dim,
                            parameters_.coarse_rows.get_const_data(),
                            local_restrict_op->get_col_idxs());
            exec->run(fixed_coarsening::make_fill_array(
                local_restrict_op->get_values(), local_coarse_dim,
                one<ValueType>()));
            exec->run(fixed_coarsening::make_fill_seq_array(
                local_restrict_op->get_row_ptrs(), local_coarse_dim + 1));

            auto local_prolong_op =
                gko::as<csr_type>(share(local_restrict_op->transpose()));

            // generate the map from coarse_row. map[i] -> new index
            // it may gives some additional work for local case, but it gives
            // the neccessary information for distributed case
            array<IndexType> coarse_map(exec, local_fine_dim);
            coarse_map.fill(invalid_index<IndexType>());
            exec->run(fixed_coarsening::make_renumber(parameters_.coarse_rows,
                                                      &coarse_map));
            std::cout << "coarse_map: " << comm.rank() << ":";
            for (int i = 0; i < local_fine_dim; i++) {
                std::cout << " " << coarse_map.get_data()[i];
            }
            std::cout << std::endl;
            // TODO: Can be done with submatrix index_set.
            auto local_coarse_matrix =
                build_coarse_matrix(local_op.get(), local_coarse_dim,
                                    parameters_.coarse_rows, coarse_map);


            // create the coarse partition
            // the coarse partition will have only one range per part
            // and only one part per rank.
            // The global indices are ordered block-wise by rank, i.e. rank
            // 0 owns [0, ..., N_1), rank 1 [N_1, ..., N_2), ...
            auto coarse_local_size =
                static_cast<int64>(local_coarse_matrix->get_size()[0]);
            auto coarse_partition = gko::share(
                experimental::distributed::build_partition_from_local_size<
                    IndexType, global_index_type>(exec, comm,
                                                  coarse_local_size));

            // get the non-local aggregates as coarse global indices
            auto non_local_cols_map =
                communicate_non_local_map(matrix, coarse_partition, coarse_map);

            std::cout << "non_local_cols_map: " << comm.rank() << ":";
            for (int i = 0; i < non_local_cols_map.get_size(); i++) {
                std::cout << " " << non_local_cols_map.get_data()[i];
            }
            std::cout << std::endl;
            // // remove 0
            gko::size_type nnz = 0;

            for (int i = 0; i < non_local_cols_map.get_size(); i++) {
                nnz += non_local_cols_map.get_const_data()[i] !=
                       invalid_index<global_index_type>();
            }
            array<global_index_type> new_map(exec, nnz);
            int index = 0;
            for (int i = 0; i < non_local_cols_map.get_size(); i++) {
                const auto val = non_local_cols_map.get_const_data()[i];
                if (val != invalid_index<global_index_type>()) {
                    new_map.get_data()[index] = val;
                    index++;
                }
            }
            std::cout << "without -1: " << comm.rank() << ":";
            for (int i = 0; i < new_map.get_size(); i++) {
                std::cout << " " << new_map.get_data()[i];
            }
            std::cout << std::endl;
            // fflush(stdout);
            // exit(1);
            // create a coarse index map based on the connection given by
            // the non-local aggregates
            auto coarse_imap =
                experimental::distributed::index_map<IndexType,
                                                     global_index_type>(
                    exec, coarse_partition, comm.rank(), new_map);

            // a mapping from the fine non-local indices to the coarse
            // non-local indices.
            // non_local_agg already maps the fine non-local indices to
            // coarse global indices, so mapping it with the coarse index
            // map results in the coarse non-local indices.
            auto non_local_map = coarse_imap.map_to_local(
                non_local_cols_map,
                experimental::distributed::index_space::non_local);

            std::cout << "non_local_map: " << comm.rank() << ":";
            for (int i = 0; i < non_local_map.get_size(); i++) {
                std::cout << " " << non_local_map.get_data()[i];
            }
            std::cout << std::endl;
            // build csr from row and col map
            // unlike non-distributed version, generate_coarse uses
            // different row and col maps.
            auto non_local_csr =
                as<const csr_type>(matrix->get_non_local_matrix());
            auto result_non_local_csr = build_coarse_matrix(
                non_local_csr.get(), coarse_imap.get_non_local_size(),
                parameters_.coarse_rows, non_local_map);
            std::cout << "result_non_local_csr: " << comm.rank() << ": nnz-"
                      << result_non_local_csr->get_num_stored_elements()
                      << ": ";
            for (int i = 0; i < result_non_local_csr->get_num_stored_elements();
                 i++) {
                std::cout << " ("
                          << result_non_local_csr->get_const_col_idxs()[i]
                          << ", " << result_non_local_csr->get_const_values()[i]
                          << ")";
            }
            std::cout << std::endl;

            // setup the generated linop.
            auto coarse = share(
                experimental::distributed::
                    Matrix<ValueType, IndexType, global_index_type>::create(
                        exec, comm, std::move(coarse_imap), local_coarse_matrix,
                        result_non_local_csr));
            auto restrict_op = share(
                experimental::distributed::
                    Matrix<ValueType, IndexType, global_index_type>::create(
                        exec, comm,
                        dim<2>(coarse->get_size()[0],
                               gko::as<LinOp>(matrix)->get_size()[0]),
                        local_restrict_op));
            auto prolong_op = share(
                experimental::distributed::
                    Matrix<ValueType, IndexType, global_index_type>::create(
                        exec, comm,
                        dim<2>(gko::as<LinOp>(matrix)->get_size()[0],
                               coarse->get_size()[0]),
                        local_prolong_op));
            this->set_multigrid_level(prolong_op, coarse, restrict_op);
        };

        // the fine op is using csr with the current ValueType
        run<fst_mtx_type, snd_mtx_type>(this->get_fine_op(), distributed_setup);
    } else
#endif  // GINKGO_BUILD_MPI
    {
        auto exec = this->get_executor();
        const auto num_rows = this->system_matrix_->get_size()[0];

        // Only support csr matrix currently.
        const csr_type* fixed_coarsening_op =
            dynamic_cast<const csr_type*>(system_matrix_.get());
        std::shared_ptr<const csr_type> fixed_coarsening_op_shared_ptr{};
        // If system matrix is not csr or need sorting, generate the csr.
        if (!parameters_.skip_sorting || !fixed_coarsening_op) {
            fixed_coarsening_op_shared_ptr = convert_to_with_sorting<csr_type>(
                exec, system_matrix_, parameters_.skip_sorting);
            fixed_coarsening_op = fixed_coarsening_op_shared_ptr.get();
            // keep the same precision data in fine_op
            this->set_fine_op(fixed_coarsening_op_shared_ptr);
        }

        GKO_ASSERT(parameters_.coarse_rows.get_data() != nullptr);
        GKO_ASSERT(parameters_.coarse_rows.get_size() > 0);
        size_type coarse_dim = parameters_.coarse_rows.get_size();

        auto fine_dim = system_matrix_->get_size()[0];
        auto restrict_op = share(
            csr_type::create(exec, gko::dim<2>{coarse_dim, fine_dim},
                             coarse_dim, fixed_coarsening_op->get_strategy()));
        exec->copy_from(parameters_.coarse_rows.get_executor(), coarse_dim,
                        parameters_.coarse_rows.get_const_data(),
                        restrict_op->get_col_idxs());
        exec->run(fixed_coarsening::make_fill_array(
            restrict_op->get_values(), coarse_dim, one<ValueType>()));
        exec->run(fixed_coarsening::make_fill_seq_array(
            restrict_op->get_row_ptrs(), coarse_dim + 1));

        auto prolong_op = gko::as<csr_type>(share(restrict_op->transpose()));

        // generate the map from coarse_row. map[i] -> new index
        // it may gives some additional work for local case, but it gives the
        // neccessary information for distributed case
        array<IndexType> coarse_map(exec, fine_dim);
        coarse_map.fill(invalid_index<IndexType>());
        exec->run(fixed_coarsening::make_renumber(parameters_.coarse_rows,
                                                  &coarse_map));
        // TODO: Can be done with submatrix index_set.
        auto coarse_matrix =
            build_coarse_matrix(fixed_coarsening_op, coarse_dim,
                                parameters_.coarse_rows, coarse_map);
        coarse_matrix->set_strategy(fixed_coarsening_op->get_strategy());

        this->set_multigrid_level(prolong_op, coarse_matrix, restrict_op);
    }
}


#define GKO_DECLARE_FIXED_COARSENING(_vtype, _itype) \
    class FixedCoarsening<_vtype, _itype>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_FIXED_COARSENING);


}  // namespace multigrid
}  // namespace gko
