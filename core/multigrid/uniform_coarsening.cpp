// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
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
#include <ginkgo/core/multigrid/uniform_coarsening.hpp>

#include "core/base/device_matrix_data_kernels.hpp"
#include "core/base/dispatch_helper.hpp"
#include "core/base/utils.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/config/config_helper.hpp"
#include "core/distributed/index_map_kernels.hpp"
#include "core/matrix/csr_builder.hpp"
#include "core/multigrid/pgm_kernels.hpp"
#include "core/multigrid/uniform_coarsening_kernels.hpp"
#include "ginkgo/core/base/math.hpp"


namespace gko {
namespace multigrid {
namespace uniform_coarsening {
namespace {


GKO_REGISTER_OPERATION(fill_restrict_op, uniform_coarsening::fill_restrict_op);
GKO_REGISTER_OPERATION(fill_incremental_indices,
                       uniform_coarsening::fill_incremental_indices);
GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(fill_seq_array, components::fill_seq_array);
GKO_REGISTER_OPERATION(gather_index, pgm::gather_index);
GKO_REGISTER_OPERATION(map_row, pgm::map_row);
GKO_REGISTER_OPERATION(map_col, pgm::map_col);
GKO_REGISTER_OPERATION(sort_agg, pgm::sort_agg);
GKO_REGISTER_OPERATION(sort_row_major, components::sort_row_major);
GKO_REGISTER_OPERATION(count_unrepeated_nnz, pgm::count_unrepeated_nnz);
GKO_REGISTER_OPERATION(compute_coarse_coo, pgm::compute_coarse_coo);
GKO_REGISTER_OPERATION(convert_idxs_to_ptrs, components::convert_idxs_to_ptrs);


}  // anonymous namespace
}  // namespace uniform_coarsening


namespace index_map {
namespace {


GKO_REGISTER_OPERATION(map_to_global, index_map::map_to_global);


}
}  // namespace index_map


namespace {


template <typename ValueType, typename IndexType>
std::shared_ptr<matrix::Csr<ValueType, IndexType>> generate_coarse(
    std::shared_ptr<const Executor> exec,
    const matrix::Csr<ValueType, IndexType>* fine_csr, IndexType num_agg,
    const gko::array<IndexType>& agg, IndexType non_local_num_agg,
    const gko::array<IndexType>& non_local_agg)
{
    const auto num = fine_csr->get_size()[0];
    auto nnz = fine_csr->get_num_stored_elements();
    gko::array<IndexType> row_idxs(exec, nnz);
    gko::array<IndexType> col_idxs(exec, nnz);
    gko::array<ValueType> vals(exec, nnz);
    exec->copy_from(exec, nnz, fine_csr->get_const_values(), vals.get_data());

    if (nnz == 0) {
        return matrix::Csr<ValueType, IndexType>::create(
            exec, dim<2>(num_agg, non_local_num_agg));
    }

    // map row_ptrs to coarse row index (may produce -1 for non-coarse rows)
    exec->run(uniform_coarsening::make_map_row(
        num, fine_csr->get_const_row_ptrs(), agg.get_const_data(),
        row_idxs.get_data()));
    // map col_idxs to coarse col index (may produce -1 for non-coarse cols)
    exec->run(uniform_coarsening::make_map_col(
        nnz, fine_csr->get_const_col_idxs(), non_local_agg.get_const_data(),
        col_idxs.get_data()));

    // Unlike PGM where every row has a valid aggregate, uniform coarsening
    // skips rows (mapped to -1). Filter out entries with invalid row/col.
    {
        auto host_exec = exec->get_master();
        array<IndexType> h_rows{host_exec, row_idxs};
        array<IndexType> h_cols{host_exec, col_idxs};
        array<ValueType> h_vals{host_exec, vals};
        size_type valid_nnz = 0;
        for (size_type i = 0; i < nnz; i++) {
            if (h_rows.get_const_data()[i] >= 0 &&
                h_cols.get_const_data()[i] >= 0) {
                h_rows.get_data()[valid_nnz] = h_rows.get_const_data()[i];
                h_cols.get_data()[valid_nnz] = h_cols.get_const_data()[i];
                h_vals.get_data()[valid_nnz] = h_vals.get_const_data()[i];
                valid_nnz++;
            }
        }
        if (valid_nnz == 0) {
            return matrix::Csr<ValueType, IndexType>::create(
                exec, dim<2>(num_agg, non_local_num_agg));
        }
        // Skip reassignment if nothing was filtered
        if (valid_nnz < nnz) {
            nnz = valid_nnz;
            row_idxs.resize_and_reset(valid_nnz);
            col_idxs.resize_and_reset(valid_nnz);
            vals.resize_and_reset(valid_nnz);
            exec->copy_from(host_exec, valid_nnz, h_rows.get_const_data(),
                            row_idxs.get_data());
            exec->copy_from(host_exec, valid_nnz, h_cols.get_const_data(),
                            col_idxs.get_data());
            exec->copy_from(host_exec, valid_nnz, h_vals.get_const_data(),
                            vals.get_data());
        }
    }

    // sort by row, col
    exec->run(uniform_coarsening::make_sort_row_major(
        nnz, row_idxs.get_data(), col_idxs.get_data(), vals.get_data()));
    // compute the total nnz and create the coarse csr
    size_type coarse_nnz = 0;
    exec->run(uniform_coarsening::make_count_unrepeated_nnz(
        nnz, row_idxs.get_const_data(), col_idxs.get_const_data(),
        &coarse_nnz));

    // reduce by key (row, col)
    auto coarse_coo = matrix::Coo<ValueType, IndexType>::create(
        exec,
        gko::dim<2>{static_cast<size_type>(num_agg),
                    static_cast<size_type>(non_local_num_agg)},
        coarse_nnz);
    exec->run(uniform_coarsening::make_compute_coarse_coo(
        nnz, row_idxs.get_const_data(), col_idxs.get_const_data(),
        vals.get_const_data(), coarse_coo->get_device_view()));
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
typename UniformCoarsening<ValueType, IndexType>::parameters_type
UniformCoarsening<ValueType, IndexType>::parse(
    const config::pnode& config, const config::registry& context,
    const config::type_descriptor& td_for_child)
{
    auto params = UniformCoarsening<ValueType, IndexType>::build();
    if (auto& obj = config.get("coarse_skip")) {
        params.with_coarse_skip(gko::config::get_value<int>(obj));
    }
    if (auto& obj = config.get("aggregation")) {
        params.with_aggregation(gko::config::get_value<bool>(obj));
    }
    if (auto& obj = config.get("skip_sorting")) {
        params.with_skip_sorting(gko::config::get_value<bool>(obj));
    }

    return params;
}


template <typename ValueType, typename IndexType>
std::tuple<std::shared_ptr<LinOp>, std::shared_ptr<LinOp>,
           std::shared_ptr<LinOp>>
UniformCoarsening<ValueType, IndexType>::generate_local(
    std::shared_ptr<const matrix::Csr<ValueType, IndexType>> local_matrix)
{
    using csr_type = matrix::Csr<ValueType, IndexType>;
    auto exec = this->get_executor();
    const auto num_rows = local_matrix->get_size()[0];

    // Only support csr matrix currently.
    const csr_type* uniform_coarsening_op =
        dynamic_cast<const csr_type*>(local_matrix.get());
    std::shared_ptr<const csr_type> uniform_coarsening_op_shared_ptr{};
    // If system matrix is not csr or need sorting, generate the csr.
    if (!parameters_.skip_sorting || !uniform_coarsening_op) {
        uniform_coarsening_op_shared_ptr = convert_to_with_sorting<csr_type>(
            exec, local_matrix, parameters_.skip_sorting);
        uniform_coarsening_op = uniform_coarsening_op_shared_ptr.get();
    }
    const auto skip = parameters_.coarse_skip;
    gko::dim<2>::dimension_type coarse_dim = gko::ceildiv(num_rows, skip);
    auto fine_dim = num_rows;

    coarse_rows_ = array<IndexType>(exec, num_rows);

    std::shared_ptr<csr_type> prolong_op;
    std::shared_ptr<csr_type> restrict_op;
    std::shared_ptr<csr_type> coarse_matrix;

    if (parameters_.aggregation) {
        // Aggregate-style: map every fine row to floor(i/skip).
        // This preserves connectivity in the Galerkin coarse matrix.
        {
            auto host = exec->get_master();
            array<IndexType> h_rows(host, num_rows);
            for (size_type i = 0; i < num_rows; ++i) {
                h_rows.get_data()[i] = static_cast<IndexType>(i / skip);
            }
            exec->copy_from(host, num_rows, h_rows.get_const_data(),
                            coarse_rows_.get_data());
        }

        // Build restriction R (coarse_dim × fine_dim) as aggregation
        // matrix: R[c, f] = 1 for all f where coarse_rows_[f] == c.
        {
            auto host = exec->get_master();
            std::vector<IndexType> row_ptrs(coarse_dim + 1, 0);
            for (size_type i = 0; i < num_rows; ++i) {
                row_ptrs[i / skip + 1]++;
            }
            for (size_type i = 0; i < coarse_dim; ++i) {
                row_ptrs[i + 1] += row_ptrs[i];
            }
            auto total_nnz = static_cast<size_type>(row_ptrs[coarse_dim]);
            std::vector<IndexType> col_idxs(total_nnz);
            std::vector<ValueType> values(total_nnz, one<ValueType>());
            std::vector<IndexType> offsets(coarse_dim, 0);
            for (size_type i = 0; i < num_rows; ++i) {
                auto c = static_cast<size_type>(i / skip);
                col_idxs[row_ptrs[c] + offsets[c]] = static_cast<IndexType>(i);
                offsets[c]++;
            }
            restrict_op = share(csr_type::create(
                exec, gko::dim<2>{coarse_dim, fine_dim}, total_nnz,
                uniform_coarsening_op->get_strategy()));
            exec->copy_from(host, coarse_dim + 1, row_ptrs.data(),
                            restrict_op->get_row_ptrs());
            exec->copy_from(host, total_nnz, col_idxs.data(),
                            restrict_op->get_col_idxs());
            exec->copy_from(host, total_nnz, values.data(),
                            restrict_op->get_values());

            prolong_op = gko::as<csr_type>(share(restrict_op->transpose()));

            coarse_matrix = share(generate_coarse(
                exec, uniform_coarsening_op, static_cast<IndexType>(coarse_dim),
                coarse_rows_));
        }
    } else {
        // Default path: row-selection (injection) restriction.
        coarse_rows_.fill(-one<IndexType>());
        exec->run(uniform_coarsening::make_fill_incremental_indices(
            skip, &coarse_rows_));

        restrict_op = share(csr_type::create(
            exec, gko::dim<2>{coarse_dim, fine_dim}, coarse_dim,
            uniform_coarsening_op->get_strategy()));
        exec->run(uniform_coarsening::make_fill_restrict_op(&coarse_rows_,
                                                            restrict_op.get()));
        exec->run(uniform_coarsening::make_fill_array(
            restrict_op->get_values(), coarse_dim, one<ValueType>()));
        exec->run(uniform_coarsening::make_fill_seq_array(
            restrict_op->get_row_ptrs(), coarse_dim + 1));

        prolong_op = gko::as<csr_type>(share(restrict_op->transpose()));

        coarse_matrix = share(
            generate_coarse(exec, uniform_coarsening_op,
                            static_cast<IndexType>(coarse_dim), coarse_rows_));
        coarse_matrix->set_strategy(uniform_coarsening_op->get_strategy());
    }

    return std::make_tuple(std::shared_ptr<LinOp>(prolong_op),
                           std::shared_ptr<LinOp>(coarse_matrix),
                           std::shared_ptr<LinOp>(restrict_op));
}


#if GINKGO_BUILD_MPI


template <typename ValueType, typename IndexType>
template <typename GlobalIndexType>
array<GlobalIndexType>
UniformCoarsening<ValueType, IndexType>::communicate_non_local_agg(
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
    exec->run(uniform_coarsening::make_gather_index(
        send_agg.get_size(), local_agg.get_const_data(),
        row_gatherer->get_const_send_idxs(), send_agg.get_data()));

    // Map local coarse indices to global indices on the coarse level
    array<GlobalIndexType> send_global_agg(exec, send_agg.get_size());
    exec->run(index_map::make_map_to_global(
        to_device_const(coarse_partition.get()),
        device_segmented_array<const GlobalIndexType>{}, comm.rank(), send_agg,
        experimental::distributed::index_space::local, send_global_agg));

    array<GlobalIndexType> non_local_agg(exec, total_recv_size);

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
                                    : non_local_agg.get_data();
    exec->synchronize();
    coll_comm
        ->i_all_to_all_v(use_host_buffer ? exec->get_master() : exec, send_ptr,
                         recv_ptr)
        .wait();
    if (use_host_buffer) {
        exec->copy_from(exec->get_master(), total_recv_size, recv_ptr,
                        non_local_agg.get_data());
    }
    return non_local_agg;
}


#endif


template <typename ValueType, typename IndexType>
void UniformCoarsening<ValueType, IndexType>::generate()
{
    using csr_type = matrix::Csr<ValueType, IndexType>;
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
                matrix->get_diag_matrix());
            auto non_local_csr = std::dynamic_pointer_cast<const csr_type>(
                matrix->get_off_diag_matrix());
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
            auto local_csr = gko::as<const csr_type>(matrix->get_diag_matrix());
            auto result = this->generate_local(local_csr);

            // Create the coarse partition from local coarse size
            auto coarse_local_size =
                static_cast<int64>(std::get<1>(result)->get_size()[0]);
            auto coarse_partition = gko::share(
                experimental::distributed::build_partition_from_local_size<
                    IndexType, global_index_type>(exec, comm,
                                                  coarse_local_size));

            // Get non-local aggregates as coarse global indices
            auto non_local_agg = communicate_non_local_agg(
                matrix, coarse_partition, coarse_rows_);

            // Create coarse index map from non-local aggregates
            auto coarse_imap =
                experimental::distributed::index_map<IndexType,
                                                     global_index_type>(
                    exec, coarse_partition, comm.rank(), non_local_agg);

            // Map fine non-local indices to coarse non-local indices
            auto non_local_map = coarse_imap.map_to_local(
                non_local_agg,
                experimental::distributed::index_space::non_local);

            // Build coarse non-local matrix using generate_coarse
            auto non_local_csr =
                gko::as<const csr_type>(matrix->get_off_diag_matrix());
            auto result_non_local_csr = generate_coarse(
                exec, non_local_csr.get(),
                static_cast<IndexType>(std::get<1>(result)->get_size()[0]),
                coarse_rows_,
                static_cast<IndexType>(coarse_imap.get_non_local_size()),
                non_local_map);

            // Create distributed coarse matrix
            auto coarse = share(
                experimental::distributed::
                    Matrix<ValueType, IndexType, global_index_type>::create(
                        exec, comm, std::move(coarse_imap), std::get<1>(result),
                        result_non_local_csr));
            // Create distributed restrict operator
            auto restrict_op = share(
                experimental::distributed::
                    Matrix<ValueType, IndexType, global_index_type>::create(
                        exec, comm,
                        dim<2>(coarse->get_size()[0],
                               gko::as<LinOp>(matrix)->get_size()[0]),
                        std::get<2>(result)));
            // Create distributed prolong operator
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
        run<fst_mtx_type, snd_mtx_type>(this->get_fine_op(), distributed_setup);
    } else
#endif  // GINKGO_BUILD_MPI
    {
        auto exec = this->get_executor();
        // Only support csr matrix currently.
        auto unif_op =
            std::dynamic_pointer_cast<const csr_type>(system_matrix_);
        // If system matrix is not csr or need sorting, generate the csr.
        if (!parameters_.skip_sorting || !unif_op) {
            unif_op = convert_to_with_sorting<csr_type>(
                exec, system_matrix_, parameters_.skip_sorting);
            // keep the same precision data in fine_op
            this->set_fine_op(unif_op);
        }
        auto result = this->generate_local(unif_op);
        this->set_multigrid_level(std::get<0>(result), std::get<1>(result),
                                  std::get<2>(result));
    }
}


#define GKO_DECLARE_UNIFORM_COARSENING(_vtype, _itype) \
    class UniformCoarsening<_vtype, _itype>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_UNIFORM_COARSENING);


}  // namespace multigrid
}  // namespace gko
