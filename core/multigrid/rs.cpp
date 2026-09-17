// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/multigrid/rs.hpp"

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
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/matrix/row_gatherer.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>

#include "core/base/dispatch_helper.hpp"
#include "core/base/utils.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/config/config_helper.hpp"
#include "core/distributed/helpers.hpp"
#include "core/distributed/index_map_kernels.hpp"
#include "core/multigrid/pgm_kernels.hpp"
#include "core/multigrid/rs_helpers.hpp"
#include "core/multigrid/rs_kernels.hpp"


namespace gko {
namespace multigrid {
namespace rs {
namespace {


GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(fill_seq_array, components::fill_seq_array);

GKO_REGISTER_OPERATION(check_m_matrix, rs::check_m_matrix);
GKO_REGISTER_OPERATION(compute_soc_and_run_rs, rs::compute_soc_and_run_rs);
GKO_REGISTER_OPERATION(mark_forced_c_points, rs::mark_forced_c_points);
GKO_REGISTER_OPERATION(fill_coarse_and_compute_prolong_row_ptrs,
                       rs::fill_coarse_and_compute_prolong_row_ptrs);
GKO_REGISTER_OPERATION(compute_interpolation, rs::compute_interpolation);
// reuse Pgm's kernel
GKO_REGISTER_OPERATION(gather_index, pgm::gather_index);


}  // anonymous namespace
}  // namespace rs
namespace index_map {
namespace {


GKO_REGISTER_OPERATION(map_to_global, index_map::map_to_global);


}
}  // namespace index_map


template <typename ValueType, typename IndexType>
std::tuple<std::shared_ptr<LinOp>, std::shared_ptr<LinOp>,
           std::shared_ptr<LinOp>>
Rs<ValueType, IndexType>::generate_local(
    std::shared_ptr<const matrix::Csr<ValueType, IndexType>> local_matrix,
    const matrix::Csr<ValueType, IndexType>* off_diag_matrix,
    size_type num_forced_c_points, const IndexType* forced_c_points)
{
    using csr_type = matrix::Csr<ValueType, IndexType>;

    auto exec = this->get_executor();
    const auto* rs_op = local_matrix.get();
    const auto fine_dim = rs_op->get_size()[0];

    array<bool> is_m_matrix_array(exec, 1);
    if (!parameters_.skip_m_matrix_check) {
        // for a distributed matrix this only sees the local block: the
        // off-diagonal block has no diagonal entries of its own, so it cannot
        // be checked by the same kernel
        exec->run(rs::make_check_m_matrix(rs_op->get_const_device_view(),
                                          is_m_matrix_array));
        if (!exec->copy_val_to_host(is_m_matrix_array.get_const_data())) {
            GKO_NOT_SUPPORTED(
                "RS coarsening requires an M-matrix (strictly positive "
                "diagonal, "
                "non-positive off-diagonals).");
        }
    }

    // define arrays
    array<bool> is_strong(exec, rs_op->get_num_stored_elements());
    array<IndexType> lambda(exec, fine_dim);
    array<IndexType> cf_marker(exec, fine_dim);
    IndexType coarse_dim{};
    // only a distributed matrix has an off-diagonal block; the empty view
    // makes the kernel skip the remote couplings
    const auto off_diag_view =
        off_diag_matrix ? off_diag_matrix->get_const_device_view()
                        : rs::no_off_diag_view<ValueType, IndexType>();
    // build Strength-of-Connection (SOC) mask, 1 byte per NNZ of the system
    // matrix, compute lambda, perform greedy RS C/F splitting:
    // 0 = undecided, 1 = C, -1 = F,
    // then extract coarse dims
    exec->run(rs::make_compute_soc_and_run_rs(
        rs_op->get_const_device_view(), off_diag_view,
        parameters_.strength_threshold, is_strong, lambda, cf_marker,
        coarse_dim));

    if (num_forced_c_points > 0) {
        // promote the rows other ranks couple to, so that their prolongation
        // rows are unit vectors and P stays block-diagonal
        exec->run(rs::make_mark_forced_c_points(
            num_forced_c_points, forced_c_points, cf_marker, coarse_dim));
    }

    const size_type coarse_dim_size = static_cast<size_type>(coarse_dim);

    // fill in coarse_rows and fine_to_coarse, build prolongation using
    // interpolation
    array<IndexType> coarse_rows(exec, coarse_dim_size);
    fine_to_coarse_ = array<IndexType>(exec, fine_dim);
    array<IndexType> prolong_row_ptrs(exec, fine_dim + 1);
    exec->run(rs::make_fill_coarse_and_compute_prolong_row_ptrs(
        cf_marker, coarse_rows, fine_to_coarse_, rs_op->get_const_device_view(),
        is_strong, prolong_row_ptrs));

    IndexType prolong_nnz =
        exec->copy_val_to_host(prolong_row_ptrs.get_const_data() + fine_dim);

    auto prolong_op = share(csr_type::create(
        exec, gko::dim<2>{fine_dim, coarse_dim_size},
        static_cast<size_type>(prolong_nnz), rs_op->get_strategy()));

    exec->copy_from(exec, fine_dim + 1, prolong_row_ptrs.get_const_data(),
                    prolong_op->get_row_ptrs());

    exec->run(rs::make_compute_interpolation(
        rs_op->get_const_device_view(), is_strong.get_const_data(), cf_marker,
        fine_to_coarse_.get_const_data(), prolong_op->get_device_view()));

    // build restriction as R = P^T
    auto restrict_op = share(as<csr_type>(prolong_op->transpose()));

    // coarse matrix (Ac = R  A  P)
    auto coarse_matrix = share(
        csr_type::create(exec, gko::dim<2>{coarse_dim_size, coarse_dim_size}));
    coarse_matrix->set_strategy(rs_op->get_strategy());

    auto tmp = csr_type::create(exec, gko::dim<2>{fine_dim, coarse_dim_size});
    tmp->set_strategy(rs_op->get_strategy());

    rs_op->apply(prolong_op, tmp);
    restrict_op->apply(tmp, coarse_matrix);

    return std::make_tuple(std::shared_ptr<LinOp>(prolong_op),
                           std::shared_ptr<LinOp>(coarse_matrix),
                           std::shared_ptr<LinOp>(restrict_op));
}


#if GINKGO_BUILD_MPI


template <typename ValueType, typename IndexType>
template <typename GlobalIndexType>
array<GlobalIndexType>
Rs<ValueType, IndexType>::communicate_off_diag_coarse_idxs(
    std::shared_ptr<const experimental::distributed::Matrix<
        ValueType, IndexType, GlobalIndexType>>
        matrix,
    std::shared_ptr<
        experimental::distributed::Partition<IndexType, GlobalIndexType>>
        coarse_partition,
    const array<IndexType>& local_fine_to_coarse)
{
    auto exec = matrix->get_executor();
    const auto comm = matrix->get_communicator();
    auto coll_comm = matrix->row_gatherer_->get_collective_communicator();
    auto row_gatherer = matrix->row_gatherer_;

    // every send index is a forced C-point, so its fine_to_coarse entry is a
    // valid coarse index rather than the -1 an F-point would carry
    array<IndexType> send_coarse(exec, coll_comm->get_send_size());
    exec->run(rs::make_gather_index(
        send_coarse.get_size(), local_fine_to_coarse.get_const_data(),
        row_gatherer->get_const_send_idxs(), send_coarse.get_data()));

    // There is no index map on the coarse level yet, so map the local indices
    // to global indices on the coarse level manually
    array<GlobalIndexType> send_global_coarse(exec, send_coarse.get_size());
    exec->run(index_map::make_map_to_global(
        to_device_const(coarse_partition.get()),
        device_segmented_array<const GlobalIndexType>{}, comm.rank(),
        send_coarse, experimental::distributed::index_space::local,
        send_global_coarse));

    return gko::detail::exchange_with_neighbors(exec, comm, coll_comm.get(),
                                                send_global_coarse);
}


#endif


template <typename ValueType, typename IndexType>
void Rs<ValueType, IndexType>::generate()
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
            auto diag_csr = std::dynamic_pointer_cast<const csr_type>(
                matrix->get_diag_matrix());
            auto off_diag_csr = std::dynamic_pointer_cast<const csr_type>(
                matrix->get_off_diag_matrix());
            // If system matrix is not csr or need sorting, generate the
            // csr.
            if (!parameters_.skip_sorting || !diag_csr || !off_diag_csr) {
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
            using dist_mtx_type =
                experimental::distributed::Matrix<ValueType, IndexType,
                                                  global_index_type>;

            auto exec = gko::as<LinOp>(matrix)->get_executor();
            auto comm =
                gko::as<experimental::distributed::DistributedBase>(matrix)
                    ->get_communicator();
            auto local_op = gko::as<const csr_type>(matrix->get_diag_matrix());
            auto off_diag_op =
                gko::as<const csr_type>(matrix->get_off_diag_matrix());

            // Coarsen the local block. The off-diagonal block only enters the
            // strength threshold; the C/F splitting itself is process-local.
            // Every local row a neighbor couples to is forced into the coarse
            // set, which makes its prolongation row a unit vector and keeps P
            // block-diagonal - so P needs no off-diagonal block, and the whole
            // halo information is one coarse index per off-diag column.
            auto row_gatherer = matrix->row_gatherer_;
            auto result = this->generate_local(
                local_op, off_diag_op.get(), row_gatherer->get_num_send_idxs(),
                row_gatherer->get_const_send_idxs());

            // create the coarse partition
            // the coarse partition will have only one range per part
            // and only one part per rank.
            // The global indices are ordered block-wise by rank, i.e. rank
            // 0 owns [0, ..., N_1), rank 1 [N_1, ..., N_2), ...
            const auto coarse_local_size = std::get<1>(result)->get_size()[0];
            auto coarse_partition = gko::share(
                experimental::distributed::build_partition_from_local_size<
                    IndexType, global_index_type>(
                    exec, comm, static_cast<int64>(coarse_local_size)));

            // get the coarse global index of every off-diag column
            auto off_diag_coarse_idxs = communicate_off_diag_coarse_idxs(
                matrix, coarse_partition, fine_to_coarse_);

            // create a coarse index map based on the connections given by the
            // off-diag coarse indices
            auto coarse_imap =
                experimental::distributed::index_map<IndexType,
                                                     global_index_type>(
                    exec, coarse_partition, comm.rank(), off_diag_coarse_idxs);

            // a mapping from the fine off-diag indices to the coarse off-diag
            // indices. off_diag_coarse_idxs already maps the fine off-diag
            // indices to coarse global indices, so mapping it with the coarse
            // index map results in the coarse off-diag indices.
            auto off_diag_map = coarse_imap.map_to_local(
                off_diag_coarse_idxs,
                experimental::distributed::index_space::non_local);
            const auto coarse_non_local_size = coarse_imap.get_non_local_size();
            const auto fine_non_local_size = off_diag_op->get_size()[1];

            // The coarse off-diagonal block is R * A_off_diag * P_non_local.
            // Every off-diag column belongs to a forced C-point, so the
            // neighbor's prolongation rows for them are unit vectors and
            // P_non_local degenerates into the 0/1 matrix given by
            // off_diag_map. Both products are process-local, which is what
            // makes an explicit distributed triple product unnecessary.
            std::shared_ptr<csr_type> coarse_off_diag_op;
            if (fine_non_local_size == 0) {
                coarse_off_diag_op = share(csr_type::create(
                    exec,
                    gko::dim<2>{coarse_local_size, coarse_non_local_size}));
            } else {
                auto non_local_prolong = csr_type::create(
                    exec,
                    gko::dim<2>{fine_non_local_size, coarse_non_local_size},
                    fine_non_local_size);
                exec->run(
                    rs::make_fill_seq_array(non_local_prolong->get_row_ptrs(),
                                            fine_non_local_size + 1));
                exec->copy_from(exec, fine_non_local_size,
                                off_diag_map.get_const_data(),
                                non_local_prolong->get_col_idxs());
                exec->run(rs::make_fill_array(non_local_prolong->get_values(),
                                              fine_non_local_size,
                                              one<ValueType>()));

                auto tmp =
                    csr_type::create(exec, gko::dim<2>{local_op->get_size()[0],
                                                       coarse_non_local_size});
                off_diag_op->apply(non_local_prolong, tmp);
                coarse_off_diag_op = share(csr_type::create(
                    exec,
                    gko::dim<2>{coarse_local_size, coarse_non_local_size}));
                gko::as<csr_type>(std::get<2>(result))
                    ->apply(tmp, coarse_off_diag_op);
            }

            // setup the generated linop. The prolongation and restriction have
            // no off-diagonal block, see above.
            auto coarse = share(
                dist_mtx_type::create(exec, comm, std::move(coarse_imap),
                                      std::get<1>(result), coarse_off_diag_op));
            auto restrict_op = share(dist_mtx_type::create(
                exec, comm,
                dim<2>(coarse->get_size()[0],
                       gko::as<LinOp>(matrix)->get_size()[0]),
                std::get<2>(result)));
            auto prolong_op = share(dist_mtx_type::create(
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
        auto rs_op = std::dynamic_pointer_cast<const csr_type>(system_matrix_);
        // If system matrix is not csr or need sorting, generate the csr.
        if (!parameters_.skip_sorting || !rs_op) {
            rs_op = convert_to_with_sorting<csr_type>(exec, system_matrix_,
                                                      parameters_.skip_sorting);
            // keep the same precision data in fine_op
            this->set_fine_op(rs_op);
        }
        auto result = this->generate_local(rs_op);
        this->set_multigrid_level(std::get<0>(result), std::get<1>(result),
                                  std::get<2>(result));
    }
}


template <typename ValueType, typename IndexType>
typename Rs<ValueType, IndexType>::parameters_type
Rs<ValueType, IndexType>::parse(const config::pnode& config,
                                const config::registry& context,
                                const config::type_descriptor& td_for_child)
{
    auto params = Rs<ValueType, IndexType>::build();
    config::config_check_decorator config_check(config);
    if (auto& obj = config_check.get("strength_threshold")) {
        params.with_strength_threshold(config::get_value<double>(obj));
    }
    if (auto& obj = config_check.get("skip_sorting")) {
        params.with_skip_sorting(config::get_value<bool>(obj));
    }
    if (auto& obj = config_check.get("skip_m_matrix_check")) {
        params.with_skip_m_matrix_check(config::get_value<bool>(obj));
    }

    return params;
}


#define GKO_DECLARE_RS(_vtype, _itype) class Rs<_vtype, _itype>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_RS);

}  // namespace multigrid
}  // namespace gko
