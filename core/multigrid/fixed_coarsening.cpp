// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
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

#include "core/base/utils.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/matrix/csr_builder.hpp"


namespace gko {
namespace multigrid {
namespace fixed_coarsening {
namespace {


GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(fill_seq_array, components::fill_seq_array);


}  // anonymous namespace
}  // namespace fixed_coarsening

template <typename ValueType, typename IndexType>
std::tuple<std::shared_ptr<LinOp>, std::shared_ptr<LinOp>,
           std::shared_ptr<LinOp>>
FixedCoarsening<ValueType, IndexType>::generate_local(
    std::shared_ptr<const matrix::Csr<ValueType, IndexType>>
        fixed_coarsening_op)
{
    using csr_type = matrix::Csr<ValueType, IndexType>;
    auto exec = this->get_executor();

    GKO_ASSERT(parameters_.coarse_rows.get_data() != nullptr);
    GKO_ASSERT(parameters_.coarse_rows.get_size() > 0);
    size_type coarse_dim = parameters_.coarse_rows.get_size();

    auto fine_dim = system_matrix_->get_size()[0];
    auto restrict_op = share(
        csr_type::create(exec, gko::dim<2>{coarse_dim, fine_dim}, coarse_dim));
    exec->copy_from(parameters_.coarse_rows.get_executor(), coarse_dim,
                    parameters_.coarse_rows.get_const_data(),
                    restrict_op->get_col_idxs());
    exec->run(fixed_coarsening::make_fill_array(restrict_op->get_values(),
                                                coarse_dim, one<ValueType>()));
    exec->run(fixed_coarsening::make_fill_seq_array(restrict_op->get_row_ptrs(),
                                                    coarse_dim + 1));

    auto prolong_op = gko::as<csr_type>(share(restrict_op->transpose()));

    // TODO: Can be done with submatrix index_set.
    auto coarse_matrix =
        share(csr_type::create(exec, gko::dim<2>{coarse_dim, coarse_dim}));
    // coarse_matrix->set_strategy(fixed_coarsening_op->get_strategy());
    auto tmp = csr_type::create(exec, gko::dim<2>{fine_dim, coarse_dim});
    // tmp->set_strategy(fixed_coarsening_op->get_strategy());
    fixed_coarsening_op->apply(prolong_op, tmp);
    restrict_op->apply(tmp, coarse_matrix);

    return {prolong_op, coarse_matrix, restrict_op};
}


template <typename ValueType, typename IndexType>
void FixedCoarsening<ValueType, IndexType>::generate()
{
    using csr_type = matrix::Csr<ValueType, IndexType>;
    using sparsity_type = matrix::SparsityCsr<ValueType, IndexType>;
    using real_type = remove_complex<ValueType>;
    auto exec = this->get_executor();
    const auto num_rows = this->system_matrix_->get_size()[0];
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
            GKO_NOT_IMPLEMENTED;
            // run<ConvertibleTo, fst_mtx_type, snd_mtx_type>(system_matrix_,
            //                                                convert_fine_op);
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
            // 0 owns [0, ..., N_1), rank 1 [N_1, ..., N_2), ...
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
            // non-local indices.
            // non_local_agg already maps the fine non-local indices to
            // coarse global indices, so mapping it with the coarse index
            // map results in the coarse non-local indices.
            // TODO: do we need move?
            // non_local_map_ = std::move(coarse_imap.map_to_local(
            //     non_local_agg,
            //     experimental::distributed::index_space::non_local));

            // build csr from row and col map
            // unlike non-distributed version, generate_coarse uses
            // different row and col maps.
            auto non_local_csr =
                as<const csr_type>(matrix->get_non_local_matrix());
            auto result_non_local_csr = generate_coarse(
                exec, non_local_csr.get(),
                static_cast<IndexType>(std::get<1>(result)->get_size()[0]),
                agg_, static_cast<IndexType>(coarse_imap.get_non_local_size()),
                non_local_map_);

            // setup the generated linop.
            auto coarse = share(
                experimental::distributed::
                    Matrix<ValueType, IndexType, global_index_type>::create(
                        exec, comm, std::move(coarse_imap), std::get<1>(result),
                        result_non_local_csr));
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
        run<fst_mtx_type, snd_mtx_type>(this->get_fine_op(), distributed_setup);
    } else
#endif  // GINKGO_BUILD_MPI
    {
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

        auto result = this->generate_local(fixed_coarsening_op);
    this->set_multigrid_level(
                             std::get<0>(result),
                             std::get<1>(result),
                             std::get<2>(result);
    }
}


#define GKO_DECLARE_FIXED_COARSENING(_vtype, _itype) \
    class FixedCoarsening<_vtype, _itype>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_FIXED_COARSENING);


}  // namespace multigrid
}  // namespace gko
