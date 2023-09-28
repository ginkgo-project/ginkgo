/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <ginkgo/core/multigrid/pgm.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/distributed/base.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/matrix/row_gatherer.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/base/iterator_factory.hpp"
#include "core/base/utils.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/format_conversion_kernels.hpp"
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
    const IndexType num = agg.get_num_elems();
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
    exec->run(pgm::make_fill_array(agg_.get_data(), agg_.get_num_elems(),
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
    exec->copy_from(exec, agg_.get_num_elems(), agg_.get_const_data(),
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


template <typename IndexType>
void communicate(
    std::shared_ptr<const experimental::distributed::MatrixBase<IndexType>>
        matrix,
    const array<IndexType>& local_agg, array<IndexType>& non_local_agg)
{
    auto exec = gko::as<LinOp>(matrix)->get_executor();
    const auto comm =
        gko::as<experimental::distributed::DistributedBase>(matrix)
            ->get_communicator();
    auto send_sizes = matrix->get_send_sizes();
    auto recv_sizes = matrix->get_recv_sizes();
    auto send_offsets = matrix->get_send_offsets();
    auto recv_offsets = matrix->get_recv_offsets();
    auto gather_idxs = matrix->get_gather_idxs();
    auto total_send_size = send_offsets.back();
    auto total_recv_size = recv_offsets.back();

    array<IndexType> send_agg(exec, total_send_size);
    exec->run(pgm::make_gather_index(
        send_agg.get_num_elems(), local_agg.get_const_data(),
        gather_idxs.get_const_data(), send_agg.get_data()));

    auto use_host_buffer = experimental::mpi::requires_host_buffer(exec, comm);
    array<IndexType> host_recv_buffer(exec->get_master());
    array<IndexType> host_send_buffer(exec->get_master());
    if (use_host_buffer) {
        host_recv_buffer.resize_and_reset(total_recv_size);
        host_send_buffer.resize_and_reset(total_send_size);
        exec->get_master()->copy_from(exec, total_send_size,
                                      send_agg.get_data(),
                                      host_send_buffer.get_data());
    }
    auto type = experimental::mpi::type_impl<IndexType>::get_type();

    const auto send_ptr = use_host_buffer ? host_send_buffer.get_const_data()
                                          : send_agg.get_const_data();
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
    return;
}


template <typename IndexType>
struct larger_index {
    using type = IndexType;
};

template <>
struct larger_index<gko::int32> {
    using type = gko::int64;
};


template <typename ValueType, typename IndexType>
void Pgm<ValueType, IndexType>::generate()
{
    using csr_type = matrix::Csr<ValueType, IndexType>;
#if GINKGO_BUILD_MPI
    if (auto matrix = std::dynamic_pointer_cast<
            const experimental::distributed::MatrixBase<IndexType>>(
            system_matrix_)) {
        // only work for the square local matrix
        auto exec = gko::as<LinOp>(matrix)->get_executor();
        auto comm = gko::as<experimental::distributed::DistributedBase>(matrix)
                        ->get_communicator();
        auto num_rank = comm.size();
        // Only support csr matrix currently.
        auto pgm_local_op = std::dynamic_pointer_cast<const csr_type>(
            matrix->get_local_matrix());
        // If system matrix is not csr or need sorting, generate the csr.
        if (!parameters_.skip_sorting || !pgm_local_op) {
            pgm_local_op = convert_to_with_sorting<csr_type>(
                exec, matrix->get_local_matrix(), parameters_.skip_sorting);
        }
        auto result = this->generate_local(pgm_local_op);
        auto non_local_matrix = matrix->get_non_local_matrix();
        auto non_local_size = non_local_matrix->get_size()[1];
        array<IndexType> non_local_agg(exec, non_local_size);
        // prolong, restrict, only needs the local infomation
        // get agg information (prolong_row_gather row idx)
        communicate(matrix, agg_, non_local_agg);
        // generate non_local_col_map
        non_local_agg.set_executor(exec->get_master());
        array<IndexType> non_local_col_map(exec->get_master(), non_local_size);
        array<IndexType> part_id(exec->get_master(), non_local_size);
        array<IndexType> index(exec->get_master(), non_local_size);
        auto recv_offsets = matrix->get_recv_offsets();
        for (int i = 0; i + 1 < recv_offsets.size(); i++) {
            for (auto j = recv_offsets.at(i); j < recv_offsets.at(i + 1); j++) {
                part_id.get_data()[j] = i;
                index.get_data()[j] = j;
            }
        }
        // do it in host currently.
        auto it = detail::make_zip_iterator(
            part_id.get_data(), non_local_agg.get_data(), index.get_data());
        // prepare tuple <part_id, local_agg, index>
        // sort by <part_id, local_agg> or did segment sort
        std::sort(it, it + non_local_size);
        // add additional in tail such that the offset easily handle it.
        array<IndexType> renumber(exec->get_master(), non_local_size + 1);
        renumber.get_data()[0] = 0;
        // renumber (prefix_sum) with not eqaul <part_id, local_agg>
        for (int i = 1; i < non_local_size; i++) {
            if (part_id.get_data()[i] != part_id.get_data()[i - 1] ||
                non_local_agg.get_data()[i] !=
                    non_local_agg.get_data()[i - 1]) {
                renumber.get_data()[i] = renumber.get_data()[i - 1] + 1;
            } else {
                renumber.get_data()[i] = renumber.get_data()[i - 1];
            }
        }
        renumber.get_data()[non_local_size] =
            renumber.get_data()[non_local_size - 1] + 1;
        // create col map
        // for each thread i, col_map[tuple[i].index] = map[i]
        for (int i = 0; i < non_local_size; i++) {
            non_local_col_map.get_data()[index.get_data()[i]] =
                renumber.get_data()[i];
        }
        // get new recv_size and recv_offsets
        std::vector<experimental::distributed::comm_index_type> new_recv_size(
            num_rank);
        std::vector<experimental::distributed::comm_index_type>
            new_recv_offsets(num_rank + 1);
        auto rank = comm.rank();
        for (int i = 0; i < num_rank; i++) {
            new_recv_size.at(i) = renumber.get_data()[recv_offsets.at(i + 1)] -
                                  renumber.get_data()[recv_offsets.at(i)];
            new_recv_offsets.at(i + 1) =
                new_recv_offsets.at(i) + new_recv_size.at(i);
        }
        IndexType non_local_num_agg = new_recv_offsets.back();
        array<IndexType> new_recv_gather_idxs(exec->get_master(),
                                              non_local_num_agg);
        for (int i = 0; i < non_local_size; i++) {
            new_recv_gather_idxs.get_data()[renumber.get_data()[i]] =
                non_local_agg.get_data()[i];
        }
        // build coo from row and col map
        // generate_coarse but the row and col map are different
        // Only support csr matrix currently.
        auto non_local_csr =
            std::dynamic_pointer_cast<const csr_type>(non_local_matrix);
        // If system matrix is not csr or need sorting, generate the csr.
        if (!parameters_.skip_sorting || !non_local_csr) {
            non_local_csr = convert_to_with_sorting<csr_type>(
                exec, non_local_matrix, parameters_.skip_sorting);
        }
        non_local_col_map.set_executor(exec);
        auto result_non_local_csr = generate_coarse(
            exec, non_local_csr.get(),
            static_cast<IndexType>(std::get<1>(result)->get_size()[0]), agg_,
            static_cast<IndexType>(non_local_num_agg), non_local_col_map);
        // use local and non-local to build coarse matrix
        // also restriction and prolongation (Local-only-global matrix)
        int64 coarse_size = std::get<1>(result)->get_size()[0];
        comm.all_reduce(exec->get_master(), &coarse_size, 1, MPI_SUM);
        new_recv_gather_idxs.set_executor(exec);
        auto setup = [&](auto global_index) {
            using global_index_type = decltype(global_index);
            auto fine = share(
                experimental::distributed::
                    Matrix<ValueType, IndexType, global_index_type>::create(
                        exec, comm, pgm_local_op, non_local_csr));
            gko::as<ConvertibleTo<experimental::distributed::Matrix<
                ValueType, IndexType, global_index_type>>>(system_matrix_)
                ->convert_to(fine);
            this->set_fine_op(fine);
            auto coarse = share(
                experimental::distributed::
                    Matrix<ValueType, IndexType, global_index_type>::create(
                        exec, comm, gko::dim<2>(coarse_size, coarse_size),
                        std::get<1>(result), result_non_local_csr,
                        new_recv_size, new_recv_offsets, new_recv_gather_idxs));
            auto restrict_op = share(
                experimental::distributed::
                    Matrix<ValueType, IndexType, global_index_type>::create(
                        exec, comm,
                        dim<2>(coarse_size,
                               gko::as<LinOp>(matrix)->get_size()[0]),
                        std::get<2>(result)));
            auto prolong_op = share(
                experimental::distributed::
                    Matrix<ValueType, IndexType, global_index_type>::create(
                        exec, comm,
                        dim<2>(gko::as<LinOp>(matrix)->get_size()[0],
                               coarse_size),
                        std::get<0>(result)));
            this->set_multigrid_level(prolong_op, coarse, restrict_op);
        };
        if (matrix->is_using_index(sizeof(IndexType))) {
            setup(IndexType{});
        } else if (matrix->is_using_index(
                       sizeof(typename larger_index<IndexType>::type))) {
            setup(typename larger_index<IndexType>::type{});
        } else {
            GKO_NOT_IMPLEMENTED;
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
