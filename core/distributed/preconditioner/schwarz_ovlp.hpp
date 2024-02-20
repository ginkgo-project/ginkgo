// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_DISTRIBUTED_PRECONDITIONER_SCHWARZ_OVLP_HPP_
#define GKO_CORE_DISTRIBUTED_PRECONDITIONER_SCHWARZ_OVLP_HPP_


#include <memory>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace experimental {
namespace distributed {
namespace preconditioner {


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
std::vector<matrix_data_entry<ValueType, GlobalIndexType>> get_recv_rows(
    const Matrix<ValueType, LocalIndexType, GlobalIndexType>* mtx,
    const index_map<LocalIndexType, GlobalIndexType>& imap)
{
    using Csr = matrix::Csr<ValueType, LocalIndexType>;
    auto exec = mtx->get_executor();
    auto host_local = make_temporary_clone(exec->get_master(),
                                           as<Csr>(mtx->get_local_matrix()));
    auto host_non_local = make_temporary_clone(
        exec->get_master(), as<Csr>(mtx->get_non_local_matrix()));

    auto& spcomm = mtx->get_sparse_communicator();
    auto& send_idxs = spcomm.template get_send_idxs<LocalIndexType>();
    auto host_send_idxs = make_temporary_clone(exec->get_master(), &send_idxs);

    using mdl = matrix_data<ValueType, GlobalIndexType>;
    mdl global_md;

    auto& send_offsets = spcomm.get_send_offsets();

    std::vector<comm_index_type> rows_send_sizes(send_offsets.size() - 1);

    // 1. segmented sum over nnz per row in send_idxs
    // 2. copy full rows for each index in send_idxs
    // 3. map rows and columns to global indices
    for (size_t pid = 0; pid < send_offsets.size() - 1; ++pid) {
        auto cur_size = global_md.nonzeros.size();
        for (size_type i = send_offsets[pid]; i < send_offsets[pid + 1]; ++i) {
            auto row = host_send_idxs->get_const_data()[i];

            for (LocalIndexType idx = host_local->get_const_row_ptrs()[row];
                 idx < host_local->get_const_row_ptrs()[row + 1]; ++idx) {
                global_md.nonzeros.emplace_back(
                    imap.get_global(row, index_space::local),
                    imap.get_global(host_local->get_const_col_idxs()[idx],
                                    index_space::local),
                    host_local->get_const_values()[idx]);
            }

            for (LocalIndexType idx = host_non_local->get_const_row_ptrs()[row];
                 idx < host_non_local->get_const_row_ptrs()[row + 1]; ++idx) {
                global_md.nonzeros.emplace_back(
                    imap.get_global(row, index_space::local),
                    imap.get_global(host_non_local->get_const_col_idxs()[idx],
                                    index_space::non_local),
                    host_non_local->get_const_values()[idx]);
            }
        }
        rows_send_sizes[pid] = global_md.nonzeros.size() - cur_size;
    }

    std::vector<comm_index_type> rows_send_offsets(send_offsets.size());
    std::partial_sum(rows_send_sizes.begin(), rows_send_sizes.end(),
                     rows_send_offsets.begin() + 1);

    std::vector<comm_index_type> rows_recv_sizes(
        spcomm.get_recv_offsets().size() - 1);
    std::vector<comm_index_type> rows_recv_offsets(
        spcomm.get_recv_offsets().size());

    auto comm = spcomm.get_communicator();
    GKO_ASSERT_NO_MPI_ERRORS(
        MPI_Neighbor_alltoall(rows_send_sizes.data(), 1, MPI_INT,
                              rows_recv_sizes.data(), 1, MPI_INT, comm.get()));
    std::partial_sum(rows_recv_sizes.begin(), rows_recv_sizes.end(),
                     rows_recv_offsets.begin() + 1);

    MPI_Datatype non_zero_type;
    {
        int blocklengths[] = {2, 1};
        MPI_Aint displs[] = {0, 2 * sizeof(GlobalIndexType)};
        MPI_Datatype types[] = {mpi::type_impl<GlobalIndexType>::get_type(),
                                mpi::type_impl<ValueType>::get_type()};
        MPI_Type_create_struct(2, blocklengths, displs, types, &non_zero_type);
        MPI_Type_commit(&non_zero_type);
    }

    std::vector<typename mdl::nonzero_type> recv_nonzeros(
        rows_recv_offsets.back());
    MPI_Neighbor_alltoallv(global_md.nonzeros.data(), rows_send_sizes.data(),
                           rows_send_offsets.data(), non_zero_type,
                           recv_nonzeros.data(), rows_recv_sizes.data(),
                           rows_recv_offsets.data(), non_zero_type, comm.get());
    MPI_Type_free(&non_zero_type);

    return recv_nonzeros;
}


template <typename LocalIndexType, typename GlobalIndexType, typename ValueType>
std::vector<matrix_data_entry<ValueType, GlobalIndexType>> filter_non_relevant(
    const std::vector<matrix_data_entry<ValueType, GlobalIndexType>>& input,
    const index_map<LocalIndexType, GlobalIndexType>& imap)
{
    std::vector<matrix_data_entry<ValueType, GlobalIndexType>> result;
    std::copy_if(input.begin(), input.end(), std::back_inserter(result),
                 [&](const auto& a) {
                     auto is = index_space::combined;
                     return imap.is_within_index_space(a.row, is) &&
                            imap.is_within_index_space(a.column, is);
                 });
    return result;
}


template <typename LocalIndexType, typename GlobalIndexType, typename ValueType>
matrix_data<ValueType, LocalIndexType> combine_overlap(
    const Matrix<ValueType, LocalIndexType, GlobalIndexType>* mat,
    const std::vector<matrix_data_entry<ValueType, GlobalIndexType>>& recv_rows,
    const index_map<LocalIndexType, GlobalIndexType>& imap)
{
    using md = matrix_data<ValueType, LocalIndexType>;
    md local;
    md non_local;

    as<WritableToMatrixData<ValueType, LocalIndexType>>(mat->get_local_matrix())
        ->write(local);
    as<WritableToMatrixData<ValueType, LocalIndexType>>(
        mat->get_non_local_matrix())
        ->write(non_local);

    for (auto& e : non_local.nonzeros) {
        auto is = index_space::non_local;
        e.column = imap.get_combined_local(e.column, is);
    }

    md local_recv_rows;
    std::transform(recv_rows.begin(), recv_rows.end(),
                   std::back_inserter(local_recv_rows.nonzeros),
                   [&](const auto& e) {
                       auto is = index_space::combined;
                       return matrix_data_entry<ValueType, LocalIndexType>{
                           imap.get_local(e.row, is),
                           imap.get_local(e.column, is), e.value};
                   });

    auto combined_size = imap.get_local_size() + imap.get_non_local_size();
    md combined{dim<2>{combined_size, combined_size}};
    std::copy(local.nonzeros.begin(), local.nonzeros.end(),
              std::back_inserter(combined.nonzeros));
    std::copy(non_local.nonzeros.begin(), non_local.nonzeros.end(),
              std::back_inserter(combined.nonzeros));
    std::copy(local_recv_rows.nonzeros.begin(), local_recv_rows.nonzeros.end(),
              std::back_inserter(combined.nonzeros));
    return combined;
}


template <typename ValueType, typename IndexType>
class OverlappingOperator
    : public EnableDistributedLinOp<OverlappingOperator<ValueType, IndexType>>,
      public DistributedBase {
    friend class EnableDistributedPolymorphicObject<OverlappingOperator, LinOp>;

    using Dense = matrix::Dense<ValueType>;

public:
    using value_type = ValueType;

    template <typename GlobalIndexType>
    static std::unique_ptr<OverlappingOperator> create(
        const Matrix<ValueType, IndexType, GlobalIndexType>* mtx,
        const index_map<IndexType, GlobalIndexType>& imap)
    {
        auto recv_rows = get_recv_rows(mtx, imap);
        recv_rows = filter_non_relevant(recv_rows, imap);
        auto ovlp_md = combine_overlap(mtx, recv_rows, imap);
        ovlp_md.sort_row_major_order();

        using Csr = matrix::Csr<ValueType, IndexType>;
        auto ovlp_mtx = Csr::create(mtx->get_executor());
        ovlp_mtx->read(std::move(ovlp_md));

        return std::unique_ptr<OverlappingOperator>(new OverlappingOperator(
            std::move(ovlp_mtx), mtx->get_sparse_communicator()));
    }

    std::shared_ptr<const LinOp> get_matrix() const { return mtx_; }

protected:
    void apply_impl(const LinOp* b, LinOp* x) const override
    {
        auto dense_b = as<Dense>(b);
        auto copy_b = clone(dense_b);

        auto exec = this->get_executor();
        auto non_local_size =
            static_cast<size_type>(spcomm_.get_recv_offsets().back());
        auto local_size = dense_b->get_size()[0] - non_local_size;
        auto combined_size = dense_b->get_size()[0];
        recv_buffer_.init(exec, dim<2>{non_local_size, dense_b->get_size()[1]});
        recv_buffer_->move_from(copy_b->create_submatrix(
            {local_size, combined_size}, {0, dense_b->get_size()[1]}));

        auto req = spcomm_.communicate(dense_b, send_buffer_, recv_buffer_);
        req.wait();

        mtx_->apply(copy_b, x);
    }
    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override GKO_NOT_IMPLEMENTED;

private:
    OverlappingOperator(std::shared_ptr<const Executor> exec,
                        mpi::communicator comm)
        : EnableDistributedLinOp<OverlappingOperator>(std::move(exec)),
          DistributedBase(std::move(comm))
    {}

    OverlappingOperator(std::shared_ptr<const LinOp> mtx,
                        sparse_communicator spcomm)
        : EnableDistributedLinOp<OverlappingOperator>(mtx->get_executor(),
                                                      mtx->get_size()),
          DistributedBase(spcomm.get_communicator()),
          mtx_(std::move(mtx)),
          spcomm_(std::move(spcomm))
    {}


    void restrict(const Dense* in, Dense* out) {}

    void interpolate(const Dense* in, Dense* out) {}


    std::shared_ptr<const LinOp> mtx_;
    sparse_communicator spcomm_;

    detail::DenseCache<value_type> send_buffer_;
    detail::DenseCache<value_type> recv_buffer_;
};


}  // namespace preconditioner
}  // namespace distributed
}  // namespace experimental
}  // namespace gko

#endif  // GKO_CORE_DISTRIBUTED_PRECONDITIONER_SCHWARZ_OVLP_HPP_
