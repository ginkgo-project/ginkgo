#include <numeric>
#include <unordered_map>
#include <unordered_set>


#include <ginkgo/core/base/mpi_executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include "ginkgo/core/base/lin_op.hpp"


namespace gko {
namespace matrix {

template <typename ValueType, typename IndexType>
class DistributedMatrix
    : public EnableLinOp<DistributedMatrix<ValueType, IndexType>>,
      public EnableCreateMethod<DistributedMatrix<ValueType, IndexType>>,
      public ReadableFromMatrixData<ValueType, IndexType> {
public:
    void read(const matrix_data<ValueType, IndexType> &data)
    {
        IndexType local_begin;
        IndexType local_end;
        auto local_size = local_end - local_begin;
        auto global_size = this->get_size()[1];
        matrix_data<ValueType, IndexType> diag_data(
            dim<2>{local_size, local_size});
        matrix_data<ValueType, IndexType> offdiag_data(
            dim<2>{local_size, global_size});
        std::unordered_set<IndexType> offdiag_col_set;
        auto is_local = [&](IndexType i) {
            return i >= local_begin && i < local_end;
        };
        auto map_to_local = [&](IndexType i) { return i - local_begin; };
        for (auto entry : data.nonzeros) {
            if (is_local(entry.row)) {
                if (is_local(entry.col)) {
                    // map row + col directly
                    diag_data.nonzeros.emplace(map_to_local(entry.row),
                                               map_to_local(entry.column),
                                               entry.value);
                } else {
                    // map row directly, defer mapping col
                    offdiag_col_set.emplace(entry.col);
                    offdiag_data.nonzeros.emplace(map_to_local(entry.row),
                                                  entry.column, entry.value);
                }
            }
        }
        // collect and sort all off-diagonal column indices
        std::vector<IndexType> offdiag_cols(offdiag_col_set.begin(),
                                            offdiag_col_set.end());
        std::sort(offdiag_cols.begin(), offdiag_cols.end());
        offdiag_col_set.clear();
        // build global-local mapping for off-diagonal indices
        std::unordered_map<IndexType, IndexType> offdiag_col_map;
        for (IndexType i = 0; i < offdiag_cols.size(); ++i) {
            offdiag_col_map[offdiag_cols[i]] = i;
        }
        // compress off-diagonal column indices
        offdiag_data.size[1] = offdiag_cols.size();
        for (auto &entry : offdiag_data.nonzeros) {
            entry.column = offdiag_col_map[entry.column];
        }
        // write data
        using Readable = ReadableFromMatrixData<ValueType, IndexType>;
        as<Readable>(diag_mtx_.get())->read(diag_data);
        as<Readable>(offdiag_mtx_.get())->read(offdiag_data);
    }

    const MpiExecutor *get_mpi_exec() const
    {
        return static_cast<const MpiExecutor *>(this->get_executor().get());
    }

protected:
    using Vec = Dense<ValueType>;
    using Mtx = Csr<ValueType, IndexType>;

    DistributedMatrix(std::shared_ptr<MpiExecutor> exec)
        : EnableLinOp<DistributedMatrix<ValueType, IndexType>>{exec, dim<2>{}},
          row_part_ranges_{exec->get_size() + 1},
          send_offsets_{exec->get_size() + 1},
          recv_offsets_{exec->get_size() + 1},
          gather_idxs_{exec},
          one_scalar_{initialize<Vec>(exec, {one<ValueType>()})},
          send_buffer_{exec},
          recv_buffer_{exec},
          diag_mtx_{Mtx::create(exec)},
          offdiag_mtx_{Mtx::create(exec)}
    {}

    void build_communication(const std::vector<IndexType> &cols)
    {
        const MpiExecutor *mpi_exec = get_mpi_exec();
        auto mpi_size = mpi_exec->get_size();
        auto mpi_rank = mpi_exec->get_rank();
        auto col_begin = cols.begin();
        auto col_cur = col_begin;
        auto col_end = cols.end();
        for (int src_rank = 0; src_rank < mpi_size; ++src_rank) {
            // find the size of src_rank's columns
            auto col_next = std::upper_bound(col_cur, col_end,
                                             row_part_ranges_[src_rank + 1]);
            recv_offsets_[src_rank + 1] = std::distance(col_cur, col_next);
            col_cur = col_next;
        }
        mpi_exec->alltoall(&recv_offsets_[1], &send_offsets_[1], 1);
        std::partial_sum(send_offsets_.begin(), send_offsets_.end(),
                         send_offsets_.begin());
        std::partial_sum(recv_offsets_.begin(), recv_offsets_.end(),
                         recv_offsets_.begin());
        Array<IndexType> host_gather_idxs{mpi_exec->get_master(),
                                          send_offsets_.back()};
        GKO_ASSERT(recv_offsets_.back() == cols.size());
        mpi_exec->alltoallv(cols.data(), host_gather_idxs.get_data(), 1,
                            send_offsets_.data(), recv_offsets_.data());
        gather_idxs_ = std::move(host_gather_idxs);
    }

    std::unique_ptr<Vec> communicate(Vec *local_b)
    {
        auto mpi_exec = get_mpi_exec();
        auto local_exec = mpi_exec->get_local();
        auto num_cols = local_b->get_size()[1];
        auto send_dim = dim<2>{send_offsets_.back(), num_cols};
        auto recv_dim = dim<2>{recv_offsets_.back(), num_cols};
        auto send_size = send_dim[0] * send_dim[1];
        auto recv_size = recv_dim[0] * recv_dim[1];
        send_buffer_.resize_and_reset(send_size);
        recv_buffer_.resize_and_reset(recv_size);
        auto send_view =
            Vec::create(local_exec, send_dim,
                        Array<ValueType>::view(local_exec, send_size,
                                               send_buffer_.get_data()),
                        num_cols);
        local_b->row_permute(&gather_idxs_, send_view);
        auto recv_view =
            Vec::create(local_exec, send_dim,
                        Array<ValueType>::view(local_exec, send_size,
                                               send_buffer_.get_data()),
                        num_cols);
        mpi_exec->alltoallv(send_buffer_->get_const_data(),
                            recv_buffer_->get_data(), num_cols,
                            send_offsets_.data(), recv_offsets_.data());
    }

    void apply_impl(LinOp *b, LinOp *x) const override
    {
        auto local_b = as<Vec>(b)->create_local_view();
        auto local_x = as<Vec>(x)->create_local_view();
        auto num_cols = b->get_size()[1];
        // assert matching local dimensions/partition size
        diag_mtx_->apply(local_b.get(), local_x.get());
        // gather data into send_buffer
        offdiag_mtx_->apply(one_scalar_.get(), recv_buffer_.get(),
                            one_scalar_.get(), local_x);
    }

    void apply_impl(LinOp *alpha, LinOp *b, LinOp *beta,
                    LinOp *x) const override
    {
        auto mpi_exec = get_mpi_exec();
        auto local_b = as<Vec>(b)->create_local_view();
        auto local_x = as<Vec>(x)->create_local_view();
        auto local_alpha = as<Vec>(alpha)->create_local_view();
        auto local_beta = as<Vec>(beta)->create_local_view();
        // assert matching local dimensions/partition size
        diag_mtx_->apply(local_alpha.get(), local_b.get(), local_beta.get(),
                         local_x.get());
        // communicate recv_buffer_
        offdiag_mtx_->apply(one_scalar_.get(), recv_buffer_.get(),
                            local_beta.get(), local_x);
    }

private:
    std::vector<int> row_part_ranges_;
    std::vector<int> send_offsets_;
    std::vector<int> recv_offsets_;
    Array<IndexType> gather_idxs_;
    std::unique_ptr<Vec> one_scalar_;
    mutable Array<ValueType> send_buffer_;
    mutable Array<ValueType> recv_buffer_;
    std::unique_ptr<LinOp> diag_mtx_;
    std::unique_ptr<LinOp> offdiag_mtx_;
};

}  // namespace matrix
}  // namespace gko