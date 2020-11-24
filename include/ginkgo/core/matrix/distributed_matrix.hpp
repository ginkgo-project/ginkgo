/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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
    friend class EnableCreateMethod<DistributedMatrix>;
    friend class EnablePolymorphicObject<DistributedMatrix, LinOp>;

public:
    void read(const matrix_data<ValueType, IndexType> &data) override
    {
        auto mpi_exec = this->get_mpi_exec();
        size_type part_size = ceildiv(data.size[0], mpi_exec->get_size());
        for (int i = 0; i < mpi_exec->get_size(); ++i) {
            row_part_ranges_[i] = part_size * i;
        }
        row_part_ranges_.back() = data.size[0];
        IndexType local_begin = part_size * mpi_exec->get_rank();
        IndexType local_end =
            std::min<IndexType>(local_begin + part_size, data.size[0]);
        auto local_size = static_cast<size_type>(local_end - local_begin);
        auto global_size = data.size[1];
        this->set_size(dim<2>{local_size, local_size});
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
                if (is_local(entry.column)) {
                    // map row + col directly
                    diag_data.nonzeros.emplace_back(map_to_local(entry.row),
                                                    map_to_local(entry.column),
                                                    entry.value);
                } else {
                    // map row directly, defer mapping col
                    offdiag_col_set.emplace(entry.column);
                    offdiag_data.nonzeros.emplace_back(
                        map_to_local(entry.row), entry.column, entry.value);
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
        this->build_communication(offdiag_cols, global_size);
    }

    const MpiExecutor *get_mpi_exec() const
    {
        return static_cast<const MpiExecutor *>(this->get_executor().get());
    }

protected:
    using Vec = Dense<ValueType>;
    using Mtx = Csr<ValueType, IndexType>;

    DistributedMatrix(std::shared_ptr<const Executor> exec)
        : EnableLinOp<DistributedMatrix<ValueType, IndexType>>{as<MpiExecutor>(
                                                                   exec),
                                                               dim<2>{}},
          gather_idxs_{get_mpi_exec()->get_local()},
          one_scalar_{
              initialize<Vec>({one<ValueType>()}, get_mpi_exec()->get_local())},
          send_buffer_{get_mpi_exec()->get_local()},
          recv_buffer_{get_mpi_exec()->get_local()},
          diag_mtx_{Mtx::create(get_mpi_exec()->get_local())},
          offdiag_mtx_{Mtx::create(get_mpi_exec()->get_local())}
    {
        auto mpi_size = get_mpi_exec()->get_size();
        row_part_ranges_.resize(mpi_size + 1);
        send_offsets_.resize(mpi_size + 1);
        recv_offsets_.resize(mpi_size + 1);
    }

    void build_communication(const std::vector<IndexType> &cols,
                             size_type global_size)
    {
        const MpiExecutor *mpi_exec = get_mpi_exec();
        auto mpi_size = mpi_exec->get_size();
        auto mpi_rank = mpi_exec->get_rank();

        auto col_begin = cols.begin();
        auto col_cur = col_begin;
        auto col_end = cols.end();
        for (int src_rank = 0; src_rank < mpi_size; ++src_rank) {
            // find iterator one past the last column from this rank
            auto col_next = std::lower_bound(col_cur, col_end,
                                             row_part_ranges_[src_rank + 1]);
            recv_offsets_[src_rank + 1] = std::distance(col_cur, col_next);
            col_cur = col_next;
        }
        mpi_exec->alltoall(&recv_offsets_[1], &send_offsets_[1], 1);
        std::partial_sum(send_offsets_.begin(), send_offsets_.end(),
                         send_offsets_.begin());
        std::partial_sum(recv_offsets_.begin(), recv_offsets_.end(),
                         recv_offsets_.begin());

        Array<IndexType> host_gather_idxs{
            mpi_exec->get_local()->get_master(),
            static_cast<size_type>(send_offsets_.back())};
        GKO_ASSERT(recv_offsets_.back() == cols.size());
        // okay, this is tricky: recv_offsets relates to the
        mpi_exec->alltoallv(cols.data(), host_gather_idxs.get_data(), 1,
                            recv_offsets_.data(), send_offsets_.data());

        size_type part_size = ceildiv(global_size, mpi_exec->get_size());
        IndexType local_begin = part_size * mpi_exec->get_rank();
        IndexType local_end =
            std::min<IndexType>(local_begin + part_size, global_size);
        auto is_local = [&](IndexType i) {
            return i >= local_begin && i < local_end;
        };
        auto map_to_local = [&](IndexType i) { return i - local_begin; };
        for (size_type i = 0; i < send_offsets_.back(); ++i) {
            auto &col = host_gather_idxs.get_data()[i];
            GKO_ASSERT(is_local(col));
            col = map_to_local(col);
        }
        gather_idxs_ = std::move(host_gather_idxs);
    }

    std::unique_ptr<Vec> communicate(const Vec *local_b) const
    {
        auto mpi_exec = this->get_mpi_exec();
        auto local_exec = mpi_exec->get_local();
        auto num_cols = local_b->get_size()[1];
        auto send_dim =
            dim<2>{static_cast<size_type>(send_offsets_.back()), num_cols};
        auto recv_dim =
            dim<2>{static_cast<size_type>(recv_offsets_.back()), num_cols};
        auto send_size = send_dim[0] * send_dim[1];
        auto recv_size = recv_dim[0] * recv_dim[1];
        send_buffer_.resize_and_reset(send_size);
        recv_buffer_.resize_and_reset(recv_size);
        auto send_view =
            Vec::create(local_exec, send_dim,
                        Array<ValueType>::view(local_exec, send_size,
                                               send_buffer_.get_data()),
                        num_cols);
        local_b->row_gather(&gather_idxs_, send_view.get());
        auto recv_view =
            Vec::create(local_exec, recv_dim,
                        Array<ValueType>::view(local_exec, recv_size,
                                               recv_buffer_.get_data()),
                        num_cols);
        mpi_exec->alltoallv(send_buffer_.get_const_data(),
                            recv_buffer_.get_data(), num_cols,
                            send_offsets_.data(), recv_offsets_.data());
        return recv_view;
    }

    void apply_impl(const LinOp *b, LinOp *x) const override
    {
        auto local_b = as<Vec>(b)->create_local_view();
        auto local_x = as<Vec>(x)->create_local_view();
        // assert matching local dimensions/partition size
        diag_mtx_->apply(local_b.get(), local_x.get());
        // gather data into send_buffer
        auto recv_view = this->communicate(local_b.get());
        offdiag_mtx_->apply(one_scalar_.get(), recv_view.get(),
                            one_scalar_.get(), local_x.get());
    }

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
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
        // gather data into send_buffer
        auto recv_view = this->communicate(local_b.get());
        offdiag_mtx_->apply(local_alpha.get(), recv_view.get(),
                            one_scalar_.get(), local_x.get());
    }

private:
    std::vector<int> row_part_ranges_;
    std::vector<int> send_offsets_;
    std::vector<int> recv_offsets_;
    Array<IndexType> gather_idxs_;
    std::shared_ptr<Vec> one_scalar_;
    mutable Array<ValueType> send_buffer_;
    mutable Array<ValueType> recv_buffer_;
    std::shared_ptr<LinOp> diag_mtx_;
    std::shared_ptr<LinOp> offdiag_mtx_;
};

}  // namespace matrix
}  // namespace gko