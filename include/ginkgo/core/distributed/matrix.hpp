/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#ifndef GKO_PUBLIC_CORE_DISTRIBUTED_MATRIX_HPP_
#define GKO_PUBLIC_CORE_DISTRIBUTED_MATRIX_HPP_


#include <numeric>
#include <unordered_map>
#include <unordered_set>


#include <ginkgo/config.hpp>


#if GINKGO_BUILD_MPI


#include <ginkgo/core/base/cache.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/base/overlap.hpp>
#include <ginkgo/core/distributed/base.hpp>
#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace distributed {


template <typename ValueType = double, typename LocalIndexType = int32>
class Matrix : public EnableLinOp<Matrix<ValueType, LocalIndexType>>,
               public EnableCreateMethod<Matrix<ValueType, LocalIndexType>>,
               public Transposable,
               public DistributedBase,
               public EnableAbsoluteComputation<
                   remove_complex<Matrix<ValueType, LocalIndexType>>> {
    friend class EnableCreateMethod<Matrix>;
    friend class EnablePolymorphicObject<Matrix, LinOp>;

public:
    using value_type = ValueType;
    using index_type = global_index_type;
    using local_index_type = LocalIndexType;
    using absolute_type = remove_complex<Matrix>;

    using GlobalVec = Vector<value_type, LocalIndexType>;
    using LocalVec = gko::matrix::Dense<value_type>;
    using LocalMtx = gko::matrix::Csr<value_type, local_index_type>;
    using LocalAbsMtx =
        gko::matrix::Csr<remove_complex<value_type>, local_index_type>;
    struct serialized_mtx {
        serialized_mtx(std::shared_ptr<const Executor> exec)
            : col_idxs{exec}, row_ptrs{exec}, values{exec}
        {}
        Array<local_index_type> col_idxs{};
        Array<local_index_type> row_ptrs{};
        Array<value_type> values{};
    };

    void convert_to(Matrix<value_type, local_index_type>* result) const override
    {
        result->diag_mtx_->copy_from(this->diag_mtx_.get());
        result->offdiag_mtx_->copy_from(this->offdiag_mtx_.get());
        // FIXME
        if (result->local_mtx_blocks_.size() > 0) {
            for (auto i = 0; i < this->local_mtx_blocks_.size(); ++i) {
                result->local_mtx_blocks_[i]->copy_from(
                    this->local_mtx_blocks_[i].get());
            }
        }
        result->gather_idxs_ = this->gather_idxs_;
        result->partition_ = this->partition_;
        result->send_offsets_ = this->send_offsets_;
        result->recv_offsets_ = this->recv_offsets_;
        result->recv_sizes_ = this->recv_sizes_;
        result->send_sizes_ = this->send_sizes_;
        result->set_size(this->get_size());
    }

    void move_to(Matrix<value_type, local_index_type>* result) override
    {
        EnableLinOp<Matrix>::move_to(result);
    }

    void read_distributed(
        const matrix_data<ValueType, global_index_type>& data,
        std::shared_ptr<const Partition<local_index_type>> partition);

    void read_distributed(
        const Array<matrix_data_entry<ValueType, global_index_type>>& data,
        dim<2> size,
        std::shared_ptr<const Partition<local_index_type>> partition);

    void validate_data() const override;

    void set_send_offsets(const std::vector<comm_index_type>& other)
    {
        this->send_offsets_ = other;
    }

    void set_recv_offsets(const std::vector<comm_index_type>& other)
    {
        this->recv_offsets_ = other;
    }

    void set_send_sizes(const std::vector<comm_index_type>& other)
    {
        this->send_sizes_ = other;
    }

    void set_recv_sizes(const std::vector<comm_index_type>& other)
    {
        this->recv_sizes_ = other;
    }

    void set_gather_idxs(const Array<local_index_type>& other)
    {
        this->gather_idxs_ = other;
    }

    void set_local_to_global_row(const Array<global_index_type>& other)
    {
        this->local_to_global_row = other;
    }

    void set_local_to_global_offdiag_col(const Array<global_index_type>& other)
    {
        this->local_to_global_offdiag_col = other;
    }

    std::unique_ptr<absolute_type> compute_absolute() const override
    {
        auto exec = this->get_executor();
        auto abs_mtx = absolute_type::create(exec, this->get_communicator());
        abs_mtx->set_send_offsets(this->send_offsets_);
        abs_mtx->set_send_sizes(this->send_sizes_);
        abs_mtx->set_recv_offsets(this->recv_offsets_);
        abs_mtx->set_recv_sizes(this->recv_sizes_);
        abs_mtx->set_gather_idxs(this->gather_idxs_);
        abs_mtx->set_local_to_global_row(this->local_to_global_row);
        abs_mtx->set_local_to_global_offdiag_col(
            this->local_to_global_offdiag_col);
        auto l_diag = this->get_local_diag();
        auto l_offdiag = this->get_local_offdiag();
        auto abs_diag = abs_mtx->get_local_diag();
        auto abs_offdiag = abs_mtx->get_local_offdiag();
        abs_diag = l_diag->compute_absolute();
        abs_offdiag = l_offdiag->compute_absolute();
        return abs_mtx;
    }

    void compute_absolute_inplace() override GKO_NOT_IMPLEMENTED;

    std::unique_ptr<LinOp> transpose() const override GKO_NOT_IMPLEMENTED;

    std::unique_ptr<LinOp> conj_transpose() const override GKO_NOT_IMPLEMENTED;

    std::shared_ptr<LocalMtx> get_local_diag() { return diag_mtx_; }

    std::shared_ptr<LocalMtx> get_local_offdiag() { return offdiag_mtx_; }

    std::vector<std::shared_ptr<LocalMtx>> get_local_mtx_blocks() const
    {
        return local_mtx_blocks_;
    }

    std::shared_ptr<const LocalMtx> get_local_diag() const { return diag_mtx_; }

    std::shared_ptr<const LocalMtx> get_local_offdiag() const
    {
        return offdiag_mtx_;
    }

    std::shared_ptr<serialized_mtx> get_serialized_mtx() const
    {
        return serialized_local_mtx_;
    }

    const Partition<local_index_type>* get_partition() const
    {
        return partition_.get();
    }

    std::vector<std::shared_ptr<LocalMtx>> get_block_approx(
        const Overlap<size_type>& block_overlaps,
        const Array<size_type>& block_sizes);

    std::vector<std::shared_ptr<const LocalMtx>> get_block_approx(
        const Overlap<size_type>& block_overlaps,
        const Array<size_type>& block_sizes) const;

protected:
    Matrix(std::shared_ptr<const Executor> exec,
           std::shared_ptr<mpi::communicator> comm =
               std::make_shared<mpi::communicator>(MPI_COMM_WORLD));

    void copy_communication_data(
        const Matrix<ValueType, LocalIndexType>* other);

    void update_matrix_blocks();

    void serialize_matrix_blocks();

    mpi::request communicate(const LocalVec* local_b) const;

    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

private:
    std::vector<comm_index_type> send_offsets_;
    std::vector<comm_index_type> send_sizes_;
    std::vector<comm_index_type> recv_offsets_;
    std::vector<comm_index_type> recv_sizes_;
    Array<local_index_type> gather_idxs_;
    Array<global_index_type> local_to_global_row;
    Array<global_index_type> local_to_global_offdiag_col;
    LocalVec one_scalar_;
    mutable DenseCache<value_type> host_send_buffer_;
    mutable DenseCache<value_type> host_recv_buffer_;
    mutable DenseCache<value_type> send_buffer_;
    mutable DenseCache<value_type> recv_buffer_;
    std::shared_ptr<LocalMtx> diag_mtx_;
    std::shared_ptr<LocalMtx> offdiag_mtx_;
    std::vector<std::shared_ptr<LocalMtx>> local_mtx_blocks_;
    std::shared_ptr<serialized_mtx> serialized_local_mtx_;
    std::shared_ptr<const Partition<local_index_type>> partition_;
};


}  // namespace distributed
}  // namespace gko


#endif
#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_MATRIX_HPP_
