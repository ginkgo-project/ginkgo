/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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


#include <ginkgo/config.hpp>


#if GINKGO_BUILD_MPI


#include <ginkgo/core/base/dense_cache.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/distributed/base.hpp>
#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/matrix/csr.hpp>


namespace gko {
namespace distributed {


template <typename ValueType>
class Vector;


template <typename ValueType = double, typename LocalIndexType = int32,
          typename GlobalIndexType = int64>
class Matrix
    : public EnableLinOp<Matrix<ValueType, LocalIndexType, GlobalIndexType>>,
      public EnableCreateMethod<
          Matrix<ValueType, LocalIndexType, GlobalIndexType>>,
      public DistributedBase {
    friend class EnableCreateMethod<
        Matrix<ValueType, LocalIndexType, GlobalIndexType>>;
    friend class EnablePolymorphicObject<
        Matrix<ValueType, LocalIndexType, GlobalIndexType>, LinOp>;

public:
    using value_type = ValueType;
    using index_type = GlobalIndexType;
    using local_index_type = LocalIndexType;
    using global_index_type = GlobalIndexType;
    using global_vector_type = gko::distributed::Vector<ValueType>;
    using local_vector_type = typename global_vector_type::local_vector_type;
    using local_matrix_type = gko::matrix::Csr<value_type, local_index_type>;

    void convert_to(Matrix<value_type, local_index_type, global_index_type>*
                        result) const override;

    void move_to(Matrix<value_type, local_index_type, global_index_type>*
                     result) override;

    void read_distributed(
        const matrix_data<ValueType, global_index_type>& data);

    void read_distributed(
        const device_matrix_data<ValueType, GlobalIndexType>& data);

    std::shared_ptr<local_matrix_type> get_local_diag() { return diag_mtx_; }

    std::shared_ptr<const local_matrix_type> get_const_local_diag() const
    {
        return diag_mtx_;
    }

    std::shared_ptr<local_matrix_type> get_local_offdiag()
    {
        return offdiag_mtx_;
    }

    std::shared_ptr<const local_matrix_type> get_const_local_offdiag() const
    {
        return offdiag_mtx_;
    }

    std::shared_ptr<const Partition<local_index_type, global_index_type>>
    get_partition() const
    {
        return partition_;
    }

protected:
    explicit Matrix(std::shared_ptr<const Executor> exec);

    Matrix(
        std::shared_ptr<const Executor> exec, mpi::communicator comm,
        std::shared_ptr<Partition<LocalIndexType, GlobalIndexType>> partition);

    mpi::request communicate(const local_vector_type* local_b) const;

    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

private:
    std::vector<comm_index_type> send_offsets_;
    std::vector<comm_index_type> send_sizes_;
    std::vector<comm_index_type> recv_offsets_;
    std::vector<comm_index_type> recv_sizes_;
    Array<local_index_type> gather_idxs_;
    Array<global_index_type> local_to_global_inner_;
    Array<global_index_type> local_to_global_ghost_;
    ::gko::detail::DenseCache<value_type> one_scalar_;
    mutable ::gko::detail::DenseCache<value_type> host_send_buffer_;
    mutable ::gko::detail::DenseCache<value_type> host_recv_buffer_;
    mutable ::gko::detail::DenseCache<value_type> send_buffer_;
    mutable ::gko::detail::DenseCache<value_type> recv_buffer_;
    std::shared_ptr<local_matrix_type> diag_mtx_;
    std::shared_ptr<local_matrix_type> offdiag_mtx_;
    std::shared_ptr<const Partition<local_index_type, global_index_type>>
        partition_;
};


}  // namespace distributed
}  // namespace gko


#endif


#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_MATRIX_HPP_
