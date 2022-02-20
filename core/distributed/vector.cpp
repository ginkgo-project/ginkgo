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

#include <ginkgo/core/distributed/vector.hpp>


#include "core/distributed/vector_kernels.hpp"
#include "core/matrix/dense_kernels.hpp"


namespace gko {
namespace distributed {
namespace vector {


GKO_REGISTER_OPERATION(compute_norm2_sqr, dense::compute_norm2_sqr);
GKO_REGISTER_OPERATION(compute_sqrt, dense::compute_sqrt);
GKO_REGISTER_OPERATION(outplace_absolute_dense, dense::outplace_absolute_dense);
GKO_REGISTER_OPERATION(build_local, distributed_vector::build_local);


}  // namespace vector


template <typename ValueType, typename LocalIndexType>
void Vector<ValueType, LocalIndexType>::apply_impl(const LinOp* b,
                                                   LinOp* x) const
{
    GKO_NOT_SUPPORTED(this);
}


template <typename ValueType, typename LocalIndexType>
void Vector<ValueType, LocalIndexType>::apply_impl(const LinOp* alpha,
                                                   const LinOp* b,
                                                   const LinOp* beta,
                                                   LinOp* x) const
{
    GKO_NOT_SUPPORTED(this);
}


template <typename ValueType, typename LocalIndexType>
void Vector<ValueType, LocalIndexType>::fill(const ValueType value)
{
    this->get_local()->fill(value);
}


template <typename ValueType, typename LocalIndexType>
void Vector<ValueType, LocalIndexType>::convert_to(
    Vector<next_precision<ValueType>, LocalIndexType>* result) const
{
    result->set_size(this->get_size());
    this->get_local()->convert_to(result->get_local());
}


template <typename ValueType, typename LocalIndexType>
void Vector<ValueType, LocalIndexType>::move_to(
    Vector<next_precision<ValueType>, LocalIndexType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename LocalIndexType>
std::unique_ptr<typename Vector<ValueType, LocalIndexType>::absolute_type>
Vector<ValueType, LocalIndexType>::compute_absolute() const
{
    auto exec = this->get_executor();

    auto result = absolute_type::create(exec, this->get_communicator(),
                                        this->get_partition(), this->get_size(),
                                        this->get_local()->get_size());

    exec->run(vector::make_outplace_absolute_dense(this->get_local(),
                                                   result->get_local()));

    return result;
}


template <typename ValueType, typename LocalIndexType>
void Vector<ValueType, LocalIndexType>::compute_absolute_inplace()
{
    this->get_local()->compute_absolute_inplace();
}


template <typename ValueType, typename LocalIndexType>
const typename Vector<ValueType, LocalIndexType>::local_mtx_type*
Vector<ValueType, LocalIndexType>::get_local() const
{
    return &local_;
}


template <typename ValueType, typename LocalIndexType>
typename Vector<ValueType, LocalIndexType>::local_mtx_type*
Vector<ValueType, LocalIndexType>::get_local()
{
    return &local_;
}


template <typename ValueType, typename LocalIndexType>
Vector<ValueType, LocalIndexType>::Vector(
    std::shared_ptr<const Executor> exec,
    std::shared_ptr<mpi::communicator> comm,
    std::shared_ptr<const Partition<LocalIndexType>> partition,
    dim<2> global_size, dim<2> local_size, size_type stride)
    : EnableLinOp<Vector<ValueType, LocalIndexType>>{exec, global_size},
      DistributedBase{comm},
      partition_{std::move(partition)},
      local_{exec, local_size, stride}
{}


template <typename ValueType, typename LocalIndexType>
Vector<ValueType, LocalIndexType>::Vector(
    std::shared_ptr<const Executor> exec,
    std::shared_ptr<mpi::communicator> comm,
    std::shared_ptr<const Partition<LocalIndexType>> partition,
    dim<2> global_size, dim<2> local_size)
    : Vector{std::move(exec), std::move(comm), std::move(partition),
             global_size,     local_size,      local_size[1]}
{}


template <typename ValueType, typename LocalIndexType>
void Vector<ValueType, LocalIndexType>::read_distributed(
    const matrix_data<ValueType, global_index_type>& data,
    std::shared_ptr<const Partition<LocalIndexType>> partition)
{
    this->partition_ = partition;
    auto exec = this->get_executor();
    auto rank = this->get_communicator()->rank();
    auto global_rows = static_cast<size_type>(partition->get_size());
    auto local_rows = static_cast<size_type>(partition->get_part_size(rank));
    Array<matrix_data_entry<ValueType, global_index_type>> global_data{
        exec, data.nonzeros.begin(), data.nonzeros.end()};
    Array<matrix_data_entry<ValueType, LocalIndexType>> local_data{exec};
    exec->run(vector::make_build_local(global_data, partition.get(), rank,
                                       local_data, ValueType{}));
    dim<2> local_size{local_rows, data.size[1]};
    dim<2> global_size{global_rows, data.size[1]};
    this->get_local()->read(local_data, local_size);
    this->set_size(global_size);
}


template <typename ValueType, typename LocalIndexType>
void Vector<ValueType, LocalIndexType>::scale(const LinOp* alpha)
{
    this->get_local()->scale(alpha);
}


template <typename ValueType, typename LocalIndexType>
void Vector<ValueType, LocalIndexType>::add_scaled(const LinOp* alpha,
                                                   const LinOp* b)
{
    auto dense_b = as<Vector<ValueType, LocalIndexType>>(b);
    this->get_local()->add_scaled(alpha, dense_b->get_local());
}


template <typename ValueType, typename LocalIndexType>
void Vector<ValueType, LocalIndexType>::compute_dot(const LinOp* b,
                                                    LinOp* result) const
{
    auto exec = this->get_executor();
    const auto comm = this->get_communicator();
    auto dense_res =
        make_temporary_clone(exec, as<matrix::Dense<ValueType>>(result));
    this->get_local()->compute_dot(as<Vector>(b)->get_local(), dense_res.get());
    exec->synchronize();
    auto use_host_buffer =
        exec->get_master() != exec || !gko::mpi::is_gpu_aware();
    if (use_host_buffer) {
        auto dense_res_host =
            make_temporary_clone(exec->get_master(), dense_res.get());
        comm->all_reduce(dense_res_host->get_values(),
                         static_cast<int>(this->get_size()[1]), MPI_SUM);
    } else {
        comm->all_reduce(dense_res->get_values(),
                         static_cast<int>(this->get_size()[1]), MPI_SUM);
    }
}


template <typename ValueType, typename LocalIndexType>
void Vector<ValueType, LocalIndexType>::compute_conj_dot(const LinOp* b,
                                                         LinOp* result) const
{
    auto exec = this->get_executor();
    const auto comm = this->get_communicator();
    auto dense_res =
        make_temporary_clone(exec, as<matrix::Dense<ValueType>>(result));
    this->get_local()->compute_conj_dot(as<Vector>(b)->get_local(),
                                        dense_res.get());
    exec->synchronize();
    auto use_host_buffer =
        exec->get_master() != exec || !gko::mpi::is_gpu_aware();
    if (use_host_buffer) {
        auto dense_res_host =
            make_temporary_clone(exec->get_master(), dense_res.get());
        comm->all_reduce(dense_res_host->get_values(),
                         static_cast<int>(this->get_size()[1]), MPI_SUM);
    } else {
        comm->all_reduce(dense_res->get_values(),
                         static_cast<int>(this->get_size()[1]), MPI_SUM);
    }
}


template <typename ValueType, typename LocalIndexType>
void Vector<ValueType, LocalIndexType>::compute_norm2(LinOp* result) const
{
    using NormVector = matrix::Dense<remove_complex<ValueType>>;
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    auto exec = this->get_executor();
    const auto comm = this->get_communicator();
    auto dense_res = make_temporary_clone(exec, as<NormVector>(result));
    exec->run(
        vector::make_compute_norm2_sqr(this->get_local(), dense_res.get()));
    exec->synchronize();
    auto use_host_buffer =
        exec->get_master() != exec || !gko::mpi::is_gpu_aware();
    if (use_host_buffer) {
        auto dense_res_host =
            make_temporary_clone(exec->get_master(), dense_res.get());
        comm->all_reduce(dense_res_host->get_values(),
                         static_cast<int>(this->get_size()[1]), MPI_SUM);
    } else {
        comm->all_reduce(dense_res->get_values(),
                         static_cast<int>(this->get_size()[1]), MPI_SUM);
    }
    exec->synchronize();
    exec->run(vector::make_compute_sqrt(dense_res.get()));
    exec->synchronize();
}


template <typename ValueType, typename LocalIndexType>
void Vector<ValueType, LocalIndexType>::validate_data() const
{
    LinOp::validate_data();
    this->get_local()->validate_data();
    const auto exec = this->get_executor();
    const auto comm = this->get_communicator();
    GKO_VALIDATION_CHECK(this->get_local()->get_executor() == exec);
    // check number of rows
    auto num_local_rows_sum = this->get_local()->get_size()[0];
    comm->all_reduce(&num_local_rows_sum, 1, MPI_SUM);
    GKO_VALIDATION_CHECK(num_local_rows_sum == this->get_size()[0]);
    // check number of columns
    size_type num_local_cols = this->get_local()->get_size()[1];
    auto num_local_cols_min = num_local_cols;
    auto num_local_cols_max = num_local_cols;
    comm->all_reduce(&num_local_cols_min, 1, MPI_MIN);
    comm->all_reduce(&num_local_cols_max, 1, MPI_MAX);
    GKO_VALIDATION_CHECK_NAMED(
        "number of columns on different nodes must match",
        num_local_cols_max == num_local_cols_min);
    GKO_VALIDATION_CHECK_NAMED("local and global number of columns must match",
                               num_local_cols_max == this->get_size()[0]);
}


#define GKO_DECLARE_DISTRIBUTED_VECTOR(ValueType, LocalIndexType) \
    class Vector<ValueType, LocalIndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_DISTRIBUTED_VECTOR);


}  // namespace distributed
}  // namespace gko
