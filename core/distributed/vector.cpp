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

#include <ginkgo/core/distributed/vector.hpp>


#include "core/distributed/vector_kernels.hpp"
#include "core/matrix/dense_kernels.hpp"


namespace gko {
namespace distributed {
namespace vector {
namespace {


GKO_REGISTER_OPERATION(compute_norm2_sqr, dense::compute_norm2_sqr);
GKO_REGISTER_OPERATION(compute_sqrt, dense::compute_sqrt);
GKO_REGISTER_OPERATION(outplace_absolute_dense, dense::outplace_absolute_dense);
GKO_REGISTER_OPERATION(build_local, distributed_vector::build_local);


}  // namespace
}  // namespace vector


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Vector<ValueType, LocalIndexType, GlobalIndexType>::apply_impl(
    const LinOp* b, LinOp* x) const
{
    GKO_NOT_SUPPORTED(this);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Vector<ValueType, LocalIndexType, GlobalIndexType>::apply_impl(
    const LinOp* alpha, const LinOp* b, const LinOp* beta, LinOp* x) const
{
    GKO_NOT_SUPPORTED(this);
}

template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Vector<ValueType, LocalIndexType, GlobalIndexType>::Vector(
    std::shared_ptr<const Executor> exec, mpi::communicator comm,
    std::shared_ptr<const Partition<LocalIndexType, GlobalIndexType>> partition,
    dim<2> global_size, dim<2> local_size)
    : Vector(exec, comm, std::move(partition), global_size, local_size,
             local_size[1])
{}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Vector<ValueType, LocalIndexType, GlobalIndexType>::Vector(
    std::shared_ptr<const Executor> exec)
    : Vector(exec, mpi::communicator(MPI_COMM_NULL), {}, {}, 0)
{}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Vector<ValueType, LocalIndexType, GlobalIndexType>::Vector(
    std::shared_ptr<const Executor> exec, mpi::communicator comm,
    std::shared_ptr<const Partition<LocalIndexType, GlobalIndexType>> partition,
    dim<2> global_size, dim<2> local_size, size_type stride)
    : EnableLinOp<
          Vector<ValueType, LocalIndexType, GlobalIndexType>>{exec,
                                                              global_size},
      DistributedBase{comm},
      partition_{std::move(partition)},
      local_{exec, local_size, stride}
{}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Vector<ValueType, LocalIndexType, GlobalIndexType>::read_distributed(
    const matrix_data<ValueType, GlobalIndexType>& data)
{
    this->read_distributed(
        device_matrix_data<value_type, global_index_type>::create_from_host(
            this->get_executor(), data));
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Vector<ValueType, LocalIndexType, GlobalIndexType>::read_distributed(
    const device_matrix_data<ValueType, GlobalIndexType>& data)
{
    auto exec = this->get_executor();

    auto global_rows = static_cast<size_type>(this->partition_->get_size());
    auto global_cols = data.get_size()[1];
    this->set_size({global_rows, global_cols});

    auto rank = this->get_communicator().rank();
    auto local_rows =
        static_cast<size_type>(this->get_partition()->get_part_size(rank));
    if (this->get_local()->get_size() != dim<2>{local_rows, global_cols}) {
        auto stride = this->get_local()->get_stride() > 0
                          ? this->get_local()->get_stride()
                          : global_cols;
        local_vector_type::create(exec, dim<2>{local_rows, global_cols}, stride)
            ->move_to(this->get_local());
    }
    this->get_local()->fill(zero<ValueType>());
    exec->run(vector::make_build_local(data, this->get_partition().get(), rank,
                                       this->get_local()));
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Vector<ValueType, LocalIndexType, GlobalIndexType>::fill(
    const ValueType value)
{
    this->get_local()->fill(value);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Vector<ValueType, LocalIndexType, GlobalIndexType>::convert_to(
    Vector<next_precision<ValueType>, LocalIndexType, GlobalIndexType>* result)
    const
{
    result->set_size(this->get_size());
    result->set_communicator(this->get_communicator());
    result->partition_ = this->partition_;
    this->get_const_local()->convert_to(result->get_local());
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Vector<ValueType, LocalIndexType, GlobalIndexType>::move_to(
    Vector<next_precision<ValueType>, LocalIndexType, GlobalIndexType>* result)
{
    result->set_size(this->get_size());
    result->set_communicator(this->get_communicator());
    result->partition_ = this->partition_;
    this->get_local()->move_to(result->get_local());
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
std::unique_ptr<
    typename Vector<ValueType, LocalIndexType, GlobalIndexType>::absolute_type>
Vector<ValueType, LocalIndexType, GlobalIndexType>::compute_absolute() const
{
    auto exec = this->get_executor();

    auto result = absolute_type::create(exec, this->get_communicator(),
                                        this->get_partition(), this->get_size(),
                                        this->get_const_local()->get_size());

    exec->run(vector::make_outplace_absolute_dense(this->get_const_local(),
                                                   result->get_local()));

    return result;
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Vector<ValueType, LocalIndexType,
            GlobalIndexType>::compute_absolute_inplace()
{
    this->get_local()->compute_absolute_inplace();
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
const typename Vector<ValueType, LocalIndexType,
                      GlobalIndexType>::local_vector_type*
Vector<ValueType, LocalIndexType, GlobalIndexType>::get_const_local() const
{
    return &local_;
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
typename Vector<ValueType, LocalIndexType, GlobalIndexType>::local_vector_type*
Vector<ValueType, LocalIndexType, GlobalIndexType>::get_local()
{
    return &local_;
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
std::unique_ptr<
    typename Vector<ValueType, LocalIndexType, GlobalIndexType>::complex_type>
Vector<ValueType, LocalIndexType, GlobalIndexType>::make_complex() const
{
    auto result = complex_type::create(
        this->get_executor(), this->get_communicator(), this->get_partition(),
        this->get_size(), this->get_const_local()->get_size(),
        this->get_const_local()->get_stride());
    this->make_complex(result.get());
    return result;
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Vector<ValueType, LocalIndexType, GlobalIndexType>::make_complex(
    Vector::complex_type* result) const
{
    this->get_const_local()->make_complex(result->get_local());
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
std::unique_ptr<
    typename Vector<ValueType, LocalIndexType, GlobalIndexType>::real_type>
Vector<ValueType, LocalIndexType, GlobalIndexType>::get_real() const
{
    auto result = real_type::create(
        this->get_executor(), this->get_communicator(), this->get_partition(),
        this->get_size(), this->get_const_local()->get_size(),
        this->get_const_local()->get_stride());
    this->get_real(result.get());
    return result;
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Vector<ValueType, LocalIndexType, GlobalIndexType>::get_real(
    Vector::real_type* result) const
{
    this->get_const_local()->get_real(result->get_local());
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
std::unique_ptr<
    typename Vector<ValueType, LocalIndexType, GlobalIndexType>::real_type>
Vector<ValueType, LocalIndexType, GlobalIndexType>::get_imag() const
{
    auto result = real_type::create(
        this->get_executor(), this->get_communicator(), this->get_partition(),
        this->get_size(), this->get_const_local()->get_size(),
        this->get_const_local()->get_stride());
    this->get_imag(result.get());
    return result;
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Vector<ValueType, LocalIndexType, GlobalIndexType>::get_imag(
    Vector::real_type* result) const
{
    this->get_const_local()->get_imag(result->get_local());
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Vector<ValueType, LocalIndexType, GlobalIndexType>::scale(
    const LinOp* alpha)
{
    this->get_local()->scale(alpha);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Vector<ValueType, LocalIndexType, GlobalIndexType>::inv_scale(
    const LinOp* alpha)
{
    this->get_local()->inv_scale(alpha);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Vector<ValueType, LocalIndexType, GlobalIndexType>::add_scaled(
    const LinOp* alpha, const LinOp* b)
{
    auto dense_b = as<Vector<ValueType, LocalIndexType, GlobalIndexType>>(b);
    this->get_local()->add_scaled(alpha, dense_b->get_const_local());
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Vector<ValueType, LocalIndexType, GlobalIndexType>::sub_scaled(
    const LinOp* alpha, const LinOp* b)
{
    auto dense_b = as<Vector<ValueType, LocalIndexType, GlobalIndexType>>(b);
    this->get_local()->sub_scaled(alpha, dense_b->get_const_local());
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Vector<ValueType, LocalIndexType, GlobalIndexType>::compute_dot(
    const LinOp* b, LinOp* result) const
{
    auto exec = this->get_executor();
    const auto comm = this->get_communicator();
    auto dense_res =
        make_temporary_clone(exec, as<matrix::Dense<ValueType>>(result));
    this->get_const_local()->compute_dot(as<Vector>(b)->get_const_local(),
                                         dense_res.get());
    exec->synchronize();
    auto use_host_buffer =
        exec->get_master() != exec || !gko::mpi::is_gpu_aware();
    if (use_host_buffer) {
        host_reduction_buffer_.init(exec->get_master(), dense_res->get_size());
        host_reduction_buffer_->copy_from(dense_res.get());
        comm.all_reduce(host_reduction_buffer_->get_values(),
                        static_cast<int>(this->get_size()[1]), MPI_SUM);
        dense_res->copy_from(host_reduction_buffer_.get());
    } else {
        comm.all_reduce(dense_res->get_values(),
                        static_cast<int>(this->get_size()[1]), MPI_SUM);
    }
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Vector<ValueType, LocalIndexType, GlobalIndexType>::compute_conj_dot(
    const LinOp* b, LinOp* result) const
{
    auto exec = this->get_executor();
    const auto comm = this->get_communicator();
    auto dense_res =
        make_temporary_clone(exec, as<matrix::Dense<ValueType>>(result));
    this->get_const_local()->compute_conj_dot(as<Vector>(b)->get_const_local(),
                                              dense_res.get());
    exec->synchronize();
    auto use_host_buffer =
        exec->get_master() != exec || !gko::mpi::is_gpu_aware();
    if (use_host_buffer) {
        host_reduction_buffer_.init(exec->get_master(), dense_res->get_size());
        host_reduction_buffer_->copy_from(dense_res.get());
        comm.all_reduce(host_reduction_buffer_->get_values(),
                        static_cast<int>(this->get_size()[1]), MPI_SUM);
        dense_res->copy_from(host_reduction_buffer_.get());
    } else {
        comm.all_reduce(dense_res->get_values(),
                        static_cast<int>(this->get_size()[1]), MPI_SUM);
    }
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Vector<ValueType, LocalIndexType, GlobalIndexType>::compute_norm2(
    LinOp* result) const
{
    using NormVector = typename local_vector_type::absolute_type;
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    auto exec = this->get_executor();
    const auto comm = this->get_communicator();
    auto dense_res = make_temporary_clone(exec, as<NormVector>(result));
    exec->run(vector::make_compute_norm2_sqr(this->get_const_local(),
                                             dense_res.get()));
    exec->synchronize();
    auto use_host_buffer =
        exec->get_master() != exec || !gko::mpi::is_gpu_aware();
    if (use_host_buffer) {
        host_norm_buffer_.init(exec->get_master(), dense_res->get_size());
        host_norm_buffer_->copy_from(dense_res.get());
        comm.all_reduce(host_norm_buffer_->get_values(),
                        static_cast<int>(this->get_size()[1]), MPI_SUM);
        dense_res->copy_from(host_norm_buffer_.get());
    } else {
        comm.all_reduce(dense_res->get_values(),
                        static_cast<int>(this->get_size()[1]), MPI_SUM);
    }
    exec->run(vector::make_compute_sqrt(dense_res.get()));
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Vector<ValueType, LocalIndexType, GlobalIndexType>::compute_norm1(
    LinOp* result) const
{
    using NormVector = typename local_vector_type::absolute_type;
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    auto exec = this->get_executor();
    const auto comm = this->get_communicator();
    auto dense_res = make_temporary_clone(exec, as<NormVector>(result));
    this->get_const_local()->compute_norm1(dense_res.get());
    exec->synchronize();
    auto use_host_buffer =
        exec->get_master() != exec || !gko::mpi::is_gpu_aware();
    if (use_host_buffer) {
        host_norm_buffer_.init(exec->get_master(), dense_res->get_size());
        host_norm_buffer_->copy_from(dense_res.get());
        comm.all_reduce(host_norm_buffer_->get_values(),
                        static_cast<int>(this->get_size()[1]), MPI_SUM);
        dense_res->copy_from(host_norm_buffer_.get());
    } else {
        comm.all_reduce(dense_res->get_values(),
                        static_cast<int>(this->get_size()[1]), MPI_SUM);
    }
}


#define GKO_DECLARE_DISTRIBUTED_VECTOR(ValueType, LocalIndexType, \
                                       GlobalIndexType)           \
    class Vector<ValueType, LocalIndexType, GlobalIndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_DISTRIBUTED_VECTOR);


}  // namespace distributed
}  // namespace gko
