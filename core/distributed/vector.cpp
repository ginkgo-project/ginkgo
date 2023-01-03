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

#include <ginkgo/core/distributed/vector.hpp>


#include <ginkgo/core/distributed/partition.hpp>


#include "core/distributed/vector_kernels.hpp"
#include "core/matrix/dense_kernels.hpp"


namespace gko {
namespace experimental {
namespace distributed {
namespace vector {
namespace {


GKO_REGISTER_OPERATION(compute_squared_norm2, dense::compute_squared_norm2);
GKO_REGISTER_OPERATION(compute_sqrt, dense::compute_sqrt);
GKO_REGISTER_OPERATION(outplace_absolute_dense, dense::outplace_absolute_dense);
GKO_REGISTER_OPERATION(build_local, distributed_vector::build_local);


}  // namespace
}  // namespace vector


dim<2> compute_global_size(std::shared_ptr<const Executor> exec,
                           mpi::communicator comm, dim<2> local_size)
{
    size_type num_global_rows = local_size[0];
    comm.all_reduce(std::move(exec), &num_global_rows, 1, MPI_SUM);
    return {num_global_rows, local_size[1]};
}


template <typename ValueType>
void Vector<ValueType>::apply_impl(const LinOp* b, LinOp* x) const
{
    GKO_NOT_SUPPORTED(this);
}


template <typename ValueType>
void Vector<ValueType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                   const LinOp* beta, LinOp* x) const
{
    GKO_NOT_SUPPORTED(this);
}

template <typename ValueType>
Vector<ValueType>::Vector(std::shared_ptr<const Executor> exec,
                          mpi::communicator comm, dim<2> global_size,
                          dim<2> local_size)
    : Vector(exec, comm, global_size, local_size, local_size[1])
{}


template <typename ValueType>
Vector<ValueType>::Vector(std::shared_ptr<const Executor> exec,
                          mpi::communicator comm, dim<2> global_size,
                          dim<2> local_size, size_type stride)
    : EnableDistributedLinOp<Vector<ValueType>>{exec, global_size},
      DistributedBase{comm},
      local_{exec, local_size, stride}
{
    GKO_ASSERT_EQUAL_COLS(global_size, local_size);
}

template <typename ValueType>
Vector<ValueType>::Vector(std::shared_ptr<const Executor> exec,
                          mpi::communicator comm, dim<2> global_size,
                          local_vector_type* local_vector)
    : EnableDistributedLinOp<Vector<ValueType>>{exec, global_size},
      DistributedBase{comm},
      local_{exec}
{
    local_vector->move_to(&local_);
}


template <typename ValueType>
Vector<ValueType>::Vector(std::shared_ptr<const Executor> exec,
                          mpi::communicator comm,
                          local_vector_type* local_vector)
    : EnableDistributedLinOp<Vector<ValueType>>{exec, {}},
      DistributedBase{comm},
      local_{exec}
{
    this->set_size(compute_global_size(exec, comm, local_vector->get_size()));
    local_vector->move_to(&local_);
}


template <typename ValueType>
std::unique_ptr<Vector<ValueType>> Vector<ValueType>::create_with_config_of(
    const Vector* other)
{
    // De-referencing `other` before calling the functions (instead of
    // using operator `->`) is currently required to be compatible with
    // CUDA 10.1.
    // Otherwise, it results in a compile error.
    return (*other).create_with_same_config();
}


template <typename ValueType>
std::unique_ptr<Vector<ValueType>> Vector<ValueType>::create_with_type_of(
    const Vector<ValueType>* other, std::shared_ptr<const Executor> exec)
{
    return (*other).create_with_type_of_impl(exec, {}, {}, 0);
}


template <typename ValueType>
std::unique_ptr<Vector<ValueType>> Vector<ValueType>::create_with_type_of(
    const Vector<ValueType>* other, std::shared_ptr<const Executor> exec,
    const dim<2>& global_size, const dim<2>& local_size, size_type stride)
{
    return (*other).create_with_type_of_impl(exec, global_size, local_size,
                                             stride);
}


template <typename ValueType>
template <typename LocalIndexType, typename GlobalIndexType>
void Vector<ValueType>::read_distributed(
    const device_matrix_data<ValueType, GlobalIndexType>& data,
    const Partition<LocalIndexType, GlobalIndexType>* partition)
{
    auto exec = this->get_executor();
    auto global_cols = data.get_size()[1];
    this->resize(
        dim<2>(partition->get_size(), global_cols),
        dim<2>(partition->get_part_size(this->get_communicator().rank()),
               global_cols));

    auto rank = this->get_communicator().rank();
    local_.fill(zero<ValueType>());
    exec->run(vector::make_build_local(
        data, make_temporary_clone(exec, partition).get(), rank, &local_));
}


template <typename ValueType>
template <typename LocalIndexType, typename GlobalIndexType>
void Vector<ValueType>::read_distributed(
    const matrix_data<ValueType, GlobalIndexType>& data,
    const Partition<LocalIndexType, GlobalIndexType>* partition)

{
    this->read_distributed(
        device_matrix_data<value_type, GlobalIndexType>::create_from_host(
            this->get_executor(), data),
        std::move(partition));
}


template <typename ValueType>
void Vector<ValueType>::fill(const ValueType value)
{
    local_.fill(value);
}


template <typename ValueType>
void Vector<ValueType>::convert_to(
    Vector<next_precision<ValueType>>* result) const
{
    GKO_ASSERT(this->get_communicator().size() ==
               result->get_communicator().size());
    result->set_size(this->get_size());
    this->get_local_vector()->convert_to(&result->local_);
}


template <typename ValueType>
void Vector<ValueType>::move_to(Vector<next_precision<ValueType>>* result)
{
    this->convert_to(result);
}


template <typename ValueType>
std::unique_ptr<typename Vector<ValueType>::absolute_type>
Vector<ValueType>::compute_absolute() const
{
    auto exec = this->get_executor();

    auto result =
        absolute_type::create(exec, this->get_communicator(), this->get_size(),
                              this->get_local_vector()->get_size());

    exec->run(vector::make_outplace_absolute_dense(this->get_local_vector(),
                                                   &result->local_));

    return result;
}


template <typename ValueType>
void Vector<ValueType>::compute_absolute_inplace()
{
    local_.compute_absolute_inplace();
}


template <typename ValueType>
const typename Vector<ValueType>::local_vector_type*
Vector<ValueType>::get_local_vector() const
{
    return &local_;
}


template <typename ValueType>
std::unique_ptr<typename Vector<ValueType>::complex_type>
Vector<ValueType>::make_complex() const
{
    auto result = complex_type::create(
        this->get_executor(), this->get_communicator(), this->get_size(),
        this->get_local_vector()->get_size(),
        this->get_local_vector()->get_stride());
    this->make_complex(result.get());
    return result;
}


template <typename ValueType>
void Vector<ValueType>::make_complex(Vector::complex_type* result) const
{
    this->get_local_vector()->make_complex(&result->local_);
}


template <typename ValueType>
std::unique_ptr<typename Vector<ValueType>::real_type>
Vector<ValueType>::get_real() const
{
    auto result = real_type::create(this->get_executor(),
                                    this->get_communicator(), this->get_size(),
                                    this->get_local_vector()->get_size(),
                                    this->get_local_vector()->get_stride());
    this->get_real(result.get());
    return result;
}


template <typename ValueType>
void Vector<ValueType>::get_real(Vector::real_type* result) const
{
    this->get_local_vector()->get_real(&result->local_);
}


template <typename ValueType>
std::unique_ptr<typename Vector<ValueType>::real_type>
Vector<ValueType>::get_imag() const
{
    auto result = real_type::create(this->get_executor(),
                                    this->get_communicator(), this->get_size(),
                                    this->get_local_vector()->get_size(),
                                    this->get_local_vector()->get_stride());
    this->get_imag(result.get());
    return result;
}


template <typename ValueType>
void Vector<ValueType>::get_imag(Vector::real_type* result) const
{
    this->get_local_vector()->get_imag(&result->local_);
}


template <typename ValueType>
void Vector<ValueType>::scale(const LinOp* alpha)
{
    local_.scale(alpha);
}


template <typename ValueType>
void Vector<ValueType>::inv_scale(const LinOp* alpha)
{
    local_.inv_scale(alpha);
}


template <typename ValueType>
void Vector<ValueType>::add_scaled(const LinOp* alpha, const LinOp* b)
{
    auto dense_b = as<Vector<ValueType>>(b);
    local_.add_scaled(alpha, dense_b->get_local_vector());
}


template <typename ValueType>
void Vector<ValueType>::sub_scaled(const LinOp* alpha, const LinOp* b)
{
    auto dense_b = as<Vector<ValueType>>(b);
    local_.sub_scaled(alpha, dense_b->get_local_vector());
}


template <typename ValueType>
void Vector<ValueType>::compute_dot(const LinOp* b, LinOp* result) const
{
    array<char> tmp{this->get_executor()};
    this->compute_dot(b, result, tmp);
}


template <typename ValueType>
void Vector<ValueType>::compute_dot(const LinOp* b, LinOp* result,
                                    array<char>& tmp) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    auto exec = this->get_executor();
    const auto comm = this->get_communicator();
    auto dense_res =
        make_temporary_clone(exec, as<matrix::Dense<ValueType>>(result));
    this->get_local_vector()->compute_dot(as<Vector>(b)->get_local_vector(),
                                          dense_res.get(), tmp);
    exec->synchronize();
    auto use_host_buffer = exec->get_master() != exec && !mpi::is_gpu_aware();
    if (use_host_buffer) {
        host_reduction_buffer_.init(exec->get_master(), dense_res->get_size());
        host_reduction_buffer_->copy_from(dense_res.get());
        comm.all_reduce(exec->get_master(),
                        host_reduction_buffer_->get_values(),
                        static_cast<int>(this->get_size()[1]), MPI_SUM);
        dense_res->copy_from(host_reduction_buffer_.get());
    } else {
        comm.all_reduce(exec, dense_res->get_values(),
                        static_cast<int>(this->get_size()[1]), MPI_SUM);
    }
}


template <typename ValueType>
void Vector<ValueType>::compute_conj_dot(const LinOp* b, LinOp* result) const
{
    array<char> tmp{this->get_executor()};
    this->compute_conj_dot(b, result, tmp);
}


template <typename ValueType>
void Vector<ValueType>::compute_conj_dot(const LinOp* b, LinOp* result,
                                         array<char>& tmp) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    auto exec = this->get_executor();
    const auto comm = this->get_communicator();
    auto dense_res =
        make_temporary_clone(exec, as<matrix::Dense<ValueType>>(result));
    this->get_local_vector()->compute_conj_dot(
        as<Vector>(b)->get_local_vector(), dense_res.get(), tmp);
    exec->synchronize();
    auto use_host_buffer = exec->get_master() != exec && !mpi::is_gpu_aware();
    if (use_host_buffer) {
        host_reduction_buffer_.init(exec->get_master(), dense_res->get_size());
        host_reduction_buffer_->copy_from(dense_res.get());
        comm.all_reduce(exec->get_master(),
                        host_reduction_buffer_->get_values(),
                        static_cast<int>(this->get_size()[1]), MPI_SUM);
        dense_res->copy_from(host_reduction_buffer_.get());
    } else {
        comm.all_reduce(exec, dense_res->get_values(),
                        static_cast<int>(this->get_size()[1]), MPI_SUM);
    }
}


template <typename ValueType>
void Vector<ValueType>::compute_norm2(LinOp* result) const
{
    array<char> tmp{this->get_executor()};
    this->compute_norm2(result, tmp);
}


template <typename ValueType>
void Vector<ValueType>::compute_norm2(LinOp* result, array<char>& tmp) const
{
    using NormVector = typename local_vector_type::absolute_type;
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    auto exec = this->get_executor();
    const auto comm = this->get_communicator();
    auto dense_res = make_temporary_clone(exec, as<NormVector>(result));
    exec->run(vector::make_compute_squared_norm2(this->get_local_vector(),
                                                 dense_res.get(), tmp));
    exec->synchronize();
    auto use_host_buffer = exec->get_master() != exec && !mpi::is_gpu_aware();
    if (use_host_buffer) {
        host_norm_buffer_.init(exec->get_master(), dense_res->get_size());
        host_norm_buffer_->copy_from(dense_res.get());
        comm.all_reduce(exec->get_master(), host_norm_buffer_->get_values(),
                        static_cast<int>(this->get_size()[1]), MPI_SUM);
        dense_res->copy_from(host_norm_buffer_.get());
    } else {
        comm.all_reduce(exec, dense_res->get_values(),
                        static_cast<int>(this->get_size()[1]), MPI_SUM);
    }
    exec->run(vector::make_compute_sqrt(dense_res.get()));
}


template <typename ValueType>
void Vector<ValueType>::compute_norm1(LinOp* result) const
{
    array<char> tmp{this->get_executor()};
    this->compute_norm1(result, tmp);
}


template <typename ValueType>
void Vector<ValueType>::compute_norm1(LinOp* result, array<char>& tmp) const
{
    using NormVector = typename local_vector_type::absolute_type;
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    auto exec = this->get_executor();
    const auto comm = this->get_communicator();
    auto dense_res = make_temporary_clone(exec, as<NormVector>(result));
    this->get_local_vector()->compute_norm1(dense_res.get());
    exec->synchronize();
    auto use_host_buffer = exec->get_master() != exec && !mpi::is_gpu_aware();
    if (use_host_buffer) {
        host_norm_buffer_.init(exec->get_master(), dense_res->get_size());
        host_norm_buffer_->copy_from(dense_res.get());
        comm.all_reduce(exec->get_master(), host_norm_buffer_->get_values(),
                        static_cast<int>(this->get_size()[1]), MPI_SUM);
        dense_res->copy_from(host_norm_buffer_.get());
    } else {
        comm.all_reduce(exec, dense_res->get_values(),
                        static_cast<int>(this->get_size()[1]), MPI_SUM);
    }
}


template <typename ValueType>
ValueType& Vector<ValueType>::at_local(size_type row, size_type col) noexcept
{
    return local_.at(row, col);
}

template <typename ValueType>
ValueType Vector<ValueType>::at_local(size_type row, size_type col) const
    noexcept
{
    return local_.at(row, col);
}

template <typename ValueType>
ValueType& Vector<ValueType>::at_local(size_type idx) noexcept
{
    return local_.at(idx);
}

template <typename ValueType>
ValueType Vector<ValueType>::at_local(size_type idx) const noexcept
{
    return local_.at(idx);
}


template <typename ValueType>
ValueType* Vector<ValueType>::get_local_values()
{
    return local_.get_values();
}


template <typename ValueType>
const ValueType* Vector<ValueType>::get_const_local_values() const
{
    return local_.get_const_values();
}


template <typename ValueType>
void Vector<ValueType>::resize(dim<2> global_size, dim<2> local_size)
{
    if (this->get_size() != global_size) {
        this->set_size(global_size);
    }
    local_.resize(local_size);
}


template <typename ValueType>
std::unique_ptr<const typename Vector<ValueType>::real_type>
Vector<ValueType>::create_real_view() const
{
    const auto num_global_rows = this->get_size()[0];
    const auto num_cols =
        is_complex<ValueType>() ? 2 * this->get_size()[1] : this->get_size()[1];

    return real_type::create(this->get_executor(), this->get_communicator(),
                             dim<2>{num_global_rows, num_cols},
                             const_cast<typename real_type::local_vector_type*>(
                                 local_.create_real_view().get()));
}


template <typename ValueType>
std::unique_ptr<typename Vector<ValueType>::real_type>
Vector<ValueType>::create_real_view()
{
    const auto num_global_rows = this->get_size()[0];
    const auto num_cols =
        is_complex<ValueType>() ? 2 * this->get_size()[1] : this->get_size()[1];

    return real_type::create(this->get_executor(), this->get_communicator(),
                             dim<2>{num_global_rows, num_cols},
                             local_.create_real_view().get());
}


template <typename ValueType>
std::unique_ptr<Vector<ValueType>> Vector<ValueType>::create_with_same_config()
    const
{
    return Vector::create(
        this->get_executor(), this->get_communicator(), this->get_size(),
        this->get_local_vector()->get_size(), this->get_stride());
}


template <typename ValueType>
std::unique_ptr<Vector<ValueType>> Vector<ValueType>::create_with_type_of_impl(
    std::shared_ptr<const Executor> exec, const dim<2>& global_size,
    const dim<2>& local_size, size_type stride) const
{
    return Vector::create(exec, this->get_communicator(), global_size,
                          local_size, stride);
}


#define GKO_DECLARE_DISTRIBUTED_VECTOR(ValueType) class Vector<ValueType>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DISTRIBUTED_VECTOR);


#define GKO_DECLARE_DISTRIBUTED_VECTOR_READ_DISTRIBUTED(                       \
    ValueType, LocalIndexType, GlobalIndexType)                                \
    void Vector<ValueType>::read_distributed<LocalIndexType, GlobalIndexType>( \
        const device_matrix_data<ValueType, GlobalIndexType>& data,            \
        const Partition<LocalIndexType, GlobalIndexType>* partition);          \
    template void                                                              \
    Vector<ValueType>::read_distributed<LocalIndexType, GlobalIndexType>(      \
        const matrix_data<ValueType, GlobalIndexType>& data,                   \
        const Partition<LocalIndexType, GlobalIndexType>* partition)

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_DISTRIBUTED_VECTOR_READ_DISTRIBUTED);


}  // namespace distributed
}  // namespace experimental
}  // namespace gko
