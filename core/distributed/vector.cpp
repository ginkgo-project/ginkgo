// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/distributed/vector.hpp"

#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/distributed/vector_kernels.hpp"
#include "core/matrix/multivector_kernels.hpp"
#include "core/mpi/mpi_op.hpp"


namespace gko {
namespace experimental {
namespace distributed {
namespace vector {
namespace {


GKO_REGISTER_OPERATION(compute_squared_norm2,
                       multivector::compute_squared_norm2);
GKO_REGISTER_OPERATION(compute_sqrt, multivector::compute_sqrt);
GKO_REGISTER_OPERATION(outplace_absolute_dense,
                       multivector::outplace_absolute_dense);
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
Vector<ValueType>::Vector(std::shared_ptr<const Executor> exec,
                          mpi::communicator comm, dim<2> global_size,
                          dim<2> local_size)
    : Vector(exec, comm, global_size, local_size, local_size[1])
{}


template <typename ValueType>
Vector<ValueType>::Vector(std::shared_ptr<const Executor> exec,
                          mpi::communicator comm, dim<2> global_size,
                          dim<2> local_size, size_type stride)
    : LinOp{exec, global_size},
      DistributedBase{comm},
      local_{exec, local_size, stride}
{
    GKO_ASSERT_EQUAL_COLS(global_size, local_size);
}

template <typename ValueType>
Vector<ValueType>::Vector(std::shared_ptr<const Executor> exec,
                          mpi::communicator comm, dim<2> global_size,
                          std::unique_ptr<local_vector_type> local_vector)
    : LinOp{exec, global_size}, DistributedBase{comm}, local_{exec}
{
    local_vector->move_to(&local_);
}


template <typename ValueType>
Vector<ValueType>::Vector(std::shared_ptr<const Executor> exec,
                          mpi::communicator comm,
                          std::unique_ptr<local_vector_type> local_vector)
    : LinOp{exec, {}}, DistributedBase{comm}, local_{exec}
{
    this->set_size(compute_global_size(exec, comm, local_vector->get_size()));
    local_vector->move_to(&local_);
}

template <typename ValueType>
std::unique_ptr<Vector<ValueType>> Vector<ValueType>::create(
    std::shared_ptr<const Executor> exec, mpi::communicator comm,
    dim<2> global_size, dim<2> local_size, size_type stride)
{
    return std::unique_ptr<Vector>{
        new Vector{exec, comm, global_size, local_size, stride}};
}


template <typename ValueType>
std::unique_ptr<Vector<ValueType>> Vector<ValueType>::create(
    std::shared_ptr<const Executor> exec, mpi::communicator comm,
    dim<2> global_size, dim<2> local_size)
{
    return std::unique_ptr<Vector>{
        new Vector{exec, comm, global_size, local_size}};
}


template <typename ValueType>
std::unique_ptr<Vector<ValueType>> Vector<ValueType>::create(
    std::shared_ptr<const Executor> exec, mpi::communicator comm,
    dim<2> global_size, std::unique_ptr<local_vector_type> local_vector)
{
    return std::unique_ptr<Vector>{
        new Vector{exec, comm, global_size, std::move(local_vector)}};
}


template <typename ValueType>
std::unique_ptr<Vector<ValueType>> Vector<ValueType>::create(
    std::shared_ptr<const Executor> exec, mpi::communicator comm,
    std::unique_ptr<local_vector_type> local_vector)
{
    return std::unique_ptr<Vector>{
        new Vector{exec, comm, std::move(local_vector)}};
}


template <typename ValueType>
std::unique_ptr<const Vector<ValueType>> Vector<ValueType>::create_const(
    std::shared_ptr<const Executor> exec, mpi::communicator comm,
    dim<2> global_size, std::unique_ptr<const local_vector_type> local_vector)
{
    auto non_const_local_vector =
        const_cast<local_vector_type*>(local_vector.release());

    return std::unique_ptr<const Vector>(
        new Vector(std::move(exec), std::move(comm), global_size,
                   std::unique_ptr<local_vector_type>{non_const_local_vector}));
}


template <typename ValueType>
std::unique_ptr<const Vector<ValueType>> Vector<ValueType>::create_const(
    std::shared_ptr<const Executor> exec, mpi::communicator comm,
    std::unique_ptr<const local_vector_type> local_vector)
{
    auto global_size =
        compute_global_size(exec, comm, local_vector->get_size());
    return Vector::create_const(std::move(exec), std::move(comm), global_size,
                                std::move(local_vector));
}


template <typename ValueType>
template <typename LocalIndexType, typename GlobalIndexType>
void Vector<ValueType>::read_distributed_impl(
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
        data, make_temporary_clone(exec, partition).get(), rank,
        local_.get_device_view()));
}


template <typename ValueType>
void Vector<ValueType>::read_distributed(
    const device_matrix_data<ValueType, int64>& data,
    ptr_param<const Partition<int64, int64>> partition)
{
    this->read_distributed_impl(data, partition.get());
}


template <typename ValueType>
void Vector<ValueType>::read_distributed(
    const device_matrix_data<ValueType, int64>& data,
    ptr_param<const Partition<int32, int64>> partition)
{
    this->read_distributed_impl(data, partition.get());
}


template <typename ValueType>
void Vector<ValueType>::read_distributed(
    const device_matrix_data<ValueType, int32>& data,
    ptr_param<const Partition<int32, int32>> partition)
{
    this->read_distributed_impl(data, partition.get());
}


template <typename ValueType>
void Vector<ValueType>::read_distributed(
    const matrix_data<ValueType, int64>& data,
    ptr_param<const Partition<int64, int64>> partition)
{
    this->read_distributed(
        device_matrix_data<value_type, int64>::create_from_host(
            this->get_executor(), data),
        partition);
}


template <typename ValueType>
void Vector<ValueType>::read_distributed(
    const matrix_data<ValueType, int64>& data,
    ptr_param<const Partition<int32, int64>> partition)
{
    this->read_distributed(
        device_matrix_data<value_type, int64>::create_from_host(
            this->get_executor(), data),
        partition);
}


template <typename ValueType>
void Vector<ValueType>::read_distributed(
    const matrix_data<ValueType, int32>& data,
    ptr_param<const Partition<int32, int32>> partition)
{
    this->read_distributed(
        device_matrix_data<value_type, int32>::create_from_host(
            this->get_executor(), data),
        partition);
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


#if GINKGO_ENABLE_HALF || GINKGO_ENABLE_BFLOAT16
template <typename ValueType>
void Vector<ValueType>::convert_to(
    Vector<next_precision<ValueType, 2>>* result) const
{
    GKO_ASSERT(this->get_communicator().size() ==
               result->get_communicator().size());
    result->set_size(this->get_size());
    this->get_local_vector()->convert_to(&result->local_);
}


template <typename ValueType>
void Vector<ValueType>::move_to(Vector<next_precision<ValueType, 2>>* result)
{
    this->convert_to(result);
}
#endif


#if GINKGO_ENABLE_HALF && GINKGO_ENABLE_BFLOAT16
template <typename ValueType>
void Vector<ValueType>::convert_to(
    Vector<next_precision<ValueType, 3>>* result) const
{
    GKO_ASSERT(this->get_communicator().size() ==
               result->get_communicator().size());
    result->set_size(this->get_size());
    this->get_local_vector()->convert_to(&result->local_);
}


template <typename ValueType>
void Vector<ValueType>::move_to(Vector<next_precision<ValueType, 3>>* result)
{
    this->convert_to(result);
}
#endif


template <typename ValueType>
const typename Vector<ValueType>::local_vector_type*
Vector<ValueType>::get_local_vector() const
{
    return &local_;
}


template <typename ValueType>
void Vector<ValueType>::compute_mean(ptr_param<LinOp> result) const
{
    array<char> tmp{this->get_executor()};
    this->compute_mean(result, tmp);
}


template <typename ValueType>
void Vector<ValueType>::compute_mean(ptr_param<LinOp> result,
                                     array<char>& tmp) const
{
    using MeanVector = local_vector_type;
    const auto global_size = this->get_size()[0];
    const auto local_size = this->get_local_vector()->get_size()[0];
    const auto num_vecs = static_cast<int>(this->get_size()[1]);
    GKO_ASSERT_EQUAL_COLS(result, this);
    auto exec = this->get_executor();
    const auto comm = this->get_communicator();
    auto dense_res = make_temporary_clone(exec, as<MeanVector>(result));
    this->get_local_vector()->compute_mean(dense_res.get());

    // scale by its weight ie ratio of local to global size
    auto weight = initialize<matrix::MultiVector<remove_complex<ValueType>>>(
        {static_cast<remove_complex<ValueType>>(local_size) / global_size},
        this->get_executor());
    dense_res->scale(weight.get());

    exec->synchronize();
    auto sum_op = gko::experimental::mpi::sum<ValueType>();
    if (mpi::requires_host_buffer(exec, comm)) {
        host_reduction_buffer_.init(exec->get_master(), dense_res->get_size());
        host_reduction_buffer_->copy_from(dense_res.get());
        comm.all_reduce(exec->get_master(),
                        host_reduction_buffer_->get_values(), num_vecs,
                        sum_op.get_op());
        dense_res->copy_from(host_reduction_buffer_.get());
    } else {
        comm.all_reduce(exec, dense_res->get_values(), num_vecs,
                        sum_op.get_op());
    }
}

template <typename ValueType>
ValueType& Vector<ValueType>::at_local(size_type row, size_type col) noexcept
{
    return local_.at(row, col);
}


template <typename ValueType>
ValueType Vector<ValueType>::at_local(size_type row,
                                      size_type col) const noexcept
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


#define GKO_DECLARE_DISTRIBUTED_VECTOR(ValueType) class Vector<ValueType>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DISTRIBUTED_VECTOR);


}  // namespace distributed
}  // namespace experimental
}  // namespace gko
