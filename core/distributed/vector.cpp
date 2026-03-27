// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/distributed/vector.hpp"

#include <ginkgo/core/distributed/partition.hpp>

#include "core/distributed/vector_kernels.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "core/mpi/mpi_op.hpp"
#include "ginkgo/core/base/temporary_conversion.hpp"

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
    : matrix::EnableMultiVector<Vector>{exec, global_size},
      DistributedBase{comm},
      local_{exec, local_size, stride}
{
    GKO_ASSERT_EQUAL_COLS(global_size, local_size);
}

template <typename ValueType>
Vector<ValueType>::Vector(std::shared_ptr<const Executor> exec,
                          mpi::communicator comm, dim<2> global_size,
                          std::unique_ptr<local_vector_type> local_vector)
    : matrix::EnableMultiVector<Vector>{exec, global_size},
      DistributedBase{comm},
      local_{exec}
{
    local_vector->move_to(&local_);
}


template <typename ValueType>
Vector<ValueType>::Vector(std::shared_ptr<const Executor> exec,
                          mpi::communicator comm,
                          std::unique_ptr<local_vector_type> local_vector)
    : matrix::EnableMultiVector<Vector>{exec, {}},
      DistributedBase{comm},
      local_{exec}
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
template <typename OtherValueType>
gko::detail::temporary_conversion<Vector<OtherValueType>>
Vector<ValueType>::as_precision()
{
    return gko::detail::temporary_conversion<Vector<OtherValueType>>::create(
        this);
}

#define GKO_DECLARE_VECTOR_AS_PRECISION(ValueType, OtherValueType) \
    auto Vector<ValueType>::as_precision()                         \
        ->gko::detail::temporary_conversion<Vector<OtherValueType>>
#define GKO_DECLARE_VECTOR_AS_PRECISION_same(ValueType) \
    GKO_DECLARE_VECTOR_AS_PRECISION(ValueType, ValueType)
GKO_INSTANTIATE_FOR_EACH_VALUE_CONVERSION(GKO_DECLARE_VECTOR_AS_PRECISION);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_VECTOR_AS_PRECISION_same);


template <typename ValueType>
template <typename OtherValueType>
gko::detail::temporary_conversion<const Vector<OtherValueType>>
Vector<ValueType>::as_precision() const
{
    return gko::detail::temporary_conversion<
        const Vector<OtherValueType>>::create(this);
}

#define GKO_DECLARE_VECTOR_CONST_AS_PRECISION(ValueType, OtherValueType) \
    auto Vector<ValueType>::as_precision()                               \
        const->gko::detail::temporary_conversion<const Vector<OtherValueType>>
#define GKO_DECLARE_VECTOR_CONST_AS_PRECISION_same(ValueType) \
    GKO_DECLARE_VECTOR_CONST_AS_PRECISION(ValueType, ValueType)
GKO_INSTANTIATE_FOR_EACH_VALUE_CONVERSION(
    GKO_DECLARE_VECTOR_CONST_AS_PRECISION);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_VECTOR_CONST_AS_PRECISION_same);


template <typename ValueType>
std::unique_ptr<Vector<ValueType>> Vector<ValueType>::create_subview_impl(
    local_span rows, local_span columns)
{
    auto exec = this->get_executor();
    auto comm = this->get_communicator();
    auto global_rows = this->get_size()[0];
    auto global_cols = this->get_size()[1];
    comm.all_reduce(exec, &global_rows, 1, MPI_SUM);
    comm.all_reduce(exec, &global_cols, 1, MPI_SUM);
    return create_subview_impl(rows, columns, {global_rows, global_cols});
}


template <typename ValueType>
std::unique_ptr<const Vector<ValueType>> Vector<ValueType>::create_subview_impl(
    local_span rows, local_span columns) const
{
    auto exec = this->get_executor();
    auto comm = this->get_communicator();
    auto global_rows = this->get_size()[0];
    auto global_cols = this->get_size()[1];
    comm.all_reduce(exec, &global_rows, 1, MPI_SUM);
    comm.all_reduce(exec, &global_cols, 1, MPI_SUM);
    return create_subview_impl(rows, columns, {global_rows, global_cols});
}


template <typename ValueType>
std::unique_ptr<const Vector<ValueType>> Vector<ValueType>::create_subview_impl(
    local_span rows, local_span columns, dim<2> global_size) const
{
    // @todo: use const-cast here until dense also has const create_submatrix
    return create(
        this->get_executor(), this->get_communicator(), global_size,
        const_cast<local_vector_type&>(local_).create_subview(rows, columns));
}


template <typename ValueType>
std::unique_ptr<Vector<ValueType>> Vector<ValueType>::create_subview_impl(
    local_span rows, local_span columns, dim<2> global_size)
{
    return create(this->get_executor(), this->get_communicator(), global_size,
                  local_.create_subview(rows, columns));
}


template <typename ValueType>
typename Vector<ValueType>::device_view
Vector<ValueType>::get_local_device_view_impl()
{
    return local_.get_device_view();
}


template <typename ValueType>
typename Vector<ValueType>::const_device_view
Vector<ValueType>::get_const_local_device_view_impl() const
{
    return local_.get_const_device_view();
}


template <typename ValueType>
template <typename LocalIndexType, typename GlobalIndexType>
void Vector<ValueType>::read_distributed_impl(
    const device_matrix_data<value_type, GlobalIndexType>& data,
    const Partition<LocalIndexType, GlobalIndexType>* partition)
{
    auto exec = this->get_executor();
    auto global_cols = data.get_size()[1];
    this->resize(
        dim<2>(partition->get_size(), global_cols),
        dim<2>(partition->get_part_size(this->get_communicator().rank()),
               global_cols));

    auto rank = this->get_communicator().rank();
    local_.fill(zero<value_type>());
    exec->run(vector::make_build_local(
        data, make_temporary_clone(exec, partition).get(), rank,
        local_.get_device_view()));
}


template <typename ValueType>
void Vector<ValueType>::read_distributed(
    const device_matrix_data<value_type, int64>& data,
    ptr_param<const Partition<int64, int64>> partition)
{
    this->read_distributed_impl(data, partition.get());
}


template <typename ValueType>
void Vector<ValueType>::read_distributed(
    const device_matrix_data<value_type, int64>& data,
    ptr_param<const Partition<int32, int64>> partition)
{
    this->read_distributed_impl(data, partition.get());
}


template <typename ValueType>
void Vector<ValueType>::read_distributed(
    const device_matrix_data<value_type, int32>& data,
    ptr_param<const Partition<int32, int32>> partition)
{
    this->read_distributed_impl(data, partition.get());
}


template <typename ValueType>
void Vector<ValueType>::read_distributed(
    const matrix_data<value_type, int64>& data,
    ptr_param<const Partition<int64, int64>> partition)
{
    this->read_distributed(
        device_matrix_data<value_type, int64>::create_from_host(
            this->get_executor(), data),
        partition);
}


template <typename ValueType>
void Vector<ValueType>::read_distributed(
    const matrix_data<value_type, int64>& data,
    ptr_param<const Partition<int32, int64>> partition)
{
    this->read_distributed(
        device_matrix_data<value_type, int64>::create_from_host(
            this->get_executor(), data),
        partition);
}


template <typename ValueType>
void Vector<ValueType>::read_distributed(
    const matrix_data<value_type, int32>& data,
    ptr_param<const Partition<int32, int32>> partition)
{
    this->read_distributed(
        device_matrix_data<value_type, int32>::create_from_host(
            this->get_executor(), data),
        partition);
}


template <typename ValueType>
void Vector<ValueType>::fill_impl(value_type value)
{
    local_.fill(value);
}


template <typename ValueType>
void Vector<ValueType>::convert_to(
    Vector<next_precision<value_type>>* result) const
{
    GKO_ASSERT(this->get_communicator().size() ==
               result->get_communicator().size());
    result->set_size(this->get_size());
    this->get_local_vector()->convert_to(&result->local_);
}


template <typename ValueType>
void Vector<ValueType>::move_to(Vector<next_precision<value_type>>* result)
{
    this->convert_to(result);
}


#if GINKGO_ENABLE_HALF || GINKGO_ENABLE_BFLOAT16
template <typename ValueType>
void Vector<ValueType>::convert_to(
    Vector<next_precision<value_type, 2>>* result) const
{
    GKO_ASSERT(this->get_communicator().size() ==
               result->get_communicator().size());
    result->set_size(this->get_size());
    this->get_local_vector()->convert_to(&result->local_);
}


template <typename ValueType>
void Vector<ValueType>::move_to(Vector<next_precision<value_type, 2>>* result)
{
    this->convert_to(result);
}
#endif


#if GINKGO_ENABLE_HALF && GINKGO_ENABLE_BFLOAT16
template <typename ValueType>
void Vector<ValueType>::convert_to(
    Vector<next_precision<value_type, 3>>* result) const
{
    GKO_ASSERT(this->get_communicator().size() ==
               result->get_communicator().size());
    result->set_size(this->get_size());
    this->get_local_vector()->convert_to(&result->local_);
}


template <typename ValueType>
void Vector<ValueType>::move_to(Vector<next_precision<value_type, 3>>* result)
{
    this->convert_to(result);
}
#endif

template <typename ValueType>
std::unique_ptr<typename Vector<ValueType>::absolute_type>
Vector<ValueType>::compute_absolute_impl() const
{
    auto exec = this->get_executor();

    auto result =
        absolute_type::create(exec, this->get_communicator(), this->get_size(),
                              this->get_local_vector()->get_size());

    exec->run(vector::make_outplace_absolute_dense(
        this->get_local_vector()->get_const_device_view(),
        result->local_.get_device_view()));

    return result;
}


template <typename ValueType>
void Vector<ValueType>::compute_absolute_impl(absolute_type* result) const
{
    local_.compute_absolute(&result->local_);
}


template <typename ValueType>
void Vector<ValueType>::compute_absolute_inplace_impl()
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
Vector<ValueType>::make_complex_impl() const
{
    auto result = complex_type::create(
        this->get_executor(), this->get_communicator(), this->get_size(),
        this->get_local_vector()->get_size(),
        this->get_local_vector()->get_stride());
    this->make_complex(result);
    return result;
}


template <typename ValueType>
void Vector<ValueType>::make_complex_impl(complex_type* result) const
{
    this->get_local_vector()->make_complex(&result->local_);
}


template <typename ValueType>
std::unique_ptr<typename Vector<ValueType>::real_type>
Vector<ValueType>::get_real_impl() const
{
    auto result = real_type::create(this->get_executor(),
                                    this->get_communicator(), this->get_size(),
                                    this->get_local_vector()->get_size(),
                                    this->get_local_vector()->get_stride());
    this->get_real(result);
    return result;
}


template <typename ValueType>
void Vector<ValueType>::get_real_impl(real_type* result) const
{
    this->get_local_vector()->get_real(&result->local_);
}


template <typename ValueType>
std::unique_ptr<typename Vector<ValueType>::real_type>
Vector<ValueType>::get_imag_impl() const
{
    auto result = real_type::create(this->get_executor(),
                                    this->get_communicator(), this->get_size(),
                                    this->get_local_vector()->get_size(),
                                    this->get_local_vector()->get_stride());
    this->get_imag(result);
    return result;
}


template <typename ValueType>
void Vector<ValueType>::get_imag_impl(real_type* result) const
{
    this->get_local_vector()->get_imag(&result->local_);
}


template <typename ValueType>
void Vector<ValueType>::scale_impl(const matrix::Dense<value_type>* alpha)
{
    local_.scale(alpha);
}


template <typename ValueType>
void Vector<ValueType>::inv_scale_impl(const matrix::Dense<value_type>* alpha)
{
    local_.inv_scale(alpha);
}


template <typename ValueType>
void Vector<ValueType>::add_scaled_impl(const matrix::Dense<value_type>* alpha,
                                        const Vector* b)
{
    auto dense_b = as<Vector>(b);
    local_.add_scaled(alpha, dense_b->get_local_vector());
}


template <typename ValueType>
void Vector<ValueType>::sub_scaled_impl(const matrix::Dense<value_type>* alpha,
                                        const Vector* b)
{
    auto dense_b = as<Vector>(b);
    local_.sub_scaled(alpha, dense_b->get_local_vector());
}


template <typename ValueType>
void Vector<ValueType>::compute_dot_impl(const Vector* b,
                                         local_vector_type* result) const
{
    array<char> tmp{this->get_executor()};
    this->compute_dot(b, result, tmp);
}


template <typename ValueType>
void Vector<ValueType>::compute_dot_impl(const Vector* b,
                                         local_vector_type* result,
                                         array<char>& tmp) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    auto exec = this->get_executor();
    const auto comm = this->get_communicator();
    auto dense_res =
        make_temporary_clone(exec, as<matrix::Dense<value_type>>(result));
    this->get_local_vector()->compute_dot(as<Vector>(b)->get_local_vector(),
                                          dense_res.get(), tmp);
    exec->synchronize();
    auto sum_op = gko::experimental::mpi::sum<value_type>();
    if (mpi::requires_host_buffer(exec, comm)) {
        host_reduction_buffer_.init(exec->get_master(), dense_res->get_size());
        host_reduction_buffer_->copy_from(dense_res.get());
        comm.all_reduce(exec->get_master(),
                        host_reduction_buffer_->get_values(),
                        static_cast<int>(this->get_size()[1]), sum_op.get_op());
        dense_res->copy_from(host_reduction_buffer_.get());
    } else {
        comm.all_reduce(exec, dense_res->get_values(),
                        static_cast<int>(this->get_size()[1]), sum_op.get_op());
    }
}


template <typename ValueType>
void Vector<ValueType>::compute_conj_dot_impl(const Vector* b,
                                              local_vector_type* result) const
{
    array<char> tmp{this->get_executor()};
    this->compute_conj_dot(b, result, tmp);
}


template <typename ValueType>
void Vector<ValueType>::compute_conj_dot_impl(const Vector* b,
                                              local_vector_type* result,
                                              array<char>& tmp) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    auto exec = this->get_executor();
    const auto comm = this->get_communicator();
    auto dense_res =
        make_temporary_clone(exec, as<matrix::Dense<value_type>>(result));
    this->get_local_vector()->compute_conj_dot(
        as<Vector>(b)->get_local_vector(), dense_res.get(), tmp);
    exec->synchronize();
    auto sum_op = gko::experimental::mpi::sum<value_type>();
    if (mpi::requires_host_buffer(exec, comm)) {
        host_reduction_buffer_.init(exec->get_master(), dense_res->get_size());
        host_reduction_buffer_->copy_from(dense_res.get());
        comm.all_reduce(exec->get_master(),
                        host_reduction_buffer_->get_values(),
                        static_cast<int>(this->get_size()[1]), sum_op.get_op());
        dense_res->copy_from(host_reduction_buffer_.get());
    } else {
        comm.all_reduce(exec, dense_res->get_values(),
                        static_cast<int>(this->get_size()[1]), sum_op.get_op());
    }
}


template <typename ValueType>
void Vector<ValueType>::compute_norm2_impl(
    local_absolute_vector_type* result) const
{
    array<char> tmp{this->get_executor()};
    this->compute_norm2(result, tmp);
}


template <typename ValueType>
void Vector<ValueType>::compute_norm2_impl(local_absolute_vector_type* result,
                                           array<char>& tmp) const
{
    using NormVector = typename local_vector_type::absolute_type;
    auto exec = this->get_executor();
    const auto comm = this->get_communicator();
    auto dense_res = make_temporary_clone(exec, as<NormVector>(result));
    this->compute_squared_norm2(dense_res.get(), tmp);
    exec->run(vector::make_compute_sqrt(dense_res->get_device_view()));
}


template <typename ValueType>
void Vector<ValueType>::compute_norm1_impl(
    local_absolute_vector_type* result) const
{
    array<char> tmp{this->get_executor()};
    this->compute_norm1(result, tmp);
}


template <typename ValueType>
void Vector<ValueType>::compute_norm1_impl(local_absolute_vector_type* result,
                                           array<char>& tmp) const
{
    using NormVector = typename local_vector_type::absolute_type;
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    auto exec = this->get_executor();
    const auto comm = this->get_communicator();
    auto dense_res = make_temporary_clone(exec, as<NormVector>(result));
    this->get_local_vector()->compute_norm1(dense_res.get());
    exec->synchronize();
    auto norm_sum_op =
        gko::experimental::mpi::sum<remove_complex<value_type>>();
    if (mpi::requires_host_buffer(exec, comm)) {
        host_norm_buffer_.init(exec->get_master(), dense_res->get_size());
        host_norm_buffer_->copy_from(dense_res.get());
        comm.all_reduce(exec->get_master(), host_norm_buffer_->get_values(),
                        static_cast<int>(this->get_size()[1]),
                        norm_sum_op.get_op());
        dense_res->copy_from(host_norm_buffer_.get());
    } else {
        comm.all_reduce(exec, dense_res->get_values(),
                        static_cast<int>(this->get_size()[1]),
                        norm_sum_op.get_op());
    }
}


template <typename ValueType>
void Vector<ValueType>::compute_squared_norm2_impl(
    local_absolute_vector_type* result) const
{
    array<char> tmp{this->get_executor()};
    this->compute_squared_norm2(result, tmp);
}


template <typename ValueType>
void Vector<ValueType>::compute_squared_norm2_impl(
    local_absolute_vector_type* result, array<char>& tmp) const
{
    using NormVector = typename local_vector_type::absolute_type;
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    auto exec = this->get_executor();
    const auto comm = this->get_communicator();
    auto dense_res = make_temporary_clone(exec, as<NormVector>(result));
    exec->run(vector::make_compute_squared_norm2(
        this->get_local_vector()->get_const_device_view(),
        dense_res->get_device_view(), tmp));
    exec->synchronize();
    auto norm_sum_op =
        gko::experimental::mpi::sum<remove_complex<value_type>>();
    if (mpi::requires_host_buffer(exec, comm)) {
        host_norm_buffer_.init(exec->get_master(), dense_res->get_size());
        host_norm_buffer_->copy_from(dense_res.get());
        comm.all_reduce(exec->get_master(), host_norm_buffer_->get_values(),
                        static_cast<int>(this->get_size()[1]),
                        norm_sum_op.get_op());
        dense_res->copy_from(host_norm_buffer_.get());
    } else {
        comm.all_reduce(exec, dense_res->get_values(),
                        static_cast<int>(this->get_size()[1]),
                        norm_sum_op.get_op());
    }
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
    auto weight = initialize<matrix::Dense<remove_complex<value_type>>>(
        {static_cast<remove_complex<value_type>>(local_size) / global_size},
        this->get_executor());
    dense_res->scale(weight.get());

    exec->synchronize();
    auto sum_op = gko::experimental::mpi::sum<value_type>();
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
auto Vector<ValueType>::at_local(size_type row, size_type col) noexcept
    -> value_type&
{
    return local_.at(row, col);
}


template <typename ValueType>
auto Vector<ValueType>::at_local(size_type row, size_type col) const noexcept
    -> value_type
{
    return local_.at(row, col);
}


template <typename ValueType>
auto Vector<ValueType>::at_local(size_type idx) noexcept -> value_type&
{
    return local_.at(idx);
}

template <typename ValueType>
auto Vector<ValueType>::at_local(size_type idx) const noexcept -> value_type
{
    return local_.at(idx);
}


template <typename ValueType>
auto Vector<ValueType>::get_local_values() -> value_type*
{
    return local_.get_values();
}


template <typename ValueType>
auto Vector<ValueType>::get_const_local_values() const -> const value_type*
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
Vector<ValueType>::create_real_view_impl() const
{
    const auto num_global_rows = this->get_size()[0];
    const auto num_cols = is_complex<value_type>() ? 2 * this->get_size()[1]
                                                   : this->get_size()[1];

    return real_type::create_const(
        this->get_executor(), this->get_communicator(),
        dim<2>{num_global_rows, num_cols}, local_.create_real_view());
}


template <typename ValueType>
std::unique_ptr<typename Vector<ValueType>::real_type>
Vector<ValueType>::create_real_view_impl()
{
    const auto num_global_rows = this->get_size()[0];
    const auto num_cols = is_complex<value_type>() ? 2 * this->get_size()[1]
                                                   : this->get_size()[1];

    return real_type::create(this->get_executor(), this->get_communicator(),
                             dim<2>{num_global_rows, num_cols},
                             local_.create_real_view());
}


template <typename ValueType>
std::unique_ptr<Vector<ValueType>>
Vector<ValueType>::create_with_same_config_impl() const
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


}  // namespace distributed
}  // namespace experimental
}  // namespace gko
