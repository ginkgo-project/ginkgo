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
    this->get_local()->convert_to(result->get_local());
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Vector<ValueType, LocalIndexType, GlobalIndexType>::move_to(
    Vector<next_precision<ValueType>, LocalIndexType, GlobalIndexType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
std::unique_ptr<
    typename Vector<ValueType, LocalIndexType, GlobalIndexType>::absolute_type>
Vector<ValueType, LocalIndexType, GlobalIndexType>::compute_absolute() const
{
    auto exec = this->get_executor();

    auto result = absolute_type::create(exec, this->get_communicator(),
                                        this->get_partition(), this->get_size(),
                                        this->get_local()->get_size());

    exec->run(vector::make_outplace_absolute_dense(this->get_local(),
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
                      GlobalIndexType>::local_mtx_type*
Vector<ValueType, LocalIndexType, GlobalIndexType>::get_local() const
{
    return &local_;
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
typename Vector<ValueType, LocalIndexType, GlobalIndexType>::local_mtx_type*
Vector<ValueType, LocalIndexType, GlobalIndexType>::get_local()
{
    return &local_;
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Vector<ValueType, LocalIndexType, GlobalIndexType>::Vector(
    std::shared_ptr<const Executor> exec, mpi::communicator comm,
    std::shared_ptr<const Partition<LocalIndexType, GlobalIndexType>> partition,
    dim<2> global_size, dim<2> local_size, size_type stride)
    : EnableLinOp<
          Vector<ValueType, LocalIndexType, GlobalIndexType>>{exec,
                                                              global_size},
      DistributedBase{comm},
      partition_{
          partition
              ? std::move(partition)
              : gko::share(
                    Partition<LocalIndexType, GlobalIndexType>::create(exec))},
      local_{exec, local_size,
             stride != invalid_index<size_type>() ? stride : local_size[1]}
{}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType,
          typename LocalMtxType>
void read_local_impl(
    std::shared_ptr<const Executor> exec, mpi::communicator comm,
    const Partition<LocalIndexType, GlobalIndexType>* partition,
    const size_type num_cols,
    const Array<matrix_data_entry<ValueType, GlobalIndexType>>& global_data,
    LocalMtxType* local_mtx)
{
    auto rank = comm.rank();

    Array<matrix_data_entry<ValueType, LocalIndexType>> local_data{exec};
    exec->run(vector::make_build_local(global_data, partition, rank, local_data,
                                       ValueType{}));

    auto local_rows = static_cast<size_type>(partition->get_part_size(rank));
    dim<2> local_size{local_rows, num_cols};
    local_mtx->read({local_size, local_data});
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Vector<ValueType, LocalIndexType, GlobalIndexType>::read_distributed(
    const matrix_data<ValueType, GlobalIndexType>& data,
    std::shared_ptr<const Partition<LocalIndexType, GlobalIndexType>> partition)
{
    this->partition_ = std::move(partition);

    auto exec = this->get_executor();
    Array<matrix_data_entry<ValueType, GlobalIndexType>> global_data{
        exec, data.nonzeros.begin(), data.nonzeros.end()};
    read_local_impl(exec, this->get_communicator(), this->get_partition().get(),
                    data.size[1], global_data, this->get_local());

    auto global_rows = static_cast<size_type>(this->partition_->get_size());
    this->set_size({global_rows, data.size[1]});
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Vector<ValueType, LocalIndexType, GlobalIndexType>::read_distributed(
    const device_matrix_data<ValueType, GlobalIndexType>& data,
    std::shared_ptr<const Partition<LocalIndexType, GlobalIndexType>> partition)
{
    this->partition_ = std::move(partition);

    read_local_impl(this->get_executor(), this->get_communicator(),
                    this->get_partition().get(), data.size[1], data.nonzeros,
                    this->get_local());

    auto global_rows = static_cast<size_type>(this->partition_->get_size());
    this->set_size({global_rows, data.size[1]});
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
    this->get_local()->add_scaled(alpha, dense_b->get_local());
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
    auto dense_res =
        make_temporary_clone(exec, as<matrix::Dense<ValueType>>(result));
    this->get_local()->compute_dot(as<Vector>(b)->get_local(), dense_res.get());
    exec->synchronize();
    auto dense_res_host =
        make_temporary_clone(exec->get_master(), dense_res.get());
    this->get_communicator().all_reduce(dense_res_host->get_values(),
                                        static_cast<int>(this->get_size()[1]),
                                        MPI_SUM);
    dense_res->copy_from(dense_res_host.get());
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Vector<ValueType, LocalIndexType, GlobalIndexType>::compute_conj_dot(
    const LinOp* b, LinOp* result) const
{
    auto exec = this->get_executor();
    auto dense_res =
        make_temporary_clone(exec, as<matrix::Dense<ValueType>>(result));
    this->get_local()->compute_conj_dot(as<Vector>(b)->get_local(),
                                        dense_res.get());
    exec->synchronize();
    this->get_communicator().all_reduce(dense_res->get_values(),
                                        static_cast<int>(this->get_size()[1]),
                                        MPI_SUM);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Vector<ValueType, LocalIndexType, GlobalIndexType>::compute_norm2(
    LinOp* result) const
{
    using NormVector = matrix::Dense<remove_complex<ValueType>>;
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    auto exec = this->get_executor();
    auto dense_res = make_temporary_clone(exec, as<NormVector>(result));
    exec->run(
        vector::make_compute_norm2_sqr(this->get_local(), dense_res.get()));
    exec->synchronize();
    this->get_communicator().all_reduce(dense_res->get_values(),
                                        static_cast<int>(this->get_size()[1]),
                                        MPI_SUM);
    exec->run(vector::make_compute_sqrt(dense_res.get()));
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Vector<ValueType, LocalIndexType, GlobalIndexType>::compute_norm1(
    LinOp* result) const
{
    using NormVector = typename local_mtx_type::absolute_type;
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    auto exec = this->get_executor();
    auto dense_res = make_temporary_clone(exec, as<NormVector>(result));
    this->get_const_local()->compute_norm1(dense_res.get());
    exec->synchronize();
    this->get_communicator().all_reduce(dense_res->get_values(),
                                        static_cast<int>(this->get_size()[1]),
                                        MPI_SUM);
}


#define GKO_DECLARE_DISTRIBUTED_VECTOR(ValueType, LocalIndexType, \
                                       GlobalIndexType)           \
    class Vector<ValueType, LocalIndexType, GlobalIndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_DISTRIBUTED_VECTOR);


}  // namespace distributed
}  // namespace gko
