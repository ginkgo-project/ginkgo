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

#ifndef GKO_PUBLIC_CORE_DISTRIBUTED_VECTOR_HPP_
#define GKO_PUBLIC_CORE_DISTRIBUTED_VECTOR_HPP_


#include <ginkgo/config.hpp>


#if GKO_HAVE_MPI


#include <ginkgo/core/distributed/base.hpp>
#include <ginkgo/core/distributed/communicator.hpp>
#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace distributed {


template <typename ValueType = double>
class Vector
    : public EnableLinOp<Vector<ValueType>>,
      public EnableCreateMethod<Vector<ValueType>>,
      public ConvertibleTo<Vector<next_precision<ValueType>>>,
      public EnableAbsoluteComputation<remove_complex<Vector<ValueType>>>,
      public DistributedBase {
    friend class EnableCreateMethod<Vector<ValueType>>;
    friend class EnablePolymorphicObject<Vector, LinOp>;
    friend class Vector<to_complex<ValueType>>;
    friend class Vector<next_precision<ValueType>>;

public:
    using EnableLinOp<Vector>::convert_to;
    using EnableLinOp<Vector>::move_to;

    using value_type = ValueType;
    using index_type = int64;
    using absolute_type = remove_complex<Vector>;
    using complex_type = to_complex<Vector>;
    using local_mtx_type = matrix::Dense<value_type>;

    void convert_to(Vector<next_precision<ValueType>> *result) const override;

    void move_to(Vector<next_precision<ValueType>> *result) override;

    /**
     * Fill the distributed vector with a given value.
     *
     * @param value  the value to be filled
     */
    void fill(const ValueType value);

    void read_distributed(const matrix_data<ValueType, global_index_type> &data,
                          std::shared_ptr<const Partition<int64>> partition);

    void read_distributed(const matrix_data<ValueType, global_index_type> &data,
                          std::shared_ptr<const Partition<int32>> partition);

    std::unique_ptr<absolute_type> compute_absolute() const override;

    void compute_absolute_inplace() override;

    /**
     * Scales the matrix with a scalar (aka: BLAS scal).
     *
     * @param alpha  If alpha is 1x1 Dense matrix, the entire matrix is scaled
     *               by alpha. If it is a Dense row vector of values,
     *               then i-th column of the matrix is scaled with the i-th
     *               element of alpha (the number of columns of alpha has to
     *               match the number of columns of the matrix).
     */
    void scale(const LinOp *alpha);

    /**
     * Adds `b` scaled by `alpha` to the matrix (aka: BLAS axpy).
     *
     * @param alpha  If alpha is 1x1 Dense matrix, the entire matrix is scaled
     *               by alpha. If it is a Dense row vector of values,
     *               then i-th column of the matrix is scaled with the i-th
     *               element of alpha (the number of columns of alpha has to
     *               match the number of columns of the matrix).
     * @param b  a matrix of the same dimension as this
     */
    void add_scaled(const LinOp *alpha, const LinOp *b);

    /**
     * Computes the column-wise dot product of this matrix and `b`.
     *
     * @param b  a Dense matrix of same dimension as this
     * @param result  a Dense row vector, used to store the dot product
     *                (the number of column in the vector must match the number
     *                of columns of this)
     */
    void compute_dot(const LinOp *b, LinOp *result) const;

    /**
     * Computes the column-wise dot product of this matrix and `conj(b)`.
     *
     * @param b  a Dense matrix of same dimension as this
     * @param result  a Dense row vector, used to store the dot product
     *                (the number of column in the vector must match the number
     *                of columns of this)
     */
    void compute_conj_dot(const LinOp *b, LinOp *result) const;

    /**
     * Computes the Euclidian (L^2) norm of this matrix.
     *
     * @param result  a Dense row vector, used to store the norm
     *                (the number of columns in the vector must match the number
     *                of columns of this)
     */
    void compute_norm2(LinOp *result) const;

    const local_mtx_type *get_local() const;

    // Promise not to break things? :)
    local_mtx_type *get_local();

protected:
    Vector(std::shared_ptr<const Executor> exec, communicator comm,
           dim<2> global_size, dim<2> local_size, size_type stride);

    Vector(std::shared_ptr<const Executor> exec, communicator comm = {},
           dim<2> global_size = {}, dim<2> local_size = {});

    void apply_impl(const LinOp *, LinOp *) const override;

    void apply_impl(const LinOp *, const LinOp *, const LinOp *,
                    LinOp *) const override;

private:
    matrix::Dense<ValueType> local_;
};


}  // namespace distributed
}  // namespace gko


#endif  // GKO_HAVE_MPI
#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_VECTOR_HPP_