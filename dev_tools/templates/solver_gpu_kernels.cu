/*
  * Copyright 2017-2018
 * 
 * Karlsruhe Institute of Technology
 * 
 * Universitat Jaume I
 * 
 * University of Tennessee
 * 
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
*/

#include "core/solver/xxsolverxx_kernels.hpp"


#include "core/base/exception_helpers.hpp"


namespace gko {
namespace kernels {
namespace gpu {
namespace xxsolverxx {


struct size {
    size_type num_rows_;
    size_type num_cols_;
    constexpr size_type get_num_rows() const noexcept { return num_rows_; }
    constexpr size_type get_num_cols() const noexcept { return num_cols_; }
};


inline int64 ceildiv(int64 a, int64 b) { return (a + b - 1) / b; }


// This is example code for the CG case - has to be modified for the new solver
/*


template <typename ValueType>
__global__ void initialize_kernel(size_type m, size_type n, size_type lda,
                                  const ValueType *b, ValueType *r,
                                  ValueType *z, ValueType *p, ValueType *q,
                                  ValueType *prev_rho, ValueType *rho)
{
    size_type tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx < n) {
        rho[tidx] = zero<ValueType>();
        prev_rho[tidx] = one<ValueType>();
    }

    if (tidx < m * lda) {
        r[tidx] = b[tidx];
        z[tidx] = zero<ValueType>();
        q[tidx] = zero<ValueType>();
        p[tidx] = zero<ValueType>();
    }
}


template <typename ValueType>
void initialize(const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *r,
                matrix::Dense<ValueType> *z, matrix::Dense<ValueType> *p,
                matrix::Dense<ValueType> *q, matrix::Dense<ValueType> *prev_rho,
                matrix::Dense<ValueType> *rho)
{
    ASSERT_EQUAL_DIMENSIONS(b, r);
    ASSERT_EQUAL_DIMENSIONS(b, z);
    ASSERT_EQUAL_DIMENSIONS(b, p);
    ASSERT_EQUAL_DIMENSIONS(b, z);
    const size vector{b->get_num_cols(), 1};
    ASSERT_EQUAL_DIMENSIONS(prev_rho, &vector);
    ASSERT_EQUAL_DIMENSIONS(rho, &vector);

    constexpr int block_size_x = 512;
    const dim3 block_size(block_size_x, 1, 1);
    const dim3 grid_size(
        ceildiv(b->get_num_rows() * b->get_padding(), block_size.x), 1, 1);

    initialize_kernel<<<grid_size, block_size, 0, 0>>>(
        b->get_num_rows(), b->get_num_cols(), b->get_padding(),
        b->get_const_values(), r->get_values(), z->get_values(),
        p->get_values(), q->get_values(), prev_rho->get_values(),
        rho->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_XXSOLVERXX_INITIALIZE_KERNEL);


template <typename ValueType>
__global__ void step_1_kernel(size_type m, size_type n, size_type lda,
                              ValueType *p, const ValueType *z,
                              const ValueType *rho, const ValueType *prev_rho)
{
    size_type tidx = blockDim.x * blockIdx.x + threadIdx.x;
    size_type col = tidx % lda;
    ValueType tmp = zero<ValueType>();


    if (tidx < m * lda) {
        tmp = rho[col] / prev_rho[col];
        p[tidx] =
            (tmp == zero<ValueType>()) ? z[tidx] : z[tidx] + tmp * p[tidx];
    }
}


template <typename ValueType>
void step_1(matrix::Dense<ValueType> *p, const matrix::Dense<ValueType> *z,
            const matrix::Dense<ValueType> *rho,
            const matrix::Dense<ValueType> *prev_rho)
{
    ASSERT_EQUAL_DIMENSIONS(p, z);
    const size vector{p->get_num_cols(), 1};
    ASSERT_EQUAL_DIMENSIONS(prev_rho, &vector);
    ASSERT_EQUAL_DIMENSIONS(rho, &vector);

    constexpr int block_size_x = 512;
    const dim3 block_size(block_size_x, 1, 1);
    const dim3 grid_size(
        ceildiv(p->get_num_rows() * p->get_padding(), block_size.x), 1, 1);

    step_1_kernel<<<grid_size, block_size, 0, 0>>>(
        p->get_num_rows(), p->get_num_cols(), p->get_padding(), p->get_values(),
        z->get_const_values(), rho->get_const_values(),
        prev_rho->get_const_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_XXSOLVERXX_STEP_1_KERNEL);


template <typename ValueType>
__global__ void step_2_kernel(size_type m, size_type n, size_type lda,
                              ValueType *x, ValueType *r, const ValueType *p,
                              const ValueType *q, const ValueType *beta,
                              const ValueType *rho)
{
    size_type tidx = blockDim.x * blockIdx.x + threadIdx.x;
    size_type col = tidx % lda;
    ValueType tmp = zero<ValueType>();

    if (tidx < m * lda) {
        tmp = rho[col] / beta[col];
        x[tidx] =
            (tmp == zero<ValueType>()) ? x[tidx] : x[tidx] + tmp * p[tidx];
        r[tidx] =
            (tmp == zero<ValueType>()) ? r[tidx] : r[tidx] - tmp * q[tidx];
    }
}


template <typename ValueType>
void step_2(matrix::Dense<ValueType> *x, matrix::Dense<ValueType> *r,
            const matrix::Dense<ValueType> *p,
            const matrix::Dense<ValueType> *q,
            const matrix::Dense<ValueType> *beta,
            const matrix::Dense<ValueType> *rho)
{
    ASSERT_EQUAL_DIMENSIONS(x, r);
    ASSERT_EQUAL_DIMENSIONS(x, p);
    ASSERT_EQUAL_DIMENSIONS(x, q);
    const size vector{x->get_num_cols(), 1};
    ASSERT_EQUAL_DIMENSIONS(beta, &vector);
    ASSERT_EQUAL_DIMENSIONS(rho, &vector);

    constexpr int block_size_x = 512;
    const dim3 block_size(block_size_x, 1, 1);
    const dim3 grid_size(
        ceildiv(p->get_num_rows() * p->get_padding(), block_size.x), 1, 1);

    step_2_kernel<<<grid_size, block_size, 0, 0>>>(
        p->get_num_rows(), p->get_num_cols(), p->get_padding(), x->get_values(),
        r->get_values(), p->get_const_values(), q->get_const_values(),
        beta->get_const_values(), rho->get_const_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_XXSOLVERXX_STEP_2_KERNEL);


*/


}  // namespace xxsolverxx
}  // namespace gpu
}  // namespace kernels
}  // namespace gko
