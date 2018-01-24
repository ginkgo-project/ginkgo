/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include "core/matrix/dense_kernels.hpp"


#include "core/base/math.hpp"
#include "gpu/base/cublas_bindings.hpp"


namespace gko {
namespace kernels {
namespace gpu {
namespace dense {


template <typename ValueType>
void simple_apply(const matrix::Dense<ValueType> *a,
                  const matrix::Dense<ValueType> *b,
                  matrix::Dense<ValueType> *c)
{
    auto handle = cublas::init();
    ASSERT_NO_CUBLAS_ERRORS(
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    auto alpha = one<ValueType>();
    auto beta = zero<ValueType>();
    cublas::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, c->get_num_cols(),
                 c->get_num_rows(), a->get_num_cols(), &alpha,
                 b->get_const_values(), b->get_padding(), a->get_const_values(),
                 a->get_padding(), &beta, c->get_values(), c->get_padding());
    cublas::destroy(handle);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_SIMPLE_APPLY_KERNEL);


template <typename ValueType>
void apply(const matrix::Dense<ValueType> *alpha,
           const matrix::Dense<ValueType> *a, const matrix::Dense<ValueType> *b,
           const matrix::Dense<ValueType> *beta, matrix::Dense<ValueType> *c)
{
    auto handle = cublas::init();
    cublas::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, c->get_num_cols(),
                 c->get_num_rows(), a->get_num_cols(),
                 alpha->get_const_values(), b->get_const_values(),
                 b->get_padding(), a->get_const_values(), a->get_padding(),
                 beta->get_const_values(), c->get_values(), c->get_padding());
    cublas::destroy(handle);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_APPLY_KERNEL);


template <typename ValueType>
void scale(const matrix::Dense<ValueType> *alpha, matrix::Dense<ValueType> *x)
{
    auto handle = cublas::init();
    if (alpha->get_num_cols() == 1) {
        cublas::scal(handle, x->get_num_stored_elements(),
                     alpha->get_const_values(), x->get_values(), 1);
    } else {
        // TODO: write a custom kernel which does this more efficiently
        for (size_type col = 0; col < x->get_num_cols(); ++col) {
            cublas::scal(handle, x->get_num_rows(),
                         alpha->get_const_values() + col, x->get_values() + col,
                         x->get_padding());
        }
    }
    cublas::destroy(handle);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_SCALE_KERNEL);


template <typename ValueType>
void add_scaled(const matrix::Dense<ValueType> *alpha,
                const matrix::Dense<ValueType> *x, matrix::Dense<ValueType> *y)
{
    auto handle = cublas::init();
    // TODO: write a custom kernel which does this more efficiently
    if (alpha->get_num_cols() == 1) {
        // cannot write as single kernel call, x and y can have different
        // paddings
        for (size_type col = 0; col < x->get_num_cols(); ++col) {
            cublas::axpy(handle, x->get_num_rows(), alpha->get_const_values(),
                         x->get_const_values() + col, x->get_padding(),
                         y->get_values() + col, y->get_padding());
        }
    } else {
        for (size_type col = 0; col < x->get_num_cols(); ++col) {
            cublas::axpy(handle, x->get_num_rows(),
                         alpha->get_const_values() + col,
                         x->get_const_values() + col, x->get_padding(),
                         y->get_values() + col, y->get_padding());
        }
    }
    cublas::destroy(handle);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_ADD_SCALED_KERNEL);


template <typename ValueType>
void compute_dot(const matrix::Dense<ValueType> *x,
                 const matrix::Dense<ValueType> *y,
                 matrix::Dense<ValueType> *result)
{
    auto handle = cublas::init();
    // TODO: write a custom kernel which does this more efficiently
    for (size_type col = 0; col < x->get_num_cols(); ++col) {
        cublas::dot(handle, x->get_num_rows(), x->get_const_values() + col,
                    x->get_padding(), y->get_const_values() + col,
                    y->get_padding(), result->get_values() + col);
    }
    cublas::destroy(handle);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_COMPUTE_DOT_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_csr(matrix::Csr<ValueType, IndexType> *result,
                    const matrix::Dense<ValueType> *source) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void move_to_csr(matrix::Csr<ValueType, IndexType> *result,
                 const matrix::Dense<ValueType> *source) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_MOVE_TO_CSR_KERNEL);


template <typename ValueType>
void count_nonzeros(const matrix::Dense<ValueType> *source,
                    size_type *result) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_COUNT_NONZEROS_KERNEL);


template <typename ValueType>
void transpose(matrix::Dense<ValueType> *trans,
               const matrix::Dense<ValueType> *orig)
{
    auto handle = cublas::init();

    ASSERT_NO_CUBLAS_ERRORS(
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

    auto alpha = one<ValueType>();
    auto beta = zero<ValueType>();

    cublas::geam(handle, CUBLAS_OP_T, CUBLAS_OP_N, orig->get_num_rows(),
                 orig->get_num_cols(), &alpha, orig->get_const_values(),
                 orig->get_padding(), &beta, nullptr, trans->get_num_cols(),
                 trans->get_values(), trans->get_padding());

    cublas::destroy(handle);
};

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_TRANSPOSE_KERNEL);


template <typename ValueType>
void conj_transpose(matrix::Dense<ValueType> *trans,
                    const matrix::Dense<ValueType> *orig)

{
    auto handle = cublas::init();

    ASSERT_NO_CUBLAS_ERRORS(
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

    auto alpha = one<ValueType>();
    auto beta = zero<ValueType>();

    cublas::geam(handle, CUBLAS_OP_C, CUBLAS_OP_N, orig->get_num_rows(),
                 orig->get_num_cols(), &alpha, orig->get_const_values(),
                 orig->get_padding(), &beta, nullptr, trans->get_num_cols(),
                 trans->get_values(), trans->get_padding());

    cublas::destroy(handle);
};

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CONJ_TRANSPOSE_KERNEL);


}  // namespace dense
}  // namespace gpu
}  // namespace kernels
}  // namespace gko
