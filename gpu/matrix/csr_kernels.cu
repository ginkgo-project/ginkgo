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

#include "core/matrix/csr_kernels.hpp"


#include "core/base/exception_helpers.hpp"
#include "core/base/math.hpp"
#include "gpu/base/cusparse_bindings.hpp"
#include "gpu/base/math.hpp"
#include "gpu/base/types.hpp"

namespace gko {
namespace kernels {
namespace gpu {
namespace csr {


constexpr int default_block_size = 512;


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const GpuExecutor> exec,
          const matrix::Csr<ValueType, IndexType> *a,
          const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c)
{
    // TODO: add implementation for int64 and multiple RHS
    auto handle = cusparse::init();
    auto descr = cusparse::create_mat_descr();
    ASSERT_NO_CUSPARSE_ERRORS(
        cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));

    auto row_ptrs = a->get_const_row_ptrs();
    auto col_idxs = a->get_const_col_idxs();
    auto alpha = one<ValueType>();
    auto beta = zero<ValueType>();
    if (b->get_stride() != 1 || c->get_stride() != 1) NOT_IMPLEMENTED;

    cusparse::spmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                   a->get_size().num_rows, a->get_size().num_cols,
                   a->get_num_stored_elements(), &alpha, descr,
                   a->get_const_values(), row_ptrs, col_idxs,
                   b->get_const_values(), &beta, c->get_values());

    cusparse::destroy(descr);
    cusparse::destroy(handle);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const GpuExecutor> exec,
                   const matrix::Dense<ValueType> *alpha,
                   const matrix::Csr<ValueType, IndexType> *a,
                   const matrix::Dense<ValueType> *b,
                   const matrix::Dense<ValueType> *beta,
                   matrix::Dense<ValueType> *c)
{
    // TODO: add implementation for int64 and multiple RHS
    auto handle = cusparse::init();
    auto descr = cusparse::create_mat_descr();

    auto row_ptrs = a->get_const_row_ptrs();
    auto col_idxs = a->get_const_col_idxs();

    if (b->get_stride() != 1 || c->get_stride() != 1) NOT_IMPLEMENTED;

    cusparse::spmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                   a->get_size().num_rows, a->get_size().num_cols,
                   a->get_num_stored_elements(), alpha->get_const_values(),
                   descr, a->get_const_values(), row_ptrs, col_idxs,
                   b->get_const_values(), beta->get_const_values(),
                   c->get_values());

    cusparse::destroy(descr);
    cusparse::destroy(handle);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_ADVANCED_SPMV_KERNEL);


template <typename IndexType>
void convert_row_ptrs_to_idxs(std::shared_ptr<const GpuExecutor> exec,
                              const IndexType *ptrs, size_type num_rows,
                              IndexType *idxs) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_CSR_CONVERT_ROW_PTRS_TO_IDXS_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_dense(
    std::shared_ptr<const GpuExecutor> exec, matrix::Dense<ValueType> *result,
    const matrix::Csr<ValueType, IndexType> *source) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONVERT_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void move_to_dense(std::shared_ptr<const GpuExecutor> exec,
                   matrix::Dense<ValueType> *result,
                   matrix::Csr<ValueType, IndexType> *source) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_MOVE_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void transpose(std::shared_ptr<const GpuExecutor> exec,
               matrix::Csr<ValueType, IndexType> *trans,
               const matrix::Csr<ValueType, IndexType> *orig)
{
    auto handle = cusparse::init();
    cusparseAction_t copyValues = CUSPARSE_ACTION_NUMERIC;
    cusparseIndexBase_t idxBase = CUSPARSE_INDEX_BASE_ZERO;

    cusparse::transpose(
        handle, orig->get_size().num_rows, orig->get_size().num_cols,
        orig->get_num_stored_elements(), orig->get_const_values(),
        orig->get_const_row_ptrs(), orig->get_const_col_idxs(),
        trans->get_values(), trans->get_col_idxs(), trans->get_row_ptrs(),
        copyValues, idxBase);

    cusparse::destroy(handle);
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_TRANSPOSE_KERNEL);


namespace {


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void conjugate_kernel(
    size_type num_nonzeros, ValueType *__restrict__ val)
{
    const auto tidx =
        static_cast<size_type>(blockIdx.x) * default_block_size + threadIdx.x;

    if (tidx < num_nonzeros) {
        val[tidx] = conj(val[tidx]);
    }
}


}  //  namespace


template <typename ValueType, typename IndexType>
void conj_transpose(std::shared_ptr<const GpuExecutor> exec,
                    matrix::Csr<ValueType, IndexType> *trans,
                    const matrix::Csr<ValueType, IndexType> *orig)
{
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(trans->get_num_stored_elements(), block_size.x), 1, 1);

    auto handle = cusparse::init();
    cusparseAction_t copyValues = CUSPARSE_ACTION_NUMERIC;
    cusparseIndexBase_t idxBase = CUSPARSE_INDEX_BASE_ZERO;

    cusparse::transpose(
        handle, orig->get_size().num_rows, orig->get_size().num_cols,
        orig->get_num_stored_elements(), orig->get_const_values(),
        orig->get_const_row_ptrs(), orig->get_const_col_idxs(),
        trans->get_values(), trans->get_col_idxs(), trans->get_row_ptrs(),
        copyValues, idxBase);

    cusparse::destroy(handle);

    conjugate_kernel<<<grid_size, block_size, 0, 0>>>(
        trans->get_num_stored_elements(), as_cuda_type(trans->get_values()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONJ_TRANSPOSE_KERNEL);


}  // namespace csr
}  // namespace gpu
}  // namespace kernels
}  // namespace gko
