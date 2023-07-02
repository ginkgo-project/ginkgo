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

#include "core/matrix/bccoo_kernels.hpp"


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/bccoo.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/components/fill_array_kernels.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/matrix/bccoo_aux_structs.hpp"
#include "core/matrix/bccoo_helper.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "cuda/base/config.hpp"
#include "cuda/base/cusparse_bindings.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"
#include "cuda/base/unaligned_access.hpp"
#include "cuda/components/atomic.cuh"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/format_conversion.cuh"
#include "cuda/components/segment_scan.cuh"
#include "cuda/components/thread_ids.cuh"


namespace gko {
namespace kernels {
/**
 * @brief The CUDA namespace.
 *
 * @ingroup cuda
 */
namespace cuda {
/**
 * @brief The Bccoordinate matrix format namespace.
 *
 * @ingroup bccoo
 */
namespace bccoo {


using namespace matrix::bccoo;


constexpr int default_block_size = 512;
constexpr int warps_in_block = 4;
constexpr int spmv_block_size = warps_in_block * config::warp_size;


#include "common/cuda_hip/matrix/bccoo_helper.hpp.inc"
#include "common/cuda_hip/matrix/bccoo_kernels.hpp.inc"


template <typename IndexType>
void get_default_block_size(std::shared_ptr<const CudaExecutor> exec,
                            IndexType* block_size)
{
    *block_size = 32;
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_GET_DEFAULT_BLOCK_SIZE_KERNEL);


void get_default_compression(std::shared_ptr<const CudaExecutor> exec,
                             compression* compression)
{
    *compression = matrix::bccoo::compression::group;
}


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const CudaExecutor> exec,
          const matrix::Bccoo<ValueType, IndexType>* a,
          const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)
{
    dense::fill(exec, c, zero<ValueType>());
    spmv2(exec, a, b, c);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_BCCOO_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const CudaExecutor> exec,
                   const matrix::Dense<ValueType>* alpha,
                   const matrix::Bccoo<ValueType, IndexType>* a,
                   const matrix::Dense<ValueType>* b,
                   const matrix::Dense<ValueType>* beta,
                   matrix::Dense<ValueType>* c)
{
    dense::scale(exec, beta, c);
    advanced_spmv2(exec, alpha, a, b, c);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_ADVANCED_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void spmv2(std::shared_ptr<const CudaExecutor> exec,
           const matrix::Bccoo<ValueType, IndexType>* a,
           const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)
{
    const IndexType nnz = a->get_num_stored_elements();
    const IndexType block_size = a->get_block_size();
    const IndexType num_blocks_matrix = a->get_num_blocks();
    const IndexType b_ncols = b->get_size()[1];
    const dim3 bccoo_block(config::warp_size, warps_in_block, 1);
    const IndexType nwarps = host_kernel::calculate_nwarps(exec, nnz);

    if (nwarps > 0) {
        // If there is work to compute
        if (a->use_group_compression()) {
            IndexType num_blocks_grid = std::min(
                num_blocks_matrix,
                static_cast<IndexType>(ceildiv(nwarps, warps_in_block)));
            const dim3 bccoo_grid(num_blocks_grid, b_ncols);
            IndexType num_lines = ceildiv(num_blocks_matrix, num_blocks_grid);

            kernel::abstract_spmv<<<bccoo_grid, bccoo_block>>>(
                nnz, num_blocks_matrix, block_size, num_lines,
                as_cuda_type(a->get_const_compressed_data()),
                as_cuda_type(a->get_const_block_offsets()),
                as_cuda_type(a->get_const_compression_types()),
                as_cuda_type(a->get_const_start_cols()),
                as_cuda_type(a->get_const_start_rows()),
                as_cuda_type(b->get_const_values()),
                static_cast<IndexType>(b->get_stride()),
                as_cuda_type(c->get_values()),
                static_cast<IndexType>(c->get_stride()));
        } else {
            GKO_NOT_SUPPORTED(a);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_BCCOO_SPMV2_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv2(std::shared_ptr<const CudaExecutor> exec,
                    const matrix::Dense<ValueType>* alpha,
                    const matrix::Bccoo<ValueType, IndexType>* a,
                    const matrix::Dense<ValueType>* b,
                    matrix::Dense<ValueType>* c)
{
    const IndexType nnz = a->get_num_stored_elements();
    const IndexType block_size = a->get_block_size();
    const IndexType num_blocks_matrix = a->get_num_blocks();
    const IndexType b_ncols = b->get_size()[1];
    const dim3 bccoo_block(config::warp_size, warps_in_block, 1);
    const IndexType nwarps = host_kernel::calculate_nwarps(exec, nnz);

    if (nwarps > 0) {
        // If there is work to compute
        if (a->use_group_compression()) {
            IndexType num_blocks_grid = std::min(
                num_blocks_matrix,
                static_cast<IndexType>(ceildiv(nwarps, warps_in_block)));
            const dim3 bccoo_grid(num_blocks_grid, b_ncols);
            IndexType num_lines = ceildiv(num_blocks_matrix, num_blocks_grid);

            kernel::abstract_spmv<<<bccoo_grid, bccoo_block>>>(
                nnz, num_blocks_matrix, block_size, num_lines,
                as_cuda_type(alpha->get_const_values()),
                as_cuda_type(a->get_const_compressed_data()),
                as_cuda_type(a->get_const_block_offsets()),
                as_cuda_type(a->get_const_compression_types()),
                as_cuda_type(a->get_const_start_cols()),
                as_cuda_type(a->get_const_start_rows()),
                as_cuda_type(b->get_const_values()),
                static_cast<IndexType>(b->get_stride()),
                as_cuda_type(c->get_values()),
                static_cast<IndexType>(c->get_stride()));
        } else {
            GKO_NOT_SUPPORTED(a);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_ADVANCED_SPMV2_KERNEL);


template <typename ValueType, typename IndexType>
void mem_size_bccoo(std::shared_ptr<const CudaExecutor> exec,
                    const matrix::Bccoo<ValueType, IndexType>* source,
                    compression compress_res, const IndexType block_size_res,
                    size_type* mem_size) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_MEM_SIZE_BCCOO_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_bccoo(std::shared_ptr<const CudaExecutor> exec,
                      const matrix::Bccoo<ValueType, IndexType>* source,
                      matrix::Bccoo<ValueType, IndexType>* result)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_CONVERT_TO_BCCOO_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_next_precision(
    std::shared_ptr<const CudaExecutor> exec,
    const matrix::Bccoo<ValueType, IndexType>* source,
    matrix::Bccoo<next_precision<ValueType>, IndexType>* result)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_CONVERT_TO_NEXT_PRECISION_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_coo(std::shared_ptr<const CudaExecutor> exec,
                    const matrix::Bccoo<ValueType, IndexType>* source,
                    matrix::Coo<ValueType, IndexType>* result)
{
    const IndexType nnz = source->get_num_stored_elements();

    IndexType* row_idxs = result->get_row_idxs();
    IndexType* col_idxs = result->get_col_idxs();
    ValueType* values = result->get_values();

    const IndexType block_size = source->get_block_size();
    const IndexType num_blocks_matrix = source->get_num_blocks();
    const dim3 bccoo_block(config::warp_size, warps_in_block, 1);
    const IndexType nwarps = host_kernel::calculate_nwarps(exec, nnz);

    if (nwarps > 0) {
        // If there is work to compute
        if (source->use_group_compression()) {
            IndexType num_blocks_grid = std::min(
                num_blocks_matrix,
                static_cast<IndexType>(ceildiv(nwarps, warps_in_block)));
            const dim3 bccoo_grid(num_blocks_grid, 1);
            IndexType num_lines = ceildiv(num_blocks_matrix, num_blocks_grid);

            kernel::abstract_fill_in_coo<<<bccoo_grid, bccoo_block>>>(
                nnz, num_blocks_matrix, block_size, num_lines,
                as_cuda_type(source->get_const_compressed_data()),
                as_cuda_type(source->get_const_block_offsets()),
                as_cuda_type(source->get_const_compression_types()),
                as_cuda_type(source->get_const_start_cols()),
                as_cuda_type(source->get_const_start_rows()),
                as_cuda_type(result->get_row_idxs()),
                as_cuda_type(result->get_col_idxs()),
                as_cuda_type(result->get_values()));
        } else {
            GKO_NOT_SUPPORTED(source);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_CONVERT_TO_COO_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_csr(std::shared_ptr<const CudaExecutor> exec,
                    const matrix::Bccoo<ValueType, IndexType>* source,
                    matrix::Csr<ValueType, IndexType>* result)
{
    const IndexType nnz = source->get_num_stored_elements();
    const IndexType num_rows = source->get_size()[0];

    array<IndexType> row_idxs(exec, nnz);

    IndexType* row_ptrs = result->get_row_ptrs();
    IndexType* col_idxs = result->get_col_idxs();
    ValueType* values = result->get_values();

    const IndexType block_size = source->get_block_size();
    const IndexType num_blocks_matrix = source->get_num_blocks();
    const dim3 bccoo_block(config::warp_size, warps_in_block, 1);
    const IndexType nwarps = host_kernel::calculate_nwarps(exec, nnz);

    if (nwarps > 0) {
        // If there is work to compute
        if (source->use_group_compression()) {
            IndexType num_blocks_grid = std::min(
                num_blocks_matrix,
                static_cast<IndexType>(ceildiv(nwarps, warps_in_block)));
            const dim3 bccoo_grid(num_blocks_grid, 1);
            IndexType num_lines = ceildiv(num_blocks_matrix, num_blocks_grid);

            kernel::abstract_fill_in_coo<<<bccoo_grid, bccoo_block>>>(
                nnz, num_blocks_matrix, block_size, num_lines,
                as_cuda_type(source->get_const_compressed_data()),
                as_cuda_type(source->get_const_block_offsets()),
                as_cuda_type(source->get_const_compression_types()),
                as_cuda_type(source->get_const_start_cols()),
                as_cuda_type(source->get_const_start_rows()),
                as_cuda_type(row_idxs.get_data()),
                as_cuda_type(result->get_col_idxs()),
                as_cuda_type(result->get_values()));

            components::convert_idxs_to_ptrs(exec, row_idxs.get_data(), nnz,
                                             num_rows + 1, row_ptrs);
        } else {
            GKO_NOT_SUPPORTED(source);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_CONVERT_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_dense(std::shared_ptr<const CudaExecutor> exec,
                      const matrix::Bccoo<ValueType, IndexType>* source,
                      matrix::Dense<ValueType>* result)
{
    const IndexType num_rows = result->get_size()[0];
    const IndexType num_cols = result->get_size()[1];
    const IndexType stride = result->get_stride();

    const IndexType nnz = source->get_num_stored_elements();
    const IndexType block_size = source->get_block_size();
    const IndexType num_blocks_matrix = source->get_num_blocks();
    const dim3 bccoo_block(config::warp_size, warps_in_block, 1);
    const IndexType nwarps = host_kernel::calculate_nwarps(exec, nnz);

    const dim3 block_size_mat(config::warp_size,
                              config::max_block_size / config::warp_size, 1);
    const dim3 init_grid_dim(ceildiv(num_cols, block_size_mat.x),
                             ceildiv(num_rows, block_size_mat.y), 1);
    dense::fill(exec, result, zero<ValueType>());

    if (nwarps > 0) {
        // If there is work to compute
        if (source->use_group_compression()) {
            IndexType num_blocks_grid = std::min(
                num_blocks_matrix,
                static_cast<IndexType>(ceildiv(nwarps, warps_in_block)));
            const dim3 bccoo_grid(num_blocks_grid, 1);
            IndexType num_lines = ceildiv(num_blocks_matrix, num_blocks_grid);

            kernel::abstract_fill_in_dense<<<bccoo_grid, bccoo_block>>>(
                nnz, num_blocks_matrix, block_size, num_lines,
                as_cuda_type(source->get_const_compressed_data()),
                as_cuda_type(source->get_const_block_offsets()),
                as_cuda_type(source->get_const_compression_types()),
                as_cuda_type(source->get_const_start_cols()),
                as_cuda_type(source->get_const_start_rows()), stride,
                as_cuda_type(result->get_values()));
        } else {
            GKO_NOT_SUPPORTED(source);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_CONVERT_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void extract_diagonal(std::shared_ptr<const CudaExecutor> exec,
                      const matrix::Bccoo<ValueType, IndexType>* orig,
                      matrix::Diagonal<ValueType>* diag)
{
    const IndexType nnz = orig->get_num_stored_elements();
    const IndexType block_size = orig->get_block_size();
    const IndexType num_blocks_matrix = orig->get_num_blocks();
    const dim3 bccoo_block(config::warp_size, warps_in_block, 1);
    const IndexType nwarps = host_kernel::calculate_nwarps(exec, nnz);

    if (nwarps > 0) {
        // If there is work to compute
        if (orig->use_group_compression()) {
            IndexType num_blocks_grid = std::min(
                num_blocks_matrix,
                static_cast<IndexType>(ceildiv(nwarps, warps_in_block)));
            const dim3 bccoo_grid(num_blocks_grid, 1);
            IndexType num_lines = ceildiv(num_blocks_matrix, num_blocks_grid);

            kernel::abstract_extract<<<bccoo_grid, bccoo_block>>>(
                nnz, num_blocks_matrix, block_size, num_lines,
                as_cuda_type(orig->get_const_compressed_data()),
                as_cuda_type(orig->get_const_block_offsets()),
                as_cuda_type(orig->get_const_compression_types()),
                as_cuda_type(orig->get_const_start_cols()),
                as_cuda_type(orig->get_const_start_rows()),
                as_cuda_type(diag->get_values()));
        } else {
            GKO_NOT_SUPPORTED(orig);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_EXTRACT_DIAGONAL_KERNEL);


template <typename ValueType, typename IndexType>
void compute_absolute_inplace(std::shared_ptr<const CudaExecutor> exec,
                              matrix::Bccoo<ValueType, IndexType>* matrix)
{
    const IndexType nnz = matrix->get_num_stored_elements();
    const IndexType block_size = matrix->get_block_size();
    const IndexType num_blocks_matrix = matrix->get_num_blocks();
    const dim3 bccoo_block(config::warp_size, warps_in_block, 1);
    const IndexType nwarps = host_kernel::calculate_nwarps(exec, nnz);

    if (nwarps > 0) {
        // If there is work to compute
        if (matrix->use_group_compression()) {
            IndexType num_blocks_grid = std::min(
                num_blocks_matrix,
                static_cast<IndexType>(ceildiv(nwarps, warps_in_block)));
            const dim3 bccoo_grid(num_blocks_grid, 1);
            IndexType num_lines = ceildiv(num_blocks_matrix, num_blocks_grid);

            kernel::abstract_absolute_inplace<
                config::warp_size, cuda_type<ValueType>, cuda_type<IndexType>>
                <<<bccoo_grid, bccoo_block>>>(
                    nnz, num_blocks_matrix, block_size, num_lines,
                    as_cuda_type(matrix->get_compressed_data()),
                    as_cuda_type(matrix->get_const_block_offsets()),
                    as_cuda_type(matrix->get_const_compression_types()),
                    as_cuda_type(matrix->get_const_start_cols()),
                    as_cuda_type(matrix->get_const_start_rows()));
        } else {
            GKO_NOT_SUPPORTED(matrix);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_COMPUTE_ABSOLUTE_INPLACE_KERNEL);


template <typename ValueType, typename IndexType>
void compute_absolute(
    std::shared_ptr<const CudaExecutor> exec,
    const matrix::Bccoo<ValueType, IndexType>* source,
    remove_complex<matrix::Bccoo<ValueType, IndexType>>* result)
{
    const IndexType nnz = source->get_num_stored_elements();
    const IndexType block_size = source->get_block_size();
    const IndexType num_blocks_matrix = source->get_num_blocks();
    const dim3 bccoo_block(config::warp_size, warps_in_block, 1);
    const IndexType nwarps = host_kernel::calculate_nwarps(exec, nnz);

    if (nwarps > 0) {
        // If there is work to compute
        if (source->use_group_compression()) {
            IndexType num_blocks_grid = std::min(
                num_blocks_matrix,
                static_cast<IndexType>(ceildiv(nwarps, warps_in_block)));
            const dim3 bccoo_grid(num_blocks_grid, 1);
            IndexType num_lines = ceildiv(num_blocks_matrix, num_blocks_grid);

            kernel::abstract_absolute<config::warp_size, cuda_type<ValueType>,
                                      cuda_type<IndexType>>
                <<<bccoo_grid, bccoo_block>>>(
                    nnz, num_blocks_matrix, block_size, num_lines,
                    as_cuda_type(source->get_const_compressed_data()),
                    as_cuda_type(source->get_const_block_offsets()),
                    as_cuda_type(source->get_const_compression_types()),
                    as_cuda_type(source->get_const_start_cols()),
                    as_cuda_type(source->get_const_start_rows()),
                    as_cuda_type(result->get_compressed_data()),
                    as_cuda_type(result->get_block_offsets()),
                    as_cuda_type(result->get_compression_types()),
                    as_cuda_type(result->get_start_cols()),
                    as_cuda_type(result->get_start_rows()));
        } else {
            GKO_NOT_SUPPORTED(source);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_COMPUTE_ABSOLUTE_KERNEL);


}  // namespace bccoo
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
