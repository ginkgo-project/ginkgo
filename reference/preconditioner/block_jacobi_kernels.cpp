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

#include "core/preconditioner/block_jacobi_kernels.hpp"


#include <cmath>
#include <numeric>
#include <vector>


#include "core/base/exception_helpers.hpp"
#include "core/base/extended_float.hpp"
#include "core/base/math.hpp"
#include "core/matrix/csr.hpp"
#include "core/matrix/dense.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace block_jacobi {
namespace {


template <typename IndexType>
inline bool has_same_nonzero_pattern(const IndexType *prev_row_ptr,
                                     const IndexType *curr_row_ptr,
                                     const IndexType *next_row_ptr)
{
    if (next_row_ptr - curr_row_ptr != curr_row_ptr - prev_row_ptr) {
        return false;
    }
    for (; curr_row_ptr < next_row_ptr; ++prev_row_ptr, ++curr_row_ptr) {
        if (*curr_row_ptr != *prev_row_ptr) {
            return false;
        }
    }
    return true;
}


template <typename ValueType, typename IndexType>
size_type find_natural_blocks(const matrix::Csr<ValueType, IndexType> *mtx,
                              uint32 max_block_size, IndexType *block_ptrs)
{
    const auto rows = mtx->get_size()[0];
    const auto row_ptrs = mtx->get_const_row_ptrs();
    const auto col_idx = mtx->get_const_col_idxs();
    block_ptrs[0] = 0;
    if (rows == 0) {
        return 0;
    }
    size_type num_blocks = 1;
    int32 current_block_size = 1;
    for (size_type i = 1; i < rows; ++i) {
        const auto prev_row_ptr = col_idx + row_ptrs[i - 1];
        const auto curr_row_ptr = col_idx + row_ptrs[i];
        const auto next_row_ptr = col_idx + row_ptrs[i + 1];
        if (current_block_size < max_block_size &&
            has_same_nonzero_pattern(prev_row_ptr, curr_row_ptr,
                                     next_row_ptr)) {
            ++current_block_size;
        } else {
            block_ptrs[num_blocks] =
                block_ptrs[num_blocks - 1] + current_block_size;
            ++num_blocks;
            current_block_size = 1;
        }
    }
    block_ptrs[num_blocks] = block_ptrs[num_blocks - 1] + current_block_size;
    return num_blocks;
}


template <typename IndexType>
inline size_type agglomerate_supervariables(uint32 max_block_size,
                                            size_type num_natural_blocks,
                                            IndexType *block_ptrs)
{
    if (num_natural_blocks == 0) {
        return 0;
    }
    size_type num_blocks = 1;
    int32 current_block_size = block_ptrs[1] - block_ptrs[0];
    for (size_type i = 1; i < num_natural_blocks; ++i) {
        const int32 block_size = block_ptrs[i + 1] - block_ptrs[i];
        if (current_block_size + block_size <= max_block_size) {
            current_block_size += block_size;
        } else {
            block_ptrs[num_blocks] = block_ptrs[i];
            ++num_blocks;
            current_block_size = block_size;
        }
    }
    block_ptrs[num_blocks] = block_ptrs[num_natural_blocks];
    return num_blocks;
}


}  // namespace


template <typename ValueType, typename IndexType>
void find_blocks(std::shared_ptr<const ReferenceExecutor> exec,
                 const matrix::Csr<ValueType, IndexType> *system_matrix,
                 uint32 max_block_size, size_type &num_blocks,
                 Array<IndexType> &block_pointers)
{
    num_blocks = find_natural_blocks(system_matrix, max_block_size,
                                     block_pointers.get_data());
    num_blocks = agglomerate_supervariables(max_block_size, num_blocks,
                                            block_pointers.get_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BLOCK_JACOBI_FIND_BLOCKS_KERNEL);


namespace {


template <typename ValueType, typename IndexType>
inline void extract_block(const matrix::Csr<ValueType, IndexType> *mtx,
                          IndexType block_size, IndexType block_start,
                          ValueType *block, size_type stride)
{
    for (int i = 0; i < block_size; ++i) {
        for (int j = 0; j < block_size; ++j) {
            block[i * stride + j] = zero<ValueType>();
        }
    }
    const auto row_ptrs = mtx->get_const_row_ptrs();
    const auto col_idxs = mtx->get_const_col_idxs();
    const auto vals = mtx->get_const_values();
    for (int row = 0; row < block_size; ++row) {
        const auto start = row_ptrs[block_start + row];
        const auto end = row_ptrs[block_start + row + 1];
        for (int i = start; i < end; ++i) {
            const auto col = col_idxs[i] - block_start;
            if (0 <= col && col < block_size) {
                block[row * stride + col] = vals[i];
            }
        }
    }
}


template <typename ValueType, typename IndexType>
inline IndexType choose_pivot(IndexType block_size, const ValueType *block,
                              size_type stride)
{
    IndexType cp = 0;
    for (IndexType i = 1; i < block_size; ++i) {
        if (abs(block[cp * stride]) < abs(block[i * stride])) {
            cp = i;
        }
    }
    return cp;
}


template <typename ValueType, typename IndexType>
inline void swap_rows(IndexType row1, IndexType row2, IndexType block_size,
                      ValueType *block, size_type stride)
{
    using std::swap;
    for (IndexType i = 0; i < block_size; ++i) {
        swap(block[row1 * stride + i], block[row2 * stride + i]);
    }
}


template <typename ValueType, typename IndexType>
inline void apply_gauss_jordan_transform(IndexType row, IndexType col,
                                         IndexType block_size, ValueType *block,
                                         size_type stride)
{
    const auto d = block[row * stride + col];
    for (IndexType i = 0; i < block_size; ++i) {
        block[i * stride + col] /= -d;
    }
    block[row * stride + col] = zero<ValueType>();
    for (IndexType i = 0; i < block_size; ++i) {
        for (IndexType j = 0; j < block_size; ++j) {
            block[i * stride + j] +=
                block[i * stride + col] * block[row * stride + j];
        }
    }
    for (IndexType j = 0; j < block_size; ++j) {
        block[row * stride + j] /= d;
    }
    block[row * stride + col] = one<ValueType>() / d;
}


template <typename SourceValueType, typename ResultValueType,
          typename IndexType,
          typename ValueConverter =
              default_converter<SourceValueType, ResultValueType>>
inline void transpose_block(IndexType block_size, const SourceValueType *from,
                            size_type from_stride, ResultValueType *to,
                            size_type to_stride,
                            ValueConverter converter = {}) noexcept
{
    for (IndexType i = 0; i < block_size; ++i) {
        for (IndexType j = 0; j < block_size; ++j) {
            to[i * to_stride + j] = converter(from[i + j * from_stride]);
        }
    }
}


template <typename SourceValueType, typename ResultValueType,
          typename IndexType,
          typename ValueConverter =
              default_converter<SourceValueType, ResultValueType>>
inline void permute_and_transpose_block(IndexType block_size,
                                        const IndexType *col_perm,
                                        const SourceValueType *source,
                                        size_type source_stride,
                                        ResultValueType *result,
                                        size_type result_stride,
                                        ValueConverter converter = {})
{
    for (IndexType i = 0; i < block_size; ++i) {
        for (IndexType j = 0; j < block_size; ++j) {
            result[i + col_perm[j] * result_stride] =
                converter(source[i * source_stride + j]);
        }
    }
}


template <typename ValueType, typename IndexType>
inline void invert_block(IndexType block_size, IndexType *perm,
                         ValueType *block, size_type stride)
{
    using std::swap;
    for (IndexType k = 0; k < block_size; ++k) {
        const auto cp =
            choose_pivot(block_size - k, block + k * stride + k, stride) + k;
        swap_rows(k, cp, block_size, block, stride);
        swap(perm[k], perm[cp]);
        apply_gauss_jordan_transform(k, k, block_size, block, stride);
    }
}


}  // namespace


template <typename ValueType, typename IndexType>
void generate(std::shared_ptr<const ReferenceExecutor> exec,
              const matrix::Csr<ValueType, IndexType> *system_matrix,
              size_type num_blocks, uint32 max_block_size,
              const preconditioner::block_interleaved_storage_scheme<IndexType>
                  &storage_scheme,
              Array<precision> &block_precisions,
              const Array<IndexType> &block_pointers, Array<ValueType> &blocks)
{
    const auto ptrs = block_pointers.get_const_data();
    for (size_type b = 0; b < num_blocks; ++b) {
        const auto block_size = ptrs[b + 1] - ptrs[b];
        Array<ValueType> block(exec, block_size * block_size);
        Array<IndexType> perm(exec, block_size);
        std::iota(perm.get_data(), perm.get_data() + block_size, IndexType(0));
        extract_block(system_matrix, block_size, ptrs[b], block.get_data(),
                      block_size);
        invert_block(block_size, perm.get_data(), block.get_data(), block_size);
        permute_and_transpose_block(
            block_size, perm.get_data(), block.get_data(), block_size,
            blocks.get_data() + storage_scheme.get_global_block_offset(b),
            storage_scheme.get_stride());
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BLOCK_JACOBI_GENERATE_KERNEL);


namespace {


template <
    typename ValueType, typename BlockValueType,
    typename ValueConverter = default_converter<BlockValueType, ValueType>>
inline void apply_block(size_type block_size, size_type num_rhs,
                        const BlockValueType *block, size_type stride,
                        ValueType alpha, const ValueType *b, size_type stride_b,
                        ValueType beta, ValueType *x, size_type stride_x,
                        ValueConverter converter = {})
{
    if (beta != zero<ValueType>()) {
        for (size_type row = 0; row < block_size; ++row) {
            for (size_type col = 0; col < num_rhs; ++col) {
                x[row * stride_x + col] *= beta;
            }
        }
    } else {
        for (size_type row = 0; row < block_size; ++row) {
            for (size_type col = 0; col < num_rhs; ++col) {
                x[row * stride_x + col] = zero<ValueType>();
            }
        }
    }

    for (size_type inner = 0; inner < block_size; ++inner) {
        for (size_type row = 0; row < block_size; ++row) {
            for (size_type col = 0; col < num_rhs; ++col) {
                x[row * stride_x + col] +=
                    alpha * converter(block[row + inner * stride]) *
                    b[inner * stride_b + col];
            }
        }
    }
}


}  // namespace


void initialize_precisions(std::shared_ptr<const ReferenceExecutor> exec,
                           const Array<precision> &source,
                           Array<precision> &precisions)
{
    const auto source_size = source.get_num_elems();
    for (auto i = 0u; i < precisions.get_num_elems(); ++i) {
        precisions.get_data()[i] = source.get_const_data()[i % source_size];
    }
}


template <typename ValueType, typename IndexType>
void apply(std::shared_ptr<const ReferenceExecutor> exec, size_type num_blocks,
           uint32 max_block_size,
           const preconditioner::block_interleaved_storage_scheme<IndexType>
               &storage_scheme,
           const Array<precision> &block_precisions,
           const Array<IndexType> &block_pointers,
           const Array<ValueType> &blocks,
           const matrix::Dense<ValueType> *alpha,
           const matrix::Dense<ValueType> *b,
           const matrix::Dense<ValueType> *beta, matrix::Dense<ValueType> *x)
{
    const auto ptrs = block_pointers.get_const_data();
    for (size_type i = 0; i < num_blocks; ++i) {
        const auto block =
            blocks.get_const_data() + storage_scheme.get_global_block_offset(i);
        const auto block_b = b->get_const_values() + b->get_stride() * ptrs[i];
        const auto block_x = x->get_values() + x->get_stride() * ptrs[i];
        const auto block_size = ptrs[i + 1] - ptrs[i];
        apply_block(block_size, b->get_size()[1], block,
                    storage_scheme.get_stride(), alpha->at(0, 0), block_b,
                    b->get_stride(), beta->at(0, 0), block_x, x->get_stride());
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BLOCK_JACOBI_APPLY_KERNEL);


template <typename ValueType, typename IndexType>
void simple_apply(
    std::shared_ptr<const ReferenceExecutor> exec, size_type num_blocks,
    uint32 max_block_size,
    const preconditioner::block_interleaved_storage_scheme<IndexType>
        &storage_scheme,
    const Array<precision> &block_precisions,
    const Array<IndexType> &block_pointers, const Array<ValueType> &blocks,
    const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *x)
{
    const auto ptrs = block_pointers.get_const_data();
    for (size_type i = 0; i < num_blocks; ++i) {
        const auto block =
            blocks.get_const_data() + storage_scheme.get_global_block_offset(i);
        const auto block_b = b->get_const_values() + b->get_stride() * ptrs[i];
        const auto block_x = x->get_values() + x->get_stride() * ptrs[i];
        const auto block_size = ptrs[i + 1] - ptrs[i];
        apply_block(block_size, b->get_size()[1], block,
                    storage_scheme.get_stride(), one<ValueType>(), block_b,
                    b->get_stride(), zero<ValueType>(), block_x,
                    x->get_stride());
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BLOCK_JACOBI_SIMPLE_APPLY_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_dense(
    std::shared_ptr<const ReferenceExecutor> exec, size_type num_blocks,
    const Array<precision> &block_precisions,
    const Array<IndexType> &block_pointers, const Array<ValueType> &blocks,
    const preconditioner::block_interleaved_storage_scheme<IndexType>
        &storage_scheme,
    ValueType *result_values, size_type result_stride)
{
    const auto ptrs = block_pointers.get_const_data();
    const size_type matrix_size = ptrs[num_blocks];
    for (size_type i = 0; i < matrix_size; ++i) {
        for (size_type j = 0; j < matrix_size; ++j) {
            result_values[i * result_stride + j] = zero<ValueType>();
        }
    }

    for (size_type i = 0; i < num_blocks; ++i) {
        const auto block =
            blocks.get_const_data() + storage_scheme.get_global_block_offset(i);
        const auto block_size = ptrs[i + 1] - ptrs[i];
        transpose_block(block_size, block, storage_scheme.get_stride(),
                        result_values + ptrs[i] * result_stride + ptrs[i],
                        result_stride);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BLOCK_JACOBI_CONVERT_TO_DENSE_KERNEL);


}  // namespace block_jacobi


/*
namespace adaptive_block_jacobi {


#define RESOLVE_PRECISION(prec, call)                                       \
    if (prec == precision<ValueType, IndexType>::double_precision) {        \
        using resolved_precision = ValueType;                               \
        call;                                                               \
    } else if (prec == precision<ValueType, IndexType>::single_precision) { \
        using resolved_precision = reduce_precision<ValueType>;             \
        call;                                                               \
    } else if (prec == precision<ValueType, IndexType>::half_precision) {   \
        using resolved_precision =                                          \
            reduce_precision<reduce_precision<ValueType>>;                  \
        call;                                                               \
    } else {                                                                \
        throw NOT_SUPPORTED(                                                \
            (precision<ValueType, IndexType>::best_precision));             \
    }


template <typename ValueType, typename IndexType>
void apply(std::shared_ptr<const ReferenceExecutor> exec, size_type num_blocks,
           uint32 max_block_size,
           const preconditioner::block_interleaved_storage_scheme<IndexType>
               &storage_scheme,
           const Array<precision<ValueType, IndexType>> &block_precisions,
           const Array<IndexType> &block_pointers,
           const Array<ValueType> &blocks,
           const matrix::Dense<ValueType> *alpha,
           const matrix::Dense<ValueType> *b,
           const matrix::Dense<ValueType> *beta, matrix::Dense<ValueType> *x)
{
    const auto ptrs = block_pointers.get_const_data();
    const auto prec = block_precisions.get_const_data();
    for (size_type i = 0; i < num_blocks; ++i) {
        // TODO: use the same precision for the block group and optimize the
        // storage scheme for it
        const auto block =
            blocks.get_const_data() + storage_scheme.get_global_block_offset(i);
        const auto block_b = b->get_const_values() + b->get_stride() * ptrs[i];
        const auto block_x = x->get_values() + x->get_stride() * ptrs[i];
        const auto block_size = ptrs[i + 1] - ptrs[i];
        RESOLVE_PRECISION(
            prec[i],
            block_jacobi::apply_block(
                block_size, b->get_size()[1],
                reinterpret_cast<const resolved_precision *>(block),
                storage_scheme.get_stride(), alpha->at(0, 0), block_b,
                b->get_stride(), beta->at(0, 0), block_x, x->get_stride()));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ADAPTIVE_BLOCK_JACOBI_APPLY_KERNEL);


template <typename ValueType, typename IndexType>
void simple_apply(
    std::shared_ptr<const ReferenceExecutor> exec, size_type num_blocks,
    uint32 max_block_size,
    const preconditioner::block_interleaved_storage_scheme<IndexType>
        &storage_scheme,
    const Array<precision<ValueType, IndexType>> &block_precisions,
    const Array<IndexType> &block_pointers, const Array<ValueType> &blocks,
    const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *x)
{
    const auto ptrs = block_pointers.get_const_data();
    const auto prec = block_precisions.get_const_data();
    for (size_type i = 0; i < num_blocks; ++i) {
        // TODO: use the same precision for the block group and optimize the
        // storage scheme for it
        const auto block =
            blocks.get_const_data() + storage_scheme.get_global_block_offset(i);
        const auto block_b = b->get_const_values() + b->get_stride() * ptrs[i];
        const auto block_x = x->get_values() + x->get_stride() * ptrs[i];
        const auto block_size = ptrs[i + 1] - ptrs[i];
        RESOLVE_PRECISION(
            prec[i],
            block_jacobi::apply_block(
                block_size, b->get_size()[1],
                reinterpret_cast<const resolved_precision *>(block),
                storage_scheme.get_stride(), one<ValueType>(), block_b,
                b->get_stride(), zero<ValueType>(), block_x, x->get_stride()));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ADAPTIVE_BLOCK_JACOBI_SIMPLE_APPLY_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_dense(
    std::shared_ptr<const ReferenceExecutor> exec, size_type num_blocks,
    const Array<precision<ValueType, IndexType>> &block_precisions,
    const Array<IndexType> &block_pointers, const Array<ValueType> &blocks,
    const preconditioner::block_interleaved_storage_scheme<IndexType>
        &storage_scheme,
    ValueType *result_values, size_type result_stride)
{
    const auto ptrs = block_pointers.get_const_data();
    const auto prec = block_precisions.get_const_data();
    const size_type matrix_size = ptrs[num_blocks];
    for (size_type i = 0; i < matrix_size; ++i) {
        for (size_type j = 0; j < matrix_size; ++j) {
            result_values[i * result_stride + j] = zero<ValueType>();
        }
    }

    for (size_type i = 0; i < num_blocks; ++i) {
        // TODO: use the same precision for the block group and optimize the
        // storage scheme for it
        const auto block =
            blocks.get_const_data() + storage_scheme.get_global_block_offset(i);
        const auto block_size = ptrs[i + 1] - ptrs[i];
        RESOLVE_PRECISION(
            prec[i],
            block_jacobi::transpose_block(
                block_size, reinterpret_cast<const resolved_precision *>(block),
                storage_scheme.get_stride(),
                result_values + ptrs[i] * result_stride + ptrs[i],
                result_stride));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ADAPTIVE_BLOCK_JACOBI_CONVERT_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void generate(std::shared_ptr<const ReferenceExecutor> exec,
              const matrix::Csr<ValueType, IndexType> *system_matrix,
              size_type num_blocks, uint32 max_block_size,
              const preconditioner::block_interleaved_storage_scheme<IndexType>
                  &storage_scheme,
              Array<precision<ValueType, IndexType>> &block_precisions,
              const Array<IndexType> &block_pointers, Array<ValueType> &blocks)
{
    const auto ptrs = block_pointers.get_const_data();
    const auto prec = block_precisions.get_data();
    for (size_type b = 0; b < num_blocks; ++b) {
        const auto block_size = ptrs[b + 1] - ptrs[b];
        Array<ValueType> block(exec, block_size * block_size);
        Array<IndexType> perm(exec, block_size);
        std::iota(perm.get_data(), perm.get_data() + block_size, IndexType(0));
        block_jacobi::extract_block(system_matrix, block_size, ptrs[b],
                                    block.get_data(), block_size);
        block_jacobi::invert_block(block_size, perm.get_data(),
                                   block.get_data(), block_size);
        if (prec[b] == precision<ValueType, IndexType>::best_precision) {
            // TODO: properly compute best precision
            prec[b] = precision<ValueType, IndexType>::double_precision;
        }
        // TODO: use the same precision for the block group and optimize the
        // storage scheme for it
        RESOLVE_PRECISION(
            prec[b],
            block_jacobi::permute_and_transpose_block(
                block_size, perm.get_data(), block.get_data(), block_size,
                reinterpret_cast<resolved_precision *>(
                    blocks.get_data() +
                    storage_scheme.get_global_block_offset(b)),
                storage_scheme.get_stride()));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ADAPTIVE_BLOCK_JACOBI_GENERATE_KERNEL);

}  // namespace adaptive_block_jacobi
*/

}  // namespace reference
}  // namespace kernels
}  // namespace gko
