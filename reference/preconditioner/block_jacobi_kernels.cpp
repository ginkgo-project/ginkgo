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
#include "core/base/math.hpp"
#include "core/matrix/csr.hpp"
#include "core/matrix/dense.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace block_jacobi {


template <typename ValueType, typename IndexType>
void find_blocks(const matrix::Csr<ValueType, IndexType> *system_matrix,
                 uint32 max_block_size, size_type &num_blocks,
                 Array<IndexType> &block_pointers) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BLOCK_JACOBI_FIND_BLOCKS_KERNEL);


namespace {


template <typename ValueType, typename IndexType>
inline void extract_block(const matrix::Csr<ValueType, IndexType> *mtx,
                          IndexType block_size, IndexType block_start,
                          ValueType *block, size_type padding)
{
    for (int i = 0; i < block_size; ++i) {
        for (int j = 0; j < block_size; ++j) {
            block[i * padding + j] = zero<ValueType>();
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
                block[row * padding + col] = vals[i];
            }
        }
    }
}


template <typename ValueType, typename IndexType>
inline IndexType choose_pivot(IndexType block_size, const ValueType *block,
                              size_type padding)
{
    using std::abs;
    IndexType cp = 0;
    for (IndexType i = 1; i < block_size; ++i) {
        if (abs(block[cp * padding]) < abs(block[i * padding])) {
            cp = i;
        }
    }
    return cp;
}


template <typename ValueType, typename IndexType>
inline void swap_rows(IndexType row1, IndexType row2, IndexType block_size,
                      ValueType *block, size_type padding)
{
    using std::swap;
    for (IndexType i = 0; i < block_size; ++i) {
        swap(block[row1 * padding + i], block[row2 * padding + i]);
    }
}


template <typename ValueType, typename IndexType>
inline void apply_gauss_jordan_transform(IndexType row, IndexType col,
                                         IndexType block_size, ValueType *block,
                                         size_type padding)
{
    const auto d = block[row * padding + col];
    for (IndexType i = 0; i < block_size; ++i) {
        block[i * padding + col] /= -d;
    }
    block[row * padding + col] = zero<ValueType>();
    for (IndexType i = 0; i < block_size; ++i) {
        for (IndexType j = 0; j < block_size; ++j) {
            block[i * padding + j] +=
                block[i * padding + col] * block[row * padding + j];
        }
    }
    for (IndexType j = 0; j < block_size; ++j) {
        block[row * padding + j] /= d;
    }
    block[row * padding + col] = one<ValueType>() / d;
}


template <typename ValueType, typename IndexType>
inline void permute_columns(const IndexType *perm, IndexType block_size,
                            ValueType *block, size_type padding)
{
    std::vector<ValueType> tmp(block_size);
    for (IndexType i = 0; i < block_size; ++i) {
        for (IndexType j = 0; j < block_size; ++j) {
            tmp[perm[j]] = block[i * padding + j];
        }
        for (IndexType j = 0; j < block_size; ++j) {
            block[i * padding + j] = tmp[j];
        }
    }
}


template <typename ValueType, typename IndexType>
inline void invert_block(IndexType block_size, ValueType *block,
                         size_type padding)
{
    using std::abs;
    using std::swap;
    std::vector<IndexType> piv(block_size);
    iota(begin(piv), end(piv), IndexType(0));
    for (IndexType k = 0; k < block_size; ++k) {
        const auto cp =
            choose_pivot(block_size - k, block + k * padding + k, padding) + k;
        swap_rows(k, cp, block_size, block, padding);
        swap(piv[k], piv[cp]);
        apply_gauss_jordan_transform(k, k, block_size, block, padding);
    }
    permute_columns(piv.data(), block_size, block, padding);
}


}  // namespace


template <typename ValueType, typename IndexType>
void generate(const matrix::Csr<ValueType, IndexType> *system_matrix,
              size_type num_blocks, uint32 max_block_size, size_type padding,
              const Array<IndexType> &block_pointers, Array<ValueType> &blocks)
{
    const auto ptrs = block_pointers.get_const_data();
    for (size_type b = 0; b < num_blocks; ++b) {
        const auto block = blocks.get_data() + padding * ptrs[b];
        const auto block_size = ptrs[b + 1] - ptrs[b];
        extract_block(system_matrix, block_size, ptrs[b], block, padding);
        invert_block(block_size, block, padding);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BLOCK_JACOBI_GENERATE_KERNEL);


namespace {


template <typename ValueType>
void apply_block(size_type block_size, size_type num_rhs,
                 const ValueType *block, size_type padding, ValueType alpha,
                 const ValueType *b, size_type padding_b, ValueType beta,
                 ValueType *x, size_type padding_x)
{
    if (beta != zero<ValueType>()) {
        for (size_type row = 0; row < block_size; ++row) {
            for (size_type col = 0; col < num_rhs; ++col) {
                x[row * padding_x + col] *= beta;
            }
        }
    } else {
        for (size_type row = 0; row < block_size; ++row) {
            for (size_type col = 0; col < num_rhs; ++col) {
                x[row * padding_x + col] = zero<ValueType>();
            }
        }
    }

    for (size_type row = 0; row < block_size; ++row) {
        for (size_type inner = 0; inner < block_size; ++inner) {
            for (size_type col = 0; col < num_rhs; ++col) {
                x[row * padding_x + col] += alpha *
                                            block[row * padding + inner] *
                                            b[inner * padding_b + col];
            }
        }
    }
}


}  // namespace


template <typename ValueType, typename IndexType>
void apply(size_type num_blocks, uint32 max_block_size, size_type padding,
           const Array<IndexType> &block_pointers,
           const Array<ValueType> &blocks,
           const matrix::Dense<ValueType> *alpha,
           const matrix::Dense<ValueType> *b,
           const matrix::Dense<ValueType> *beta, matrix::Dense<ValueType> *x)
{
    const auto ptrs = block_pointers.get_const_data();
    for (size_type i = 0; i < num_blocks; ++i) {
        const auto block = blocks.get_const_data() + padding * ptrs[i];
        const auto block_b = b->get_const_values() + b->get_padding() * ptrs[i];
        const auto block_x = x->get_values() + x->get_padding() * ptrs[i];
        const auto block_size = ptrs[i + 1] - ptrs[i];
        apply_block(block_size, b->get_num_cols(), block, padding,
                    alpha->at(0, 0), block_b, b->get_padding(), beta->at(0, 0),
                    block_x, x->get_padding());
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BLOCK_JACOBI_APPLY_KERNEL);


template <typename ValueType, typename IndexType>
void simple_apply(size_type num_blocks, uint32 max_block_size,
                  size_type padding, const Array<IndexType> &block_pointers,
                  const Array<ValueType> &blocks,
                  const matrix::Dense<ValueType> *b,
                  matrix::Dense<ValueType> *x)
{
    const auto ptrs = block_pointers.get_const_data();
    for (size_type i = 0; i < num_blocks; ++i) {
        const auto block = blocks.get_const_data() + padding * ptrs[i];
        const auto block_b = b->get_const_values() + b->get_padding() * ptrs[i];
        const auto block_x = x->get_values() + x->get_padding() * ptrs[i];
        const auto block_size = ptrs[i + 1] - ptrs[i];
        apply_block(block_size, b->get_num_cols(), block, padding,
                    one<ValueType>(), block_b, b->get_padding(),
                    zero<ValueType>(), block_x, x->get_padding());
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BLOCK_JACOBI_SIMPLE_APPLY_KERNEL);


}  // namespace block_jacobi
}  // namespace reference
}  // namespace kernels
}  // namespace gko
