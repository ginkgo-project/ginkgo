// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/preconditioner/jacobi_kernels.hpp"


#include <algorithm>
#include <cmath>
#include <iterator>
#include <numeric>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/allocator.hpp"
#include "core/base/extended_float.hpp"
#include "core/preconditioner/jacobi_utils.hpp"
#include "reference/components/matrix_operations.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The Jacobi preconditioner namespace.
 *
 * @ingroup jacobi
 */
namespace jacobi {
namespace {


template <typename IndexType>
inline bool has_same_nonzero_pattern(const IndexType* prev_row_ptr,
                                     const IndexType* curr_row_ptr,
                                     const IndexType* next_row_ptr)
{
    return std::distance(curr_row_ptr, next_row_ptr) ==
               std::distance(prev_row_ptr, curr_row_ptr) &&
           std::equal(curr_row_ptr, next_row_ptr, prev_row_ptr);
}


template <typename ValueType, typename IndexType>
size_type find_natural_blocks(const matrix::Csr<ValueType, IndexType>* mtx,
                              uint32 max_block_size, IndexType* block_ptrs)
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
                                            IndexType* block_ptrs)
{
    if (num_natural_blocks == 0) {
        return 0;
    }
    size_type num_blocks = 1;
    auto current_block_size = block_ptrs[1] - block_ptrs[0];
    for (size_type i = 1; i < num_natural_blocks; ++i) {
        const auto block_size = block_ptrs[i + 1] - block_ptrs[i];
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
void find_blocks(std::shared_ptr<const DefaultExecutor> exec,
                 const matrix::Csr<ValueType, IndexType>* system_matrix,
                 uint32 max_block_size, size_type& num_blocks,
                 array<IndexType>& block_pointers)
{
    num_blocks = find_natural_blocks(system_matrix, max_block_size,
                                     block_pointers.get_data());
    num_blocks = agglomerate_supervariables(max_block_size, num_blocks,
                                            block_pointers.get_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_JACOBI_FIND_BLOCKS_KERNEL);


namespace {


template <typename ValueType, typename IndexType>
inline void extract_block(const matrix::Csr<ValueType, IndexType>* mtx,
                          IndexType block_size, IndexType block_start,
                          ValueType* block, size_type stride)
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
        for (auto i = start; i < end; ++i) {
            const auto col = col_idxs[i] - block_start;
            if (0 <= col && col < block_size) {
                block[row * stride + col] = vals[i];
            }
        }
    }
}


template <typename ValueType, typename IndexType>
inline IndexType choose_pivot(IndexType block_size, const ValueType* block,
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
                      ValueType* block, size_type stride)
{
    using std::swap;
    for (IndexType i = 0; i < block_size; ++i) {
        swap(block[row1 * stride + i], block[row2 * stride + i]);
    }
}


template <typename ValueType, typename IndexType>
inline bool apply_gauss_jordan_transform(IndexType row, IndexType col,
                                         IndexType block_size, ValueType* block,
                                         size_type stride)
{
    const auto d = block[row * stride + col];
    if (is_zero(d)) {
        return false;
    }
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
    return true;
}


template <typename SourceValueType, typename ResultValueType,
          typename IndexType,
          typename ValueConverter =
              default_converter<SourceValueType, ResultValueType>>
inline void transpose_block(IndexType block_size, const SourceValueType* from,
                            size_type from_stride, ResultValueType* to,
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
inline void conj_transpose_block(IndexType block_size,
                                 const SourceValueType* from,
                                 size_type from_stride, ResultValueType* to,
                                 size_type to_stride,
                                 ValueConverter converter = {}) noexcept
{
    for (IndexType i = 0; i < block_size; ++i) {
        for (IndexType j = 0; j < block_size; ++j) {
            to[i * to_stride + j] = conj(converter(from[i + j * from_stride]));
        }
    }
}


template <typename SourceValueType, typename ResultValueType,
          typename IndexType,
          typename ValueConverter =
              default_converter<SourceValueType, ResultValueType>>
inline void permute_and_transpose_block(IndexType block_size,
                                        const IndexType* col_perm,
                                        const SourceValueType* source,
                                        size_type source_stride,
                                        ResultValueType* result,
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
inline bool invert_block(IndexType block_size, IndexType* perm,
                         ValueType* block, size_type stride)
{
    using std::swap;
    for (IndexType k = 0; k < block_size; ++k) {
        const auto cp =
            choose_pivot(block_size - k, block + k * stride + k, stride) + k;
        swap_rows(k, cp, block_size, block, stride);
        swap(perm[k], perm[cp]);
        auto status =
            apply_gauss_jordan_transform(k, k, block_size, block, stride);
        if (!status) {
            return false;
        }
    }
    return true;
}


template <typename ReducedType, typename ValueType, typename IndexType>
inline bool validate_precision_reduction_feasibility(
    std::shared_ptr<const DefaultExecutor> exec, IndexType block_size,
    const ValueType* block, size_type stride)
{
    using gko::detail::float_traits;
    vector<ValueType> tmp(block_size * block_size, {}, exec);
    vector<IndexType> perm(block_size, {}, exec);
    std::iota(begin(perm), end(perm), IndexType{0});
    for (IndexType i = 0; i < block_size; ++i) {
        for (IndexType j = 0; j < block_size; ++j) {
            tmp[i * block_size + j] = static_cast<ValueType>(
                static_cast<ReducedType>(block[i * stride + j]));
        }
    }
    auto cond =
        compute_inf_norm(block_size, block_size, tmp.data(), block_size);
    auto succeeded =
        invert_block(block_size, perm.data(), tmp.data(), block_size);
    if (!succeeded) {
        return false;
    }
    cond *= compute_inf_norm(block_size, block_size, tmp.data(), block_size);
    return cond >= 1.0 &&
           cond * static_cast<remove_complex<ValueType>>(
                      float_traits<remove_complex<ValueType>>::eps) <
               remove_complex<ValueType>{1e-3};
}


}  // namespace


template <typename ValueType, typename IndexType>
void generate(std::shared_ptr<const DefaultExecutor> exec,
              const matrix::Csr<ValueType, IndexType>* system_matrix,
              size_type num_blocks, uint32 max_block_size,
              remove_complex<ValueType> accuracy,
              const preconditioner::block_interleaved_storage_scheme<IndexType>&
                  storage_scheme,
              array<remove_complex<ValueType>>& conditioning,
              array<precision_reduction>& block_precisions,
              const array<IndexType>& block_pointers, array<ValueType>& blocks)
{
    const auto ptrs = block_pointers.get_const_data();
    const auto prec = block_precisions.get_data();
    const auto group_size = storage_scheme.get_group_size();
    const auto cond = conditioning.get_data();
    for (size_type g = 0; g < num_blocks; g += group_size) {
        vector<array<ValueType>> block(group_size, {}, exec);
        vector<array<IndexType>> perm(group_size, {}, exec);
        vector<uint32> pr_descriptors(group_size, uint32{} - 1, exec);
        // extract group of blocks, invert them, figure out storage precision
        for (IndexType b = 0; b < group_size; ++b) {
            if (b + g >= num_blocks) {
                break;
            }
            const auto block_size = ptrs[g + b + 1] - ptrs[g + b];
            block[b] = array<ValueType>(exec, block_size * block_size);
            perm[b] = array<IndexType>(exec, block_size);
            std::iota(perm[b].get_data(), perm[b].get_data() + block_size,
                      IndexType(0));
            extract_block(system_matrix, block_size, ptrs[g + b],
                          block[b].get_data(), block_size);
            if (cond) {
                cond[g + b] =
                    compute_inf_norm(block_size, block_size,
                                     block[b].get_const_data(), block_size);
            }
            invert_block(block_size, perm[b].get_data(), block[b].get_data(),
                         block_size);
            if (cond) {
                cond[g + b] *=
                    compute_inf_norm(block_size, block_size,
                                     block[b].get_const_data(), block_size);
            }
            const auto local_prec = prec ? prec[g + b] : precision_reduction();
            if (local_prec == precision_reduction::autodetect() && cond) {
                using preconditioner::detail::get_supported_storage_reductions;
                pr_descriptors[b] = get_supported_storage_reductions<ValueType>(
                    accuracy, cond[g + b],
                    [&exec, &block_size, &block, &b] {
                        using target = reduce_precision<ValueType>;
                        return validate_precision_reduction_feasibility<target>(
                            exec, block_size, block[b].get_const_data(),
                            block_size);
                    },
                    [&exec, &block_size, &block, &b] {
                        using target =
                            reduce_precision<reduce_precision<ValueType>>;
                        return validate_precision_reduction_feasibility<target>(
                            exec, block_size, block[b].get_const_data(),
                            block_size);
                    });
            } else {
                pr_descriptors[b] = preconditioner::detail::
                    precision_reduction_descriptor::singleton(local_prec);
            }
        }

        // make sure everyone in the group uses the same precision
        const auto p = preconditioner::detail::get_optimal_storage_reduction(
            std::accumulate(begin(pr_descriptors), end(pr_descriptors),
                            uint32{} - 1,
                            [](uint32 x, uint32 y) { return x & y; }));

        // store the blocks
        for (IndexType b = 0; b < group_size; ++b) {
            if (b + g >= num_blocks) {
                break;
            }
            if (prec) {
                prec[g + b] = p;
            }
            const auto block_size = ptrs[g + b + 1] - ptrs[g + b];
            GKO_PRECONDITIONER_JACOBI_RESOLVE_PRECISION(
                ValueType, p,
                permute_and_transpose_block(
                    block_size, perm[b].get_data(), block[b].get_data(),
                    block_size,
                    reinterpret_cast<resolved_precision*>(
                        blocks.get_data() +
                        storage_scheme.get_group_offset(g + b)) +
                        storage_scheme.get_block_offset(g + b),
                    storage_scheme.get_stride()));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_JACOBI_GENERATE_KERNEL);


namespace {


template <
    typename ValueType, typename BlockValueType,
    typename ValueConverter = default_converter<BlockValueType, ValueType>>
inline void apply_block(size_type block_size, size_type num_rhs,
                        const BlockValueType* block, size_type stride,
                        ValueType alpha, const ValueType* b, size_type stride_b,
                        ValueType beta, ValueType* x, size_type stride_x,
                        ValueConverter converter = {})
{
    if (is_nonzero(beta)) {
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


void initialize_precisions(std::shared_ptr<const DefaultExecutor> exec,
                           const array<precision_reduction>& source,
                           array<precision_reduction>& precisions)
{
    const auto source_size = source.get_size();
    for (auto i = 0u; i < precisions.get_size(); ++i) {
        precisions.get_data()[i] = source.get_const_data()[i % source_size];
    }
}


template <typename ValueType, typename IndexType>
void apply(std::shared_ptr<const DefaultExecutor> exec, size_type num_blocks,
           uint32 max_block_size,
           const preconditioner::block_interleaved_storage_scheme<IndexType>&
               storage_scheme,
           const array<precision_reduction>& block_precisions,
           const array<IndexType>& block_pointers,
           const array<ValueType>& blocks,
           const matrix::Dense<ValueType>* alpha,
           const matrix::Dense<ValueType>* b,
           const matrix::Dense<ValueType>* beta, matrix::Dense<ValueType>* x)
{
    const auto ptrs = block_pointers.get_const_data();
    const auto prec = block_precisions.get_const_data();
    for (size_type i = 0; i < num_blocks; ++i) {
        const auto group =
            blocks.get_const_data() + storage_scheme.get_group_offset(i);
        const auto block_b = b->get_const_values() + b->get_stride() * ptrs[i];
        const auto block_x = x->get_values() + x->get_stride() * ptrs[i];
        const auto block_size = ptrs[i + 1] - ptrs[i];
        const auto p = prec ? prec[i] : precision_reduction();
        GKO_PRECONDITIONER_JACOBI_RESOLVE_PRECISION(
            ValueType, p,
            apply_block(block_size, b->get_size()[1],
                        reinterpret_cast<const resolved_precision*>(group) +
                            storage_scheme.get_block_offset(i),
                        storage_scheme.get_stride(), alpha->at(0, 0), block_b,
                        b->get_stride(), beta->at(0, 0), block_x,
                        x->get_stride()));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_JACOBI_APPLY_KERNEL);


template <typename ValueType, typename IndexType>
void simple_apply(
    std::shared_ptr<const DefaultExecutor> exec, size_type num_blocks,
    uint32 max_block_size,
    const preconditioner::block_interleaved_storage_scheme<IndexType>&
        storage_scheme,
    const array<precision_reduction>& block_precisions,
    const array<IndexType>& block_pointers, const array<ValueType>& blocks,
    const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* x)
{
    const auto ptrs = block_pointers.get_const_data();
    const auto prec = block_precisions.get_const_data();
    for (size_type i = 0; i < num_blocks; ++i) {
        const auto group =
            blocks.get_const_data() + storage_scheme.get_group_offset(i);
        const auto block_b = b->get_const_values() + b->get_stride() * ptrs[i];
        const auto block_x = x->get_values() + x->get_stride() * ptrs[i];
        const auto block_size = ptrs[i + 1] - ptrs[i];
        const auto p = prec ? prec[i] : precision_reduction();
        GKO_PRECONDITIONER_JACOBI_RESOLVE_PRECISION(
            ValueType, p,
            apply_block(block_size, b->get_size()[1],
                        reinterpret_cast<const resolved_precision*>(group) +
                            storage_scheme.get_block_offset(i),
                        storage_scheme.get_stride(), one<ValueType>(), block_b,
                        b->get_stride(), zero<ValueType>(), block_x,
                        x->get_stride()));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_JACOBI_SIMPLE_APPLY_KERNEL);


template <typename ValueType>
void scalar_apply(std::shared_ptr<const DefaultExecutor> exec,
                  const array<ValueType>& diag,
                  const matrix::Dense<ValueType>* alpha,
                  const matrix::Dense<ValueType>* b,
                  const matrix::Dense<ValueType>* beta,
                  matrix::Dense<ValueType>* x)
{
    for (size_type i = 0; i < x->get_size()[0]; ++i) {
        for (size_type j = 0; j < x->get_size()[1]; ++j) {
            x->at(i, j) = beta->at(0) * x->at(i, j) +
                          alpha->at(0) * b->at(i, j) * diag.get_const_data()[i];
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_JACOBI_SCALAR_APPLY_KERNEL);


template <typename ValueType>
void simple_scalar_apply(std::shared_ptr<const DefaultExecutor> exec,
                         const array<ValueType>& diag,
                         const matrix::Dense<ValueType>* b,
                         matrix::Dense<ValueType>* x)
{
    for (size_type i = 0; i < x->get_size()[0]; ++i) {
        for (size_type j = 0; j < x->get_size()[1]; ++j) {
            x->at(i, j) = b->at(i, j) * diag.get_const_data()[i];
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_JACOBI_SIMPLE_SCALAR_APPLY_KERNEL);


template <typename ValueType>
void scalar_conj(std::shared_ptr<const DefaultExecutor> exec,
                 const array<ValueType>& diag, array<ValueType>& conj_diag)
{
    for (size_type i = 0; i < diag.get_size(); ++i) {
        conj_diag.get_data()[i] = conj(diag.get_const_data()[i]);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_JACOBI_SCALAR_CONJ_KERNEL);


template <typename ValueType>
void invert_diagonal(std::shared_ptr<const DefaultExecutor> exec,
                     const array<ValueType>& diag, array<ValueType>& inv_diag)
{
    for (size_type i = 0; i < diag.get_size(); ++i) {
        auto diag_val = is_zero(diag.get_const_data()[i])
                            ? one<ValueType>()
                            : diag.get_const_data()[i];
        inv_diag.get_data()[i] = one<ValueType>() / diag_val;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_JACOBI_INVERT_DIAGONAL_KERNEL);


template <typename ValueType, typename IndexType>
void transpose_jacobi(
    std::shared_ptr<const DefaultExecutor> exec, size_type num_blocks,
    uint32 max_block_size, const array<precision_reduction>& block_precisions,
    const array<IndexType>& block_pointers, const array<ValueType>& blocks,
    const preconditioner::block_interleaved_storage_scheme<IndexType>&
        storage_scheme,
    array<ValueType>& out_blocks)
{
    const auto ptrs = block_pointers.get_const_data();
    const auto prec = block_precisions.get_const_data();
    const size_type matrix_size = ptrs[num_blocks];

    for (size_type i = 0; i < num_blocks; ++i) {
        const auto group_ofs = storage_scheme.get_group_offset(i);
        const auto block_ofs = storage_scheme.get_block_offset(i);
        const auto block_stride = storage_scheme.get_stride();
        const auto group = blocks.get_const_data() + group_ofs;
        auto out_group = out_blocks.get_data() + group_ofs;
        const auto block_size = ptrs[i + 1] - ptrs[i];
        const auto p = prec ? prec[i] : precision_reduction();
        GKO_PRECONDITIONER_JACOBI_RESOLVE_PRECISION(
            ValueType, p,
            transpose_block(
                block_size,
                reinterpret_cast<const resolved_precision*>(group) + block_ofs,
                block_stride,
                reinterpret_cast<resolved_precision*>(out_group) + block_ofs,
                block_stride));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_JACOBI_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void conj_transpose_jacobi(
    std::shared_ptr<const DefaultExecutor> exec, size_type num_blocks,
    uint32 max_block_size, const array<precision_reduction>& block_precisions,
    const array<IndexType>& block_pointers, const array<ValueType>& blocks,
    const preconditioner::block_interleaved_storage_scheme<IndexType>&
        storage_scheme,
    array<ValueType>& out_blocks)
{
    const auto ptrs = block_pointers.get_const_data();
    const auto prec = block_precisions.get_const_data();
    const size_type matrix_size = ptrs[num_blocks];

    for (size_type i = 0; i < num_blocks; ++i) {
        const auto group_ofs = storage_scheme.get_group_offset(i);
        const auto block_ofs = storage_scheme.get_block_offset(i);
        const auto block_stride = storage_scheme.get_stride();
        const auto group = blocks.get_const_data() + group_ofs;
        auto out_group = out_blocks.get_data() + group_ofs;
        const auto block_size = ptrs[i + 1] - ptrs[i];
        const auto p = prec ? prec[i] : precision_reduction();
        GKO_PRECONDITIONER_JACOBI_RESOLVE_PRECISION(
            ValueType, p,
            conj_transpose_block(
                block_size,
                reinterpret_cast<const resolved_precision*>(group) + block_ofs,
                block_stride,
                reinterpret_cast<resolved_precision*>(out_group) + block_ofs,
                block_stride));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_JACOBI_CONJ_TRANSPOSE_KERNEL);


template <typename ValueType>
void scalar_convert_to_dense(std::shared_ptr<const DefaultExecutor> exec,
                             const array<ValueType>& blocks,
                             matrix::Dense<ValueType>* result)
{
    auto matrix_size = result->get_size();
    for (size_type i = 0; i < matrix_size[0]; ++i) {
        for (size_type j = 0; j < matrix_size[1]; ++j) {
            result->at(i, j) = zero<ValueType>();
            if (i == j) {
                result->at(i, j) = blocks.get_const_data()[i];
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_JACOBI_SCALAR_CONVERT_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_dense(
    std::shared_ptr<const DefaultExecutor> exec, size_type num_blocks,
    const array<precision_reduction>& block_precisions,
    const array<IndexType>& block_pointers, const array<ValueType>& blocks,
    const preconditioner::block_interleaved_storage_scheme<IndexType>&
        storage_scheme,
    ValueType* result_values, size_type result_stride)
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
        const auto group =
            blocks.get_const_data() + storage_scheme.get_group_offset(i);
        const auto block_size = ptrs[i + 1] - ptrs[i];
        const auto p = prec ? prec[i] : precision_reduction();
        GKO_PRECONDITIONER_JACOBI_RESOLVE_PRECISION(
            ValueType, p,
            transpose_block(block_size,
                            reinterpret_cast<const resolved_precision*>(group) +
                                storage_scheme.get_block_offset(i),
                            storage_scheme.get_stride(),
                            result_values + ptrs[i] * result_stride + ptrs[i],
                            result_stride));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_JACOBI_CONVERT_TO_DENSE_KERNEL);


}  // namespace jacobi
}  // namespace reference
}  // namespace kernels
}  // namespace gko
