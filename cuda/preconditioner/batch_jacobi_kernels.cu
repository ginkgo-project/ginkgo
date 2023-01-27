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

#include "core/preconditioner/batch_jacobi_kernels.hpp"


#include <ginkgo/core/matrix/batch_csr.hpp>
#include <ginkgo/core/matrix/batch_ell.hpp>


#include "core/matrix/batch_struct.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "cuda/base/config.hpp"
#include "cuda/base/exception.cuh"
#include "cuda/base/types.hpp"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/intrinsics.cuh"
#include "cuda/components/thread_ids.cuh"
#include "cuda/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace cuda {
namespace batch_jacobi {

namespace {

constexpr int default_block_size = 128;

#include "common/cuda_hip/components/uninitialized_array.hpp.inc"
#include "common/cuda_hip/preconditioner/batch_block_jacobi.hpp.inc"
#include "common/cuda_hip/preconditioner/batch_scalar_jacobi.hpp.inc"
// Note: Do not change the ordering
#include "common/cuda_hip/preconditioner/batch_jacobi_kernels.hpp.inc"
}  // namespace


template <typename ValueType, typename IndexType>
void batch_jacobi_apply(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* const sys_mat,
    const size_type num_blocks, const uint32 max_block_size,
    const gko::preconditioner::batched_blocks_storage_scheme& storage_scheme,
    const ValueType* const blocks_array, const IndexType* const block_ptrs,
    const IndexType* const row_part_of_which_block_info,
    const matrix::BatchDense<ValueType>* const r,
    matrix::BatchDense<ValueType>* const z)
{
    const auto a_ub = get_batch_struct(sys_mat);
    batch_jacobi_apply_helper(a_ub, num_blocks, max_block_size, storage_scheme,
                              blocks_array, block_ptrs,
                              row_part_of_which_block_info, r, z);

    GKO_CUDA_LAST_IF_ERROR_THROW;
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_JACOBI_APPLY_KERNEL);

template <typename ValueType, typename IndexType>
void batch_jacobi_apply(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::BatchEll<ValueType, IndexType>* const sys_mat,
    const size_type num_blocks, const uint32 max_block_size,
    const gko::preconditioner::batched_blocks_storage_scheme& storage_scheme,
    const ValueType* const blocks_array, const IndexType* const block_ptrs,
    const IndexType* const row_part_of_which_block_info,
    const matrix::BatchDense<ValueType>* const r,
    matrix::BatchDense<ValueType>* const z)
{
    const auto a_ub = get_batch_struct(sys_mat);
    batch_jacobi_apply_helper(a_ub, num_blocks, max_block_size, storage_scheme,
                              blocks_array, block_ptrs,
                              row_part_of_which_block_info, r, z);
    GKO_CUDA_LAST_IF_ERROR_THROW;
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_JACOBI_ELL_APPLY_KERNEL);


template <typename ValueType, typename IndexType>
void extract_common_blocks_pattern(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* const first_sys_csr,
    const size_type num_blocks,
    const preconditioner::batched_blocks_storage_scheme& storage_scheme,
    const IndexType* const block_pointers,
    const IndexType* const row_part_of_which_block_info,
    IndexType* const blocks_pattern)
{
    const auto nrows = first_sys_csr->get_size()[0];
    dim3 block(default_block_size);
    dim3 grid(ceildiv(nrows * config::warp_size, default_block_size));

    extract_common_block_pattern_kernel<<<grid, block>>>(
        static_cast<int>(nrows), first_sys_csr->get_const_row_ptrs(),
        first_sys_csr->get_const_col_idxs(), num_blocks, storage_scheme,
        block_pointers, row_part_of_which_block_info, blocks_pattern);

    GKO_CUDA_LAST_IF_ERROR_THROW;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_BLOCK_JACOBI_EXTRACT_PATTERN_KERNEL);


// DECLARE_BATCH_BLOCK_JACOBI_COMPUTE_INSTANTIATION(ValueType, IndexType);
// template <int subwarp_size, typename ValueType, typename IndexType>
// void compute_block_jacobi_helper(
//     const matrix::BatchCsr<ValueType, IndexType>* const sys_csr,
//     const size_type num_blocks,
//     const preconditioner::batched_blocks_storage_scheme& storage_scheme,
//     const IndexType* const block_pointers,
//     const IndexType* const blocks_pattern, ValueType* const blocks);


namespace {

constexpr int get_larger_power(int value, int guess = 1)
{
    return guess >= value ? guess : get_larger_power(value, guess << 1);
}

template <int compiled_max_block_size, typename ValueType, typename IndexType>
void compute_block_jacobi_helper(
    syn::value_list<int, compiled_max_block_size>,
    const matrix::BatchCsr<ValueType, IndexType>* const sys_csr,
    const size_type num_blocks,
    const preconditioner::batched_blocks_storage_scheme& storage_scheme,
    const IndexType* const block_pointers,
    const IndexType* const blocks_pattern, ValueType* const blocks)
{
    constexpr int subwarp_size = get_larger_power(compiled_max_block_size);
    std::cout << "\nCompiled max block size: " << compiled_max_block_size
              << "  and subwarp size: " << subwarp_size;
    const auto nbatch = sys_csr->get_num_batch_entries();
    const auto nrows = sys_csr->get_size().at(0)[0];
    const auto nnz = sys_csr->get_num_stored_elements() / nbatch;

    dim3 block(default_block_size);
    dim3 grid(ceildiv(num_blocks * nbatch * subwarp_size, default_block_size));

    compute_block_jacobi_kernel<subwarp_size><<<grid, block>>>(
        nbatch, static_cast<int>(nnz),
        as_cuda_type(sys_csr->get_const_values()), num_blocks, storage_scheme,
        block_pointers, blocks_pattern, as_cuda_type(blocks));

    GKO_CUDA_LAST_IF_ERROR_THROW;
}


GKO_ENABLE_IMPLEMENTATION_SELECTION(select_compute_block_jacobi_helper,
                                    compute_block_jacobi_helper);

// template <int max_block_size, typename ValueType, typename IndexType>
// void compute_block_jacobi_helper(
//     const matrix::BatchCsr<ValueType, IndexType>* const sys_csr,
//     const size_type num_blocks,
//     const preconditioner::batched_blocks_storage_scheme& storage_scheme,
//     const IndexType* const block_pointers,
//     const IndexType* const blocks_pattern, ValueType* const blocks)
// {
//     constexpr int subwarp_size = get_larger_power(max_block_size);
//     const auto nbatch = sys_csr->get_num_batch_entries();
//     const auto nrows = sys_csr->get_size().at(0)[0];
//     const auto nnz = sys_csr->get_num_stored_elements() / nbatch;

//     dim3 block(default_block_size);
//     dim3 grid(ceildiv(num_blocks * nbatch * subwarp_size,
//     default_block_size));

//     compute_block_jacobi_kernel<subwarp_size><<<grid, block>>>(
//         nbatch, static_cast<int>(nnz),
//         as_cuda_type(sys_csr->get_const_values()), num_blocks,
//         storage_scheme, block_pointers, blocks_pattern,
//         as_cuda_type(blocks));

//     GKO_CUDA_LAST_IF_ERROR_THROW;
// }


// template <typename ValueType, typename IndexType>
// void select_compute_block_jacobi_helper(
//     const matrix::BatchCsr<ValueType, IndexType>* const sys_csr,
//     const uint32 max_block_size, const size_type num_blocks,
//     const preconditioner::batched_blocks_storage_scheme& storage_scheme,
//     const IndexType* const block_pointers,
//     const IndexType* const blocks_pattern, ValueType* const blocks)
// {

//     switch(max_block_size){

//         case 1:
//             compute_block_jacobi_helper<1>(sys_csr, num_blocks,
//             storage_scheme,block_pointers, blocks_pattern, blocks); break;

//         case 2:
//             compute_block_jacobi_helper<2>(sys_csr, num_blocks,
//             storage_scheme,block_pointers, blocks_pattern, blocks); break;

//         case 3:
//             compute_block_jacobi_helper<3>(sys_csr, num_blocks,
//             storage_scheme,block_pointers, blocks_pattern, blocks); break;

//         case 4:
//             compute_block_jacobi_helper<4>(sys_csr, num_blocks,
//             storage_scheme,block_pointers, blocks_pattern, blocks); break;

//         case 5:
//             compute_block_jacobi_helper<5>(sys_csr, num_blocks,
//             storage_scheme,block_pointers, blocks_pattern, blocks); break;

//         case 6:
//             compute_block_jacobi_helper<6>(sys_csr, num_blocks,
//             storage_scheme,block_pointers, blocks_pattern, blocks); break;

//         case 7:
//             compute_block_jacobi_helper<7>(sys_csr, num_blocks,
//             storage_scheme,block_pointers, blocks_pattern, blocks); break;

//         case 8:
//             compute_block_jacobi_helper<8>(sys_csr, num_blocks,
//             storage_scheme,block_pointers, blocks_pattern, blocks); break;

//         case 9:
//             compute_block_jacobi_helper<9>(sys_csr, num_blocks,
//             storage_scheme,block_pointers, blocks_pattern, blocks); break;

//         case 10:
//             compute_block_jacobi_helper<10>(sys_csr, num_blocks,
//             storage_scheme,block_pointers, blocks_pattern, blocks); break;

//         case 11:
//             compute_block_jacobi_helper<11>(sys_csr, num_blocks,
//             storage_scheme,block_pointers, blocks_pattern, blocks); break;

//         case 12:
//             compute_block_jacobi_helper<12>(sys_csr, num_blocks,
//             storage_scheme,block_pointers, blocks_pattern, blocks); break;

//         case 13:
//             compute_block_jacobi_helper<13>(sys_csr, num_blocks,
//             storage_scheme,block_pointers, blocks_pattern, blocks); break;

//         case 14:
//             compute_block_jacobi_helper<14>(sys_csr, num_blocks,
//             storage_scheme,block_pointers, blocks_pattern, blocks); break;

//         case 15:
//             compute_block_jacobi_helper<15>(sys_csr, num_blocks,
//             storage_scheme,block_pointers, blocks_pattern, blocks); break;

//         case 16:
//             compute_block_jacobi_helper<16>(sys_csr, num_blocks,
//             storage_scheme,block_pointers, blocks_pattern, blocks); break;

//         case 17:
//             compute_block_jacobi_helper<17>(sys_csr, num_blocks,
//             storage_scheme,block_pointers, blocks_pattern, blocks); break;

//         case 18:
//             compute_block_jacobi_helper<18>(sys_csr, num_blocks,
//             storage_scheme,block_pointers, blocks_pattern, blocks); break;

//         case 19:
//             compute_block_jacobi_helper<19>(sys_csr, num_blocks,
//             storage_scheme,block_pointers, blocks_pattern, blocks); break;

//         case 20:
//             compute_block_jacobi_helper<20>(sys_csr, num_blocks,
//             storage_scheme,block_pointers, blocks_pattern, blocks); break;

//         case 21:
//             compute_block_jacobi_helper<21>(sys_csr, num_blocks,
//             storage_scheme,block_pointers, blocks_pattern, blocks); break;

//         case 22:
//             compute_block_jacobi_helper<22>(sys_csr, num_blocks,
//             storage_scheme,block_pointers, blocks_pattern, blocks); break;

//         case 23:
//             compute_block_jacobi_helper<23>(sys_csr, num_blocks,
//             storage_scheme,block_pointers, blocks_pattern, blocks); break;

//         case 24:
//             compute_block_jacobi_helper<24>(sys_csr, num_blocks,
//             storage_scheme,block_pointers, blocks_pattern, blocks); break;

//         case 25:
//             compute_block_jacobi_helper<25>(sys_csr, num_blocks,
//             storage_scheme,block_pointers, blocks_pattern, blocks); break;

//         case 26:
//             compute_block_jacobi_helper<26>(sys_csr, num_blocks,
//             storage_scheme,block_pointers, blocks_pattern, blocks); break;

//         case 27:
//             compute_block_jacobi_helper<27>(sys_csr, num_blocks,
//             storage_scheme,block_pointers, blocks_pattern, blocks); break;

//         case 28:
//             compute_block_jacobi_helper<28>(sys_csr, num_blocks,
//             storage_scheme,block_pointers, blocks_pattern, blocks); break;

//         case 29:
//             compute_block_jacobi_helper<29>(sys_csr, num_blocks,
//             storage_scheme,block_pointers, blocks_pattern, blocks); break;

//         case 30:
//             compute_block_jacobi_helper<30>(sys_csr, num_blocks,
//             storage_scheme,block_pointers, blocks_pattern, blocks); break;

//         case 31:
//             compute_block_jacobi_helper<31>(sys_csr, num_blocks,
//             storage_scheme,block_pointers, blocks_pattern, blocks); break;

//         case 32:
//             compute_block_jacobi_helper<32>(sys_csr, num_blocks,
//             storage_scheme,block_pointers, blocks_pattern, blocks); break;
//     }
// }

}  // anonymous namespace


template <typename ValueType, typename IndexType>
void compute_block_jacobi(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* const sys_csr,
    const uint32 user_given_max_block_size, const size_type num_blocks,
    const preconditioner::batched_blocks_storage_scheme& storage_scheme,
    const IndexType* const block_pointers,
    const IndexType* const blocks_pattern, ValueType* const blocks)
{
    using batch_jacobi_compiled_max_block_sizes =
        syn::value_list<int, 2, 4, 8, 13, 16, 32>;
    select_compute_block_jacobi_helper(
        batch_jacobi_compiled_max_block_sizes(),
        [&](int compiled_block_size) {
            return user_given_max_block_size <= compiled_block_size;
        },
        syn::value_list<int>(), syn::type_list<>(), sys_csr, num_blocks,
        storage_scheme, block_pointers, blocks_pattern, blocks);

    // select_compute_block_jacobi_helper(sys_csr, max_block_size, num_blocks,
    //                                    storage_scheme, block_pointers,
    //                                    blocks_pattern, blocks);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_BLOCK_JACOBI_COMPUTE_KERNEL);

}  // namespace batch_jacobi
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
